"""Prove that resolution lost nothing, and put back anything it did.

Entity resolution is the one pipeline stage that is allowed to delete rows, so
it is the one stage where a bug looks exactly like success: fewer records out
than in is the point. The only defence is to check that every input is still
reachable from some output, and Abzu checks it at several independent points
rather than trusting any single one.

Abzu's checks report. ``original_coverage_pct`` has to clear 99.99,
``source_uuid_error_pct`` has to stay under 0.01, and a failure is logged and
the run continues. That is the right call for a pipeline whose repairs already
happened upstream, but it leaves the last gap unguarded, so the repair is done
here as well as reported: a record no output accounts for is added back as its
own output record before the report is written.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from serf.config import config
from serf.logs import get_logger

logger = get_logger(__name__)

MISSING_IN_OUTPUT_REASON = "missing_in_match_output"


@runtime_checkable
class HasLineage(Protocol):
    """Anything carrying an id and the ids it absorbed."""

    id: int
    source_ids: list[int] | None


def coverage_of(record: Any) -> set[int]:
    """Return every input id a record accounts for.

    Parameters
    ----------
    record : Any
        A mapping with ``id`` and optionally ``source_ids``, or an object with
        those attributes

    Returns
    -------
    set[int]
        The record's own id together with its lineage
    """
    if isinstance(record, dict):
        return {int(record["id"]), *(int(value) for value in record.get("source_ids") or [])}
    return {int(record.id), *(int(value) for value in record.source_ids or [])}


def _lineage_of(record: Any) -> list[int]:
    """Return the ids a record claims to have absorbed.

    Parameters
    ----------
    record : Any
        A mapping with ``source_ids``, or an object with that attribute

    Returns
    -------
    list[int]
        Lineage entries, excluding the record's own id
    """
    if isinstance(record, dict):
        return [int(value) for value in record.get("source_ids") or []]
    return [int(value) for value in record.source_ids or []]


@dataclass
class ConservationReport:
    """What happened to the records between the input and the output.

    Parameters
    ----------
    input_records : int
        Distinct input ids the run started from, lineage included
    output_records : int
        Records the run produced
    covered_ids : int
        Input ids reachable from some output record
    coverage_pct : float
        ``covered_ids`` as a percentage of ``input_records``
    missing_ids : list[int]
        Input ids no output record accounts for. Empty after recovery.
    invalid_reference_ids : list[int]
        Ids referenced by an output's ``source_ids`` that were never an input
    reference_error_pct : float
        ``invalid_reference_ids`` as a percentage of all lineage references
    duplicated_ids : list[int]
        Input ids accounted for by more than one output record, which means
        the same record is claimed by two entities
    recovered_records : int
        Records added back because nothing else accounted for them
    """

    input_records: int = 0
    output_records: int = 0
    covered_ids: int = 0
    coverage_pct: float = 100.0
    missing_ids: list[int] = field(default_factory=list)
    invalid_reference_ids: list[int] = field(default_factory=list)
    reference_error_pct: float = 0.0
    duplicated_ids: list[int] = field(default_factory=list)
    recovered_records: int = 0

    @property
    def coverage_passes(self) -> bool:
        """Whether coverage clears the configured threshold.

        Returns
        -------
        bool
            True when every input is reachable, to the configured tolerance
        """
        threshold = float(config.get("merge.conservation.coverage_threshold", 0.9999)) * 100
        return self.coverage_pct >= threshold

    @property
    def references_pass(self) -> bool:
        """Whether dangling lineage references stay under the threshold.

        Returns
        -------
        bool
            True when almost no source_id points at a record that never existed
        """
        threshold = float(config.get("merge.conservation.reference_error_threshold", 0.0001)) * 100
        return self.reference_error_pct <= threshold

    @property
    def passes(self) -> bool:
        """Whether every conservation check holds.

        Returns
        -------
        bool
            True when nothing was lost, nothing dangles and nothing is claimed
            twice
        """
        return self.coverage_passes and self.references_pass and not self.duplicated_ids

    def summary(self) -> str:
        """Render the report as one line for a log or a CLI.

        Returns
        -------
        str
            Coverage, references and duplication, with a verdict
        """
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"{verdict}: {self.covered_ids}/{self.input_records} input records reachable "
            f"({self.coverage_pct:.4f}%), {len(self.invalid_reference_ids)} dangling references "
            f"({self.reference_error_pct:.4f}%), {len(self.duplicated_ids)} claimed twice, "
            f"{self.recovered_records} recovered"
        )


def check_conservation(
    inputs: Iterable[Any],
    outputs: Iterable[Any],
    minted_ids: Iterable[int] | None = None,
) -> ConservationReport:
    """Report whether every input record is still reachable from the output.

    Parameters
    ----------
    inputs : Iterable[Any]
        Records the run started from
    outputs : Iterable[Any]
        Records the run produced
    minted_ids : Iterable[int] | None
        Ids the run itself created for merged entities. A multi-round run
        merges entities it merged earlier, so an intermediate entity's id
        legitimately appears in a later record's lineage. Without this the
        check reads those as references to records that never existed.

    Returns
    -------
    ConservationReport
        Coverage, dangling references and double-claimed records
    """
    input_ids: set[int] = set()
    for record in inputs:
        input_ids |= coverage_of(record)
    referenceable = input_ids | {int(value) for value in minted_ids or ()}

    output_list = list(outputs)
    claims: dict[int, int] = {}
    lineage_refs: list[int] = []
    for record in output_list:
        for record_id in coverage_of(record):
            claims[record_id] = claims.get(record_id, 0) + 1
        lineage_refs.extend(_lineage_of(record))

    covered = input_ids & set(claims)
    missing = sorted(input_ids - set(claims))
    # A merged record's own id is minted, so it is deliberately not an input
    # id. Only lineage entries are references, and only they can dangle.
    invalid = sorted(set(lineage_refs) - referenceable)
    duplicated = sorted(
        record_id for record_id, count in claims.items() if count > 1 and record_id in input_ids
    )

    return ConservationReport(
        input_records=len(input_ids),
        output_records=len(output_list),
        covered_ids=len(covered),
        coverage_pct=100.0 if not input_ids else len(covered) / len(input_ids) * 100,
        missing_ids=missing,
        invalid_reference_ids=invalid,
        reference_error_pct=0.0 if not lineage_refs else len(invalid) / len(lineage_refs) * 100,
        duplicated_ids=duplicated,
    )


def recover_dropped_records(
    inputs: list[Any],
    outputs: list[Any],
    minted_ids: Iterable[int] | None = None,
) -> tuple[list[Any], ConservationReport]:
    """Add back every input record the output fails to account for.

    This is the anti-join Abzu's evaluation stops short of: it detects missing
    lineage and reports a coverage percentage, but does not put the record
    back. Here the input itself is appended to the output, marked so a reader
    can tell a recovered record from a resolved one, and the report is written
    afterwards so it describes the repaired data.

    Parameters
    ----------
    inputs : list[Any]
        Records the run started from
    outputs : list[Any]
        Records the run produced
    minted_ids : Iterable[int] | None
        Ids the run created for merged entities, which are legitimate lineage
        targets even though they were never input records

    Returns
    -------
    tuple[list[Any], ConservationReport]
        The output with the missing records appended, and the report for the
        repaired output
    """
    minted = set(minted_ids or ())
    before = check_conservation(inputs, outputs, minted)

    repaired = outputs
    added = 0
    if before.missing_ids:
        missing = set(before.missing_ids)
        repaired = list(outputs)
        for record in inputs:
            if not coverage_of(record) & missing:
                continue
            repaired.append(_mark_recovered(record))
            missing -= coverage_of(record)
            added += 1
        logger.warning(
            f"Conservation: {len(before.missing_ids)} input records were not reachable from any "
            f"output record; added {added} of them back as unmatched records"
        )

    after = check_conservation(inputs, repaired, minted) if added else before
    after.recovered_records = added
    if not after.passes:
        message = f"Conservation failed after recovery: {after.summary()}"
        if bool(config.get("merge.conservation.fail_on_loss", True)):
            raise ValueError(message)
        logger.error(message)
    return repaired, after


def _mark_recovered(record: Any) -> Any:
    """Label a record as one that had to be recovered rather than resolved.

    Parameters
    ----------
    record : Any
        Input record being added back

    Returns
    -------
    Any
        A copy carrying the skip reason, or the record itself when it has
        nowhere to put one
    """
    if isinstance(record, dict):
        return {**record, "match_skip": True, "match_skip_reason": MISSING_IN_OUTPUT_REASON}
    fields = getattr(type(record), "model_fields", {})
    if hasattr(record, "model_copy") and "match_skip_reason" in fields:
        return record.model_copy(
            update={"match_skip": True, "match_skip_reason": MISSING_IN_OUTPUT_REASON}
        )
    return record
