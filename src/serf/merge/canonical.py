"""Canonical records: what a group of matched records becomes.

A merge produces a new thing, so it gets a new id. The records that went into
it are not thrown away; their ids, and the ids they had already absorbed, move
into ``source_ids``, which is the only place lineage lives. Nothing that went
in is ever unreachable from what comes out.

Every field except ``id`` holds a list, because a merged record can legitimately
carry two states or three descriptions, and a schema that says otherwise forces
a lossy choice at exactly the moment the data got richer. The list is ordered
most complete first, so a consumer that wants one value takes the head and gets
the fullest one rather than an arbitrary one. The shape is Senzing's idea -
features are multi-valued and lineage is explicit - without its wire format.
"""

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from serf.analyze.field_detection import detect_field_type
from serf.config import config
from serf.logs import get_logger
from serf.merge.semantics import MergePolicy, merge_values

logger = get_logger(__name__)

ID_FIELD = "id"
SOURCE_IDS_FIELD = "source_ids"
RESERVED_FIELDS = frozenset({ID_FIELD, SOURCE_IDS_FIELD})

STRATEGY_SMALLEST_UNUSED = "smallest_unused"
STRATEGY_SEQUENTIAL = "sequential"
ID_STRATEGIES = (STRATEGY_SMALLEST_UNUSED, STRATEGY_SEQUENTIAL)


class CanonicalRecord(BaseModel):
    """One resolved entity and the records it stands for.

    Parameters
    ----------
    id : int
        Identifier for this entity. Minted fresh when the record is a merge of
        several inputs, carried over when it stands for a single input.
    source_ids : list[int]
        Every input id this record absorbed, transitively, sorted. Excludes
        ``id`` itself.
    fields : dict[str, list[Any]]
        Field values, each a list ordered most complete first, in the order the
        fields were first seen across the inputs
    """

    id: int
    source_ids: list[int] = Field(default_factory=list)
    fields: dict[str, list[Any]] = Field(default_factory=dict)

    def to_dict(self, include_empty_source_ids: bool = False) -> dict[str, Any]:
        """Render the record as a flat mapping, ``id`` first and lineage last.

        Parameters
        ----------
        include_empty_source_ids : bool
            Emit ``source_ids`` even when the record absorbed nothing, which a
            fixed-schema writer needs and a human reader does not

        Returns
        -------
        dict[str, Any]
            ``id``, then each field in first-seen order, then ``source_ids``
        """
        record: dict[str, Any] = {ID_FIELD: self.id}
        record.update(self.fields)
        if self.source_ids or include_empty_source_ids:
            record[SOURCE_IDS_FIELD] = self.source_ids
        return record

    def covers(self) -> set[int]:
        """Return every input id this record accounts for.

        Returns
        -------
        set[int]
            ``source_ids`` plus ``id``
        """
        return {self.id, *self.source_ids}


class IdAllocator:
    """Hands out ids that no record and no lineage entry has claimed.

    Parameters
    ----------
    reserved : Iterable[int]
        Every id already in use, including ids that appear only inside a
        record's ``source_ids``. Minting one of those would make lineage
        ambiguous.
    strategy : str | None
        ``smallest_unused`` to fill gaps, ``sequential`` to count up from the
        highest id seen. Defaults to config ``merge.id_minting.strategy``.
    """

    def __init__(self, reserved: Iterable[int], strategy: str | None = None) -> None:
        self._used: set[int] = {int(value) for value in reserved}
        configured = strategy or str(config.get("merge.id_minting.strategy", STRATEGY_SEQUENTIAL))
        if configured not in ID_STRATEGIES:
            logger.warning(
                f"Unknown id minting strategy '{configured}'; using '{STRATEGY_SEQUENTIAL}'. "
                f"Valid strategies: {ID_STRATEGIES}"
            )
            configured = STRATEGY_SEQUENTIAL
        self.strategy = configured
        self._cursor = (
            1 if configured == STRATEGY_SMALLEST_UNUSED else max(self._used, default=0) + 1
        )
        self._issued: set[int] = set()

    @property
    def issued(self) -> set[int]:
        """Return every id this allocator has minted.

        An entity merged in one round can be merged again in the next, and its
        minted id then becomes a lineage entry of the result. Those ids are
        real - each identifies an entity that existed - but they were never
        input records, so a conservation check has to be told about them to
        tell them apart from a reference to nothing.

        Returns
        -------
        set[int]
            Minted ids
        """
        return set(self._issued)

    def mint(self) -> int:
        """Return an unused id and mark it used.

        Returns
        -------
        int
            An id no record carries and no ``source_ids`` list references
        """
        while self._cursor in self._used:
            self._cursor += 1
        minted = self._cursor
        self._used.add(minted)
        self._issued.add(minted)
        self._cursor += 1
        return minted

    def reserve(self, value: int) -> None:
        """Mark an id as taken so it is never minted.

        Parameters
        ----------
        value : int
            Id to reserve
        """
        self._used.add(int(value))


def known_ids(records: Iterable[dict[str, Any]]) -> set[int]:
    """Collect every id a set of records uses, lineage included.

    Parameters
    ----------
    records : Iterable[dict[str, Any]]
        Source records, each carrying ``id`` and optionally ``source_ids``

    Returns
    -------
    set[int]
        Ids that must never be minted for a new entity
    """
    seen: set[int] = set()
    for record in records:
        seen.add(int(record[ID_FIELD]))
        seen.update(int(value) for value in record.get(SOURCE_IDS_FIELD) or [])
    return seen


def canonicalize(
    records: list[dict[str, Any]],
    allocator: IdAllocator,
    field_types: dict[str, str] | None = None,
    policies: dict[str, MergePolicy] | None = None,
) -> CanonicalRecord:
    """Combine a group of matched records into the single entity they denote.

    A group of one keeps its id, because nothing was merged. A group of several
    gets a minted id, and every id in the group - including the ones the group
    members had already absorbed - moves into ``source_ids``.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records the matcher put in one group, each with ``id`` and optionally
        ``source_ids`` and any number of scalar or list-valued fields
    allocator : IdAllocator
        Source of the merged record's new id
    field_types : dict[str, str] | None
        Field name to inferred type, overriding detection. Supply this from a
        schema when the values alone would be read wrongly.
    policies : dict[str, MergePolicy] | None
        Field name to merge policy, overriding the one the field's type would
        get. Supply this from a schema.

    Returns
    -------
    CanonicalRecord
        The merged entity, every field a list ordered most complete first

    Raises
    ------
    ValueError
        If the group is empty
    """
    if not records:
        raise ValueError("Cannot canonicalize an empty group of records")

    lineage: set[int] = known_ids(records)

    entity_id = int(records[0][ID_FIELD]) if len(records) == 1 else allocator.mint()

    ordered_fields: list[str] = []
    collected: dict[str, list[Any]] = {}
    for record in records:
        for key, value in record.items():
            if key in RESERVED_FIELDS:
                continue
            if key not in collected:
                collected[key] = []
                ordered_fields.append(key)
            collected[key].extend(value if isinstance(value, list) else [value])

    overrides = field_types or {}
    policy_overrides = policies or {}
    fields: dict[str, list[Any]] = {}
    for key in ordered_fields:
        field_type = overrides.get(key) or detect_field_type(key, collected[key])
        merged = merge_values(collected[key], field_type, policy_overrides.get(key))
        if merged:
            fields[key] = merged

    return CanonicalRecord(
        id=entity_id,
        source_ids=sorted(lineage - {entity_id}),
        fields=fields,
    )


def canonicalize_groups(
    groups: list[list[dict[str, Any]]],
    field_types: dict[str, str] | None = None,
    allocator: IdAllocator | None = None,
    reserved: Iterable[int] | None = None,
    policies: dict[str, MergePolicy] | None = None,
) -> list[CanonicalRecord]:
    """Canonicalize every group of a partition against one id space.

    One allocator serves all the groups, so two merges in the same pass can
    never mint the same id.

    ``groups`` is often a slice of the data - one block, one partition of a
    shard - and an id space inferred from a slice does not know about the
    records outside it. Pass ``reserved`` with every id in the dataset, or an
    ``allocator`` already seeded with them, or a merge in this pass can be
    handed the id of a record it never saw.

    Parameters
    ----------
    groups : list[list[dict[str, Any]]]
        Partition of the input records
    field_types : dict[str, str] | None
        Field name to inferred type, overriding detection
    allocator : IdAllocator | None
        Reused allocator, which takes precedence over ``reserved``
    reserved : Iterable[int] | None
        Every id in the wider dataset. Defaults to the ids inside ``groups``,
        which is only enough when ``groups`` is the whole dataset.
    policies : dict[str, MergePolicy] | None
        Field name to merge policy, overriding the field type's

    Returns
    -------
    list[CanonicalRecord]
        One record per group, in group order
    """
    if allocator is None:
        seed = known_ids(record for group in groups for record in group)
        seed.update(int(value) for value in reserved or ())
        allocator = IdAllocator(seed)
    return [canonicalize(group, allocator, field_types, policies) for group in groups]
