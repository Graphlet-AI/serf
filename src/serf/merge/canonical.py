"""Canonical records: what a group of matched records becomes.

A merge produces a new thing, so it gets a new identity. The records that went
into it are not thrown away; their uuids, and the uuids they had already
absorbed, move into ``source_uuids``, which is the only place lineage lives.
Nothing that went in is ever unreachable from what comes out.

Identity is a UUID throughout. Integer ids exist in exactly one place in SERF:
``serf.match.uuid_mapper.UUIDMapper`` renumbers a block's records 1..n before
the LLM sees them, because small integers are cheap tokens and a model copies
them back reliably, and maps them straight back to uuids afterwards. Outside
that boundary an integer identifier is either a key belonging to some source
system or a bug. A minted UUID also needs no allocator and can never collide
with a record the current process has not seen, which an integer minted from
the ids in one block always could.

Every field except the identity holds a list, because a merged record can
legitimately carry two states or three descriptions, and a schema that says
otherwise forces a lossy choice at exactly the moment the data got richer. The
list is ordered most complete first, so a consumer that wants one value takes
the head and gets the fullest one rather than an arbitrary one. The shape is
Senzing's idea - features are multi-valued and lineage is explicit - without
its wire format.
"""

import uuid as uuid_module
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from serf.analyze.field_detection import detect_field_type
from serf.logs import get_logger
from serf.merge.semantics import MergePolicy, merge_values

logger = get_logger(__name__)

UUID_FIELD = "uuid"
SOURCE_UUIDS_FIELD = "source_uuids"
RESERVED_FIELDS = frozenset({UUID_FIELD, SOURCE_UUIDS_FIELD})


# Namespace for deriving an identity from a source key, used only to repair a
# record that reached merging without one.
_SOURCE_KEY_NAMESPACE = uuid_module.UUID("2f1c9d84-6b2a-4a7f-9d3e-8c5b71a40f26")


def mint_uuid() -> str:
    """Return a fresh identity for a merged entity.

    Returns
    -------
    str
        A random UUID, which no other record can hold
    """
    return str(uuid_module.uuid4())


def uuid_for_source_key(key: object) -> str:
    """Derive a stable identity from a source system's key.

    A record should arrive carrying a uuid. One that does not would contribute
    nothing to its merged record's lineage, which is a silent hole in the audit
    trail, so its identity is derived from the key it does have rather than
    left empty. Deriving rather than randomising keeps the result the same
    across runs, so lineage stays comparable.

    Parameters
    ----------
    key : object
        The record's key in its own source system

    Returns
    -------
    str
        A UUID determined by the key
    """
    return str(uuid_module.uuid5(_SOURCE_KEY_NAMESPACE, str(key)))


class CanonicalRecord(BaseModel):
    """One resolved entity and the records it stands for.

    Parameters
    ----------
    uuid : str
        Identity of this entity. Minted fresh when the record is a merge of
        several inputs, carried over when it stands for a single input.
    source_uuids : list[str]
        Every input uuid this record absorbed, transitively, sorted. Excludes
        ``uuid`` itself.
    fields : dict[str, list[Any]]
        Field values, each a list ordered most complete first, in the order the
        fields were first seen across the inputs
    """

    uuid: str
    source_uuids: list[str] = Field(default_factory=list)
    fields: dict[str, list[Any]] = Field(default_factory=dict)

    def to_dict(self, include_empty_source_uuids: bool = False) -> dict[str, Any]:
        """Render the record as a flat mapping, identity first and lineage last.

        Parameters
        ----------
        include_empty_source_uuids : bool
            Emit ``source_uuids`` even when the record absorbed nothing, which
            a fixed-schema writer needs and a human reader does not

        Returns
        -------
        dict[str, Any]
            ``uuid``, then each field in first-seen order, then ``source_uuids``
        """
        record: dict[str, Any] = {UUID_FIELD: self.uuid}
        record.update(self.fields)
        if self.source_uuids or include_empty_source_uuids:
            record[SOURCE_UUIDS_FIELD] = self.source_uuids
        return record

    def covers(self) -> set[str]:
        """Return every input uuid this record accounts for.

        Returns
        -------
        set[str]
            ``source_uuids`` plus ``uuid``
        """
        return {self.uuid, *self.source_uuids}


def known_uuids(records: Iterable[dict[str, Any]]) -> set[str]:
    """Collect every uuid a set of records uses, lineage included.

    Parameters
    ----------
    records : Iterable[dict[str, Any]]
        Source records, each carrying ``uuid`` and optionally ``source_uuids``

    Returns
    -------
    set[str]
        Every uuid mentioned
    """
    seen: set[str] = set()
    for record in records:
        seen.add(str(record[UUID_FIELD]))
        seen.update(str(value) for value in record.get(SOURCE_UUIDS_FIELD) or [])
    return seen


def canonicalize(
    records: list[dict[str, Any]],
    field_types: dict[str, str] | None = None,
    policies: dict[str, MergePolicy] | None = None,
) -> CanonicalRecord:
    """Combine a group of matched records into the single entity they denote.

    A group of one keeps its uuid, because nothing was merged. A group of
    several gets a minted uuid, and every uuid in the group - including the
    ones the group members had already absorbed - moves into ``source_uuids``.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records the matcher put in one group, each with ``uuid`` and optionally
        ``source_uuids`` and any number of scalar or list-valued fields
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

    lineage = known_uuids(records)
    identity = str(records[0][UUID_FIELD]) if len(records) == 1 else mint_uuid()

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
        uuid=identity,
        source_uuids=sorted(lineage - {identity}),
        fields=fields,
    )


def canonicalize_groups(
    groups: list[list[dict[str, Any]]],
    field_types: dict[str, str] | None = None,
    policies: dict[str, MergePolicy] | None = None,
) -> list[CanonicalRecord]:
    """Canonicalize every group of a partition.

    Parameters
    ----------
    groups : list[list[dict[str, Any]]]
        Partition of the input records
    field_types : dict[str, str] | None
        Field name to inferred type, overriding detection
    policies : dict[str, MergePolicy] | None
        Field name to merge policy, overriding the field type's

    Returns
    -------
    list[CanonicalRecord]
        One record per group, in group order
    """
    return [canonicalize(group, field_types, policies) for group in groups]
