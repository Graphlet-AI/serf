"""Entity merging for resolved matches.

Merges multiple entities into one canonical record, selecting the best value
for singular fields (via the LLM or completeness ranking) and deduplicating
list fields (such as addresses or names) into lists of individual values
according to their semantic type.
"""

from collections.abc import Iterable
from typing import Any

from serf.analyze.field_detection import detect_field_type
from serf.dspy.types import Entity
from serf.logs import get_logger
from serf.merge.semantics import (
    CARDINALITY_LIST,
    CARDINALITY_SINGLE,
    FIELD_TYPE_NAME,
    FIELD_TYPE_TEXT,
    MergePolicy,
    merge_values,
    select_best_value,
)
from serf.schema.spec import EntitySchema

logger = get_logger(__name__)


def _pick_best_value(a: Any, b: Any) -> Any:
    """Pick the most complete/longest non-empty string value.

    Parameters
    ----------
    a : Any
        First value
    b : Any
        Second value

    Returns
    -------
    Any
        The preferred value (longer non-empty string, or first if equal)
    """
    if a is None or a == "":
        return b
    if b is None or b == "":
        return a
    if isinstance(a, str) and isinstance(b, str):
        return a if len(a) >= len(b) else b
    return a


def _merge_attributes(attrs_a: dict[str, Any], attrs_b: dict[str, Any]) -> dict[str, Any]:
    """Merge attribute dicts, deduplicating list values and picking longest scalars.

    Parameters
    ----------
    attrs_a : dict[str, Any]
        First entity's attributes
    attrs_b : dict[str, Any]
        Second entity's attributes

    Returns
    -------
    dict[str, Any]
        Merged attributes
    """
    merged: dict[str, Any] = dict(attrs_a)
    for key, val_b in attrs_b.items():
        if key not in merged:
            merged[key] = val_b
        elif isinstance(merged[key], (list, tuple, set)) or isinstance(val_b, (list, tuple, set)):
            field_type = detect_field_type(key, [merged[key], val_b])
            merged[key] = merge_values([merged[key], val_b], field_type)
        else:
            merged[key] = _pick_best_value(merged[key], val_b)
    return merged


class EntityMerger:
    """Merges multiple entities into a single canonical record.

    Lowest ID becomes master. All other IDs and UUIDs go into ``source_ids``
    and ``source_uuids``. List fields (whether holding lists on input entities
    or configured in ``list_fields`` / ``policies`` / ``schema``) are
    deduplicated into lists of individual values using their semantic type
    (``address``, ``name``, ``email``, ``phone``, ``url``, ``identifier``,
    ``date``, ``numeric``, ``text``). Singular fields pick the best single
    value via the LLM when ``use_llm=True`` (or ``lm`` is provided), or
    deterministically otherwise.

    Parameters
    ----------
    field_types : dict[str, str] | None
        Mapping of field name to semantic type (e.g. ``{"addresses": "address"}``)
    list_fields : Iterable[str] | dict[str, str] | None
        Field names (or mapping of field name to semantic type) that should be
        merged as deduplicated lists of individual values
    singular_fields : Iterable[str] | dict[str, str] | None
        Field names (or mapping of field name to semantic type) that should
        resolve to a single best value
    policies : dict[str, MergePolicy] | None
        Per-field merge policies
    schema : EntitySchema | None
        Optional entity schema supplying field types, policies, and descriptions
    use_llm : bool
        Select the best value for singular fields via the LLM
    lm : Any
        Optional DSPy LM instance for LLM value selection
    """

    def __init__(
        self,
        *,
        field_types: dict[str, str] | None = None,
        list_fields: Iterable[str] | dict[str, str] | None = None,
        singular_fields: Iterable[str] | dict[str, str] | None = None,
        policies: dict[str, MergePolicy] | None = None,
        schema: EntitySchema | None = None,
        use_llm: bool = False,
        lm: Any = None,
    ) -> None:
        self.field_types: dict[str, str] = {}
        self.policies: dict[str, MergePolicy] = {}
        self.field_descriptions: dict[str, str] = {}
        if schema is not None:
            self.field_types.update(schema.field_types())
            self.policies.update(schema.merge_policies())
            self.field_descriptions.update(schema.field_descriptions())
        if field_types:
            self.field_types.update(field_types)
        if policies:
            self.policies.update(policies)

        if isinstance(list_fields, dict):
            self.list_fields: set[str] = set(list_fields.keys())
            self.field_types.update(list_fields)
        else:
            self.list_fields = set(list_fields or ())

        if isinstance(singular_fields, dict):
            self.singular_fields: set[str] = set(singular_fields.keys())
            self.field_types.update(singular_fields)
        else:
            self.singular_fields = set(singular_fields or ())

        self.use_llm = use_llm
        self.lm = lm

    def _is_list_field(self, key: str, values: list[Any]) -> bool:
        """Return whether an attribute should resolve to a deduplicated list."""
        if key in self.singular_fields:
            return False
        if key in self.list_fields:
            return True
        policy = self.policies.get(key)
        if policy is not None:
            return policy.cardinality == CARDINALITY_LIST
        return any(isinstance(value, (list, tuple, set)) for value in values)

    def _resolve_singular(
        self,
        key: str,
        values: list[Any],
        default_type: str = FIELD_TYPE_TEXT,
    ) -> Any:
        """Resolve a singular field across entities."""
        field_type = self.field_types.get(key) or default_type
        policy = self.policies.get(key)
        if self.use_llm or self.lm is not None or key in self.field_types or policy is not None:
            selected = select_best_value(
                values,
                field_type,
                policy,
                field_name=key,
                field_description=self.field_descriptions.get(key, ""),
                use_llm=self.use_llm,
                lm=self.lm,
            )
            if selected is not None:
                return selected
        best: Any = None
        for value in values:
            best = _pick_best_value(best, value) if best is not None else value
        return best if best is not None else ""

    def _merge_entity_attributes(self, entities: list[Entity]) -> dict[str, Any]:
        """Merge attributes across entities, handling both list and singular fields."""
        ordered_keys: list[str] = []
        collected: dict[str, list[Any]] = {}
        for entity in entities:
            for key, value in entity.attributes.items():
                if key not in collected:
                    collected[key] = []
                    ordered_keys.append(key)
                collected[key].append(value)

        merged: dict[str, Any] = {}
        for key in ordered_keys:
            values = collected[key]
            if self._is_list_field(key, values):
                field_type = self.field_types.get(key) or detect_field_type(key, values)
                merged[key] = merge_values(values, field_type, self.policies.get(key))
            else:
                inferred = self.field_types.get(key)
                if inferred is None and (self.use_llm or self.lm is not None):
                    inferred = detect_field_type(key, values)
                resolved = self._resolve_singular(
                    key,
                    values,
                    default_type=inferred or FIELD_TYPE_TEXT,
                )
                merged[key] = resolved
        return merged

    def merge_entities(self, entities: list[Entity]) -> Entity:
        """Merge multiple entities into one.

        Parameters
        ----------
        entities : list[Entity]
            Entities to merge (must be non-empty)

        Returns
        -------
        Entity
            Single merged entity with lowest id as master
        """
        if not entities:
            raise ValueError("Cannot merge empty entity list")
        if len(entities) == 1:
            return entities[0]

        ordered = sorted(entities, key=lambda entity: entity.id)
        master = ordered[0]
        others = ordered[1:]

        merged_source_ids: list[int] = list(master.source_ids or [])
        for other in others:
            merged_source_ids.append(other.id)
            merged_source_ids.extend(other.source_ids or [])
        merged_source_ids = list(dict.fromkeys(merged_source_ids))
        merged_source_ids = [s for s in merged_source_ids if s != master.id]

        merged_source_uuids: list[str] = list(master.source_uuids or [])
        for other in others:
            if other.uuid:
                merged_source_uuids.append(other.uuid)
            merged_source_uuids.extend(other.source_uuids or [])
        merged_source_uuids = list(dict.fromkeys(merged_source_uuids))
        merged_source_uuids = [s for s in merged_source_uuids if s != master.uuid]

        all_from_master = [master, *others]
        name = str(
            self._resolve_singular(
                "name",
                [e.name for e in all_from_master],
                default_type=FIELD_TYPE_NAME,
            )
            or ""
        )
        description = str(
            self._resolve_singular(
                "description",
                [e.description for e in all_from_master],
                default_type=FIELD_TYPE_TEXT,
            )
            or ""
        )
        entity_type = str(
            self._resolve_singular(
                "entity_type",
                [e.entity_type for e in all_from_master],
                default_type=FIELD_TYPE_TEXT,
            )
            or "entity"
        )
        attributes = self._merge_entity_attributes(all_from_master)

        return Entity(
            id=master.id,
            uuid=master.uuid,
            name=name,
            description=description,
            entity_type=entity_type,
            attributes=attributes,
            source_ids=merged_source_ids or None,
            source_uuids=merged_source_uuids or None,
            match_skip=master.match_skip,
            match_skip_reason=master.match_skip_reason,
            match_skip_history=master.match_skip_history,
        )

    def merge_pair(self, a: Entity, b: Entity) -> Entity:
        """Merge two entities into one.

        Parameters
        ----------
        a : Entity
            First entity
        b : Entity
            Second entity

        Returns
        -------
        Entity
            Merged entity (lowest id is master)
        """
        if a.id <= b.id:
            return self.merge_entities([a, b])
        return self.merge_entities([b, a])

    @staticmethod
    def policy_for_cardinality(
        cardinality: str = CARDINALITY_SINGLE,
        dedupe: str = "exact",
        threshold: float = 0.8,
    ) -> MergePolicy:
        """Build a ``MergePolicy`` with explicit cardinality."""
        return MergePolicy(dedupe=dedupe, threshold=threshold, cardinality=cardinality)
