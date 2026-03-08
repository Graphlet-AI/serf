"""Entity merging for resolved matches.

Merges multiple entities into one canonical record, selecting the most
complete field values and accumulating source_ids/source_uuids.
"""

from typing import Any

from serf.dspy.types import Entity
from serf.logs import get_logger

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
    """Merge attribute dicts, picking longest/most complete values.

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
        else:
            merged[key] = _pick_best_value(merged[key], val_b)
    return merged


class EntityMerger:
    """Merges multiple entities into a single canonical record.

    Lowest ID becomes master. All other IDs and UUIDs go into source_ids
    and source_uuids. Field-level merge picks the most complete value.
    """

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

        master = entities[0]
        for e in entities[1:]:
            master = self.merge_pair(master, e)
        return master

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
            master, other = a, b
        else:
            master, other = b, a

        master_source_ids = list(master.source_ids or [])
        master_source_uuids = list(master.source_uuids or [])

        master_source_ids.append(other.id)
        master_source_ids.extend(other.source_ids or [])

        if other.uuid:
            master_source_uuids.append(other.uuid)
        master_source_uuids.extend(other.source_uuids or [])

        name = _pick_best_value(master.name, other.name)
        description = _pick_best_value(master.description, other.description)
        entity_type = _pick_best_value(master.entity_type, other.entity_type)
        attributes = _merge_attributes(master.attributes, other.attributes)

        return Entity(
            id=master.id,
            uuid=master.uuid,
            name=name,
            description=description,
            entity_type=entity_type,
            attributes=attributes,
            source_ids=master_source_ids or None,
            source_uuids=master_source_uuids or None,
            match_skip=master.match_skip,
            match_skip_reason=master.match_skip_reason,
            match_skip_history=master.match_skip_history,
        )
