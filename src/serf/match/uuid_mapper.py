"""UUID-to-int mapping for LLM block matching.

Maps entity UUIDs/IDs to consecutive integers before LLM calls (LLMs work
better with small ints) and restores them after the LLM returns.
"""

from typing import Any

from serf.dspy.types import BlockResolution, Entity, EntityBlock
from serf.logs import get_logger

logger = get_logger(__name__)


class UUIDMapper:
    """Maps entity IDs to consecutive integers for LLM compatibility.

    Caches source_uuids and original IDs for restoration after the LLM
    returns. Performs two-phase recovery for entities missing from LLM output.
    """

    def __init__(self) -> None:
        """Initialize the mapper with empty cache."""
        self._id_to_int: dict[int, int] = {}
        self._int_to_original: dict[int, dict[str, Any]] = {}
        self._mapped_ids: set[int] = set()

    def map_block(self, block: EntityBlock) -> EntityBlock:
        """Replace entity IDs with consecutive ints and strip source_uuids.

        Parameters
        ----------
        block : EntityBlock
            Block with entities to map

        Returns
        -------
        EntityBlock
            New block with mapped IDs (0, 1, 2, ...) and source_uuids stripped
        """
        self._id_to_int.clear()
        self._int_to_original.clear()
        self._mapped_ids.clear()

        mapped_entities: list[Entity] = []
        for i, entity in enumerate(block.entities):
            self._id_to_int[entity.id] = i
            self._mapped_ids.add(entity.id)
            self._int_to_original[i] = {
                "id": entity.id,
                "uuid": entity.uuid,
                "source_ids": entity.source_ids or [],
                "source_uuids": entity.source_uuids or [],
                "entity": entity,
            }

            mapped_entity = entity.model_copy(deep=True)
            mapped_entity.id = i
            mapped_entity.uuid = None
            mapped_entity.source_uuids = None
            mapped_entities.append(mapped_entity)

        return EntityBlock(
            block_key=block.block_key,
            block_key_type=block.block_key_type,
            block_size=len(mapped_entities),
            entities=mapped_entities,
        )

    def unmap_block(
        self, resolution: BlockResolution, original_block: EntityBlock
    ) -> BlockResolution:
        """Restore original UUIDs and IDs in the resolution.

        Performs two-phase missing entity recovery:
        - Phase 1: Add missing entity IDs to existing resolved entities' source_ids
        - Phase 2: Recover entire missing entities with match_skip_reason

        Parameters
        ----------
        resolution : BlockResolution
            LLM output with mapped IDs
        original_block : EntityBlock
            Original block before mapping (for recovery)

        Returns
        -------
        BlockResolution
            Resolution with restored IDs and source_uuids
        """
        resolved_ids = {e.id for e in resolution.resolved_entities}
        all_mapped_ids = set(self._int_to_original.keys())
        missing_ids = all_mapped_ids - resolved_ids

        # Phase 1: Add missing IDs to first resolved entity's source_ids
        if missing_ids and resolution.resolved_entities:
            first = resolution.resolved_entities[0]
            existing_sources = set(first.source_ids or [])
            first_sources = list(existing_sources | missing_ids)
            resolution.resolved_entities[0] = first.model_copy(update={"source_ids": first_sources})

        # Phase 2: Recover entire missing entities
        for mapped_id in sorted(missing_ids):
            orig = self._int_to_original[mapped_id]
            entity = orig["entity"].model_copy(deep=True)
            entity.match_skip = True
            entity.match_skip_reason = "missing_in_match_output"
            resolution.resolved_entities.append(entity)

        # Restore IDs and source_uuids for all resolved entities
        restored: list[Entity] = []
        for entity in resolution.resolved_entities:
            if entity.match_skip_reason == "missing_in_match_output":
                restored.append(entity)
                continue

            orig_id = self._int_to_original.get(entity.id)
            if orig_id is None:
                restored.append(entity)
                continue

            # Restore master id and source_ids to original space
            new_id = orig_id["id"]
            new_source_ids: list[int] = []
            new_source_uuids: list[str] = []

            for sid in entity.source_ids or []:
                if sid in self._int_to_original:
                    new_source_ids.append(self._int_to_original[sid]["id"])
                    new_source_uuids.extend(self._int_to_original[sid]["source_uuids"])
                    orig_uuid = self._int_to_original[sid]["uuid"]
                    if orig_uuid:
                        new_source_uuids.append(orig_uuid)

            # Add master's own source_uuids from cache
            new_source_uuids.extend(orig_id["source_uuids"])
            new_source_ids.extend(orig_id["source_ids"])

            restored_entity = entity.model_copy(
                update={
                    "id": new_id,
                    "uuid": orig_id["uuid"],
                    "source_ids": new_source_ids or None,
                    "source_uuids": new_source_uuids or None,
                }
            )
            restored.append(restored_entity)

        # Restore match IDs in MatchDecision (LLM uses mapped ints)
        restored_matches = []
        for m in resolution.matches:
            orig_a = self._int_to_original.get(m.entity_a_id, {}).get("id")
            orig_b = self._int_to_original.get(m.entity_b_id, {}).get("id")
            if orig_a is not None and orig_b is not None:
                restored_matches.append(
                    m.model_copy(
                        update={
                            "entity_a_id": orig_a,
                            "entity_b_id": orig_b,
                        }
                    )
                )
            else:
                restored_matches.append(m)

        return BlockResolution(
            block_key=resolution.block_key,
            matches=restored_matches,
            resolved_entities=restored,
            was_resolved=resolution.was_resolved,
            original_count=resolution.original_count,
            resolved_count=len(restored),
        )
