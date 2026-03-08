"""Tests for UUIDMapper."""

from serf.dspy.types import BlockResolution, Entity, EntityBlock
from serf.match.uuid_mapper import UUIDMapper


def test_map_block_replaces_ids_with_consecutive_ints() -> None:
    """map_block replaces entity IDs with 0, 1, 2, ..."""
    block = EntityBlock(
        block_key="b1",
        block_size=3,
        entities=[
            Entity(id=100, name="A"),
            Entity(id=200, name="B"),
            Entity(id=300, name="C"),
        ],
    )
    mapper = UUIDMapper()
    mapped = mapper.map_block(block)
    assert [e.id for e in mapped.entities] == [0, 1, 2]
    assert [e.name for e in mapped.entities] == ["A", "B", "C"]
    assert all(e.source_uuids is None for e in mapped.entities)


def test_map_block_strips_source_uuids() -> None:
    """map_block strips source_uuids from entities."""
    block = EntityBlock(
        block_key="b1",
        block_size=1,
        entities=[
            Entity(id=1, name="A", source_uuids=["uuid-a"]),
        ],
    )
    mapper = UUIDMapper()
    mapped = mapper.map_block(block)
    assert mapped.entities[0].source_uuids is None


def test_unmap_block_restores_ids_and_source_uuids() -> None:
    """unmap_block restores original IDs and source_uuids."""
    block = EntityBlock(
        block_key="b1",
        block_size=2,
        entities=[
            Entity(id=100, name="A", uuid="uuid-100"),
            Entity(id=200, name="B", uuid="uuid-200"),
        ],
    )
    mapper = UUIDMapper()
    mapper.map_block(block)
    resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[
            Entity(id=0, name="A merged", source_ids=[1]),
        ],
        original_count=2,
        resolved_count=1,
    )
    restored = mapper.unmap_block(resolution, block)
    assert restored.resolved_entities[0].id == 100
    assert restored.resolved_entities[0].source_ids == [200]
    assert restored.resolved_entities[0].source_uuids == ["uuid-200"]


def test_unmap_block_recovers_missing_entities_as_singletons() -> None:
    """Missing entities are recovered as un-merged singletons, not merged into first entity."""
    block = EntityBlock(
        block_key="b1",
        block_size=3,
        entities=[
            Entity(id=100, name="A"),
            Entity(id=200, name="B"),
            Entity(id=300, name="C"),
        ],
    )
    mapper = UUIDMapper()
    mapper.map_block(block)
    # LLM only returned entity 0 (A merged with B), entity 2 (C) is missing
    resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[
            Entity(id=0, name="A", source_ids=[1]),
        ],
        original_count=3,
        resolved_count=1,
    )
    restored = mapper.unmap_block(resolution, block)
    # Should have 2 entities: A (merged with B) + C (recovered singleton)
    assert len(restored.resolved_entities) == 2
    # The merged entity keeps its source_ids
    merged = restored.resolved_entities[0]
    assert merged.id == 100
    assert 200 in (merged.source_ids or [])
    # The recovered entity is a singleton with match_skip
    recovered = [e for e in restored.resolved_entities if e.match_skip_reason]
    assert len(recovered) == 1
    assert recovered[0].id == 300
    assert recovered[0].match_skip_reason == "missing_in_match_output"
