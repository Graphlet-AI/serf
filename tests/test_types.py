"""Tests for SERF pipeline types."""

import json

from serf.dspy.types import (
    BlockingMetrics,
    BlockResolution,
    DatasetProfile,
    Entity,
    EntityBlock,
    FieldProfile,
    IterationMetrics,
    MatchDecision,
)


def test_entity_creation() -> None:
    """Test basic entity creation with required fields."""
    entity = Entity(id=1, name="Apple Inc.")
    assert entity.id == 1
    assert entity.name == "Apple Inc."
    assert entity.description == ""
    assert entity.entity_type == "entity"
    assert entity.attributes == {}
    assert entity.source_ids is None
    assert entity.match_skip is None


def test_entity_with_all_fields() -> None:
    """Test entity creation with all optional fields."""
    entity = Entity(
        id=1,
        uuid="abc-123",
        name="Apple Inc.",
        description="Technology company",
        entity_type="company",
        attributes={"ticker": "AAPL", "revenue": 394000000000},
        source_ids=[2, 3],
        source_uuids=["def-456", "ghi-789"],
        match_skip=False,
        match_skip_reason=None,
        match_skip_history=[1],
    )
    assert entity.uuid == "abc-123"
    assert entity.attributes["ticker"] == "AAPL"
    assert entity.source_ids == [2, 3]
    assert entity.match_skip_history == [1]


def test_entity_text_for_embedding() -> None:
    """Test text generation for embedding."""
    entity = Entity(
        id=1,
        name="Apple Inc.",
        description="Technology company",
        attributes={"location": "Cupertino, CA"},
    )
    text = entity.text_for_embedding()
    assert "Apple Inc." in text
    assert "Technology company" in text
    assert "Cupertino, CA" in text


def test_entity_text_for_embedding_no_description() -> None:
    """Test text generation with no description."""
    entity = Entity(id=1, name="Apple Inc.")
    text = entity.text_for_embedding()
    assert text == "Apple Inc."


def test_entity_serialization() -> None:
    """Test entity JSON serialization roundtrip."""
    entity = Entity(
        id=1,
        name="Test Corp",
        description="A test company",
        attributes={"field1": "value1"},
    )
    json_str = entity.model_dump_json()
    restored = Entity.model_validate_json(json_str)
    assert restored.id == entity.id
    assert restored.name == entity.name
    assert restored.attributes == entity.attributes


def test_entity_block() -> None:
    """Test EntityBlock creation and validation."""
    entities = [
        Entity(id=1, name="Company A"),
        Entity(id=2, name="Company B"),
    ]
    block = EntityBlock(
        block_key="cluster_0",
        block_key_type="semantic",
        block_size=2,
        entities=entities,
    )
    assert block.block_key == "cluster_0"
    assert block.block_size == 2
    assert len(block.entities) == 2


def test_match_decision() -> None:
    """Test MatchDecision creation."""
    decision = MatchDecision(
        entity_a_id=1,
        entity_b_id=2,
        is_match=True,
        confidence=0.95,
        reasoning="Same company name and location",
    )
    assert decision.is_match is True
    assert decision.confidence == 0.95


def test_match_decision_confidence_bounds() -> None:
    """Test that confidence must be between 0 and 1."""
    import pytest

    with pytest.raises(ValueError):
        MatchDecision(
            entity_a_id=1,
            entity_b_id=2,
            is_match=True,
            confidence=1.5,
            reasoning="test",
        )


def test_block_resolution() -> None:
    """Test BlockResolution creation."""
    resolution = BlockResolution(
        block_key="cluster_0",
        matches=[
            MatchDecision(
                entity_a_id=1,
                entity_b_id=2,
                is_match=True,
                confidence=0.9,
                reasoning="Same entity",
            )
        ],
        resolved_entities=[
            Entity(id=1, name="Merged Entity", source_ids=[2]),
        ],
        was_resolved=True,
        original_count=2,
        resolved_count=1,
    )
    assert resolution.was_resolved is True
    assert resolution.original_count == 2
    assert resolution.resolved_count == 1


def test_block_resolution_defaults() -> None:
    """Test BlockResolution default values."""
    resolution = BlockResolution()
    assert resolution.block_key == ""
    assert resolution.matches == []
    assert resolution.resolved_entities == []
    assert resolution.was_resolved is False


def test_field_profile() -> None:
    """Test FieldProfile creation."""
    profile = FieldProfile(
        name="title",
        inferred_type="name",
        completeness=0.98,
        uniqueness=0.85,
        sample_values=["iPhone 14", "Galaxy S23"],
        is_blocking_candidate=True,
        is_matching_feature=True,
    )
    assert profile.name == "title"
    assert profile.is_blocking_candidate is True


def test_dataset_profile() -> None:
    """Test DatasetProfile creation."""
    profile = DatasetProfile(
        record_count=1000,
        field_profiles=[
            FieldProfile(name="title", inferred_type="name"),
            FieldProfile(name="price", inferred_type="numeric"),
        ],
        recommended_blocking_fields=["title"],
        recommended_matching_fields=["title", "price"],
        estimated_duplicate_rate=0.15,
    )
    assert profile.record_count == 1000
    assert len(profile.field_profiles) == 2
    assert "title" in profile.recommended_blocking_fields


def test_iteration_metrics() -> None:
    """Test IterationMetrics creation."""
    metrics = IterationMetrics(
        iteration=1,
        input_entities=1000,
        output_entities=800,
        reduction_pct=20.0,
        overall_reduction_pct=20.0,
        blocks_count=50,
        singleton_blocks=5,
        largest_block=45,
    )
    assert metrics.reduction_pct == 20.0


def test_blocking_metrics() -> None:
    """Test BlockingMetrics creation."""
    metrics = BlockingMetrics(
        total_blocks=100,
        total_entities=5000,
        avg_block_size=50.0,
        max_block_size=200,
        singleton_blocks=10,
        pair_completeness=0.95,
        reduction_ratio=0.99,
    )
    assert metrics.total_blocks == 100
    assert metrics.reduction_ratio == 0.99


def test_entity_block_serialization() -> None:
    """Test EntityBlock JSON serialization."""
    block = EntityBlock(
        block_key="test",
        block_key_type="semantic",
        block_size=1,
        entities=[Entity(id=1, name="Test")],
    )
    data = json.loads(block.model_dump_json())
    assert data["block_key"] == "test"
    assert len(data["entities"]) == 1
