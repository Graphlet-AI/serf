"""Tests for EntityMerger."""

import pytest

from serf.dspy.types import Entity
from serf.merge.merger import EntityMerger


def test_merge_pair_lowest_id_becomes_master() -> None:
    """Lowest id becomes master in merge_pair."""
    a = Entity(id=22, name="Acme Corp", description="Big company")
    b = Entity(id=1, name="ACME Corporation", description="A big company")
    merger = EntityMerger()
    result = merger.merge_pair(a, b)
    assert result.id == 1
    assert 22 in (result.source_ids or [])


def test_merge_pair_accumulates_source_ids() -> None:
    """merge_pair accumulates source_ids from both entities."""
    a = Entity(id=1, name="A", source_ids=[3, 7])
    b = Entity(id=22, name="B", source_ids=[2, 4])
    merger = EntityMerger()
    result = merger.merge_pair(a, b)
    assert 22 in (result.source_ids or [])
    assert 2 in (result.source_ids or [])
    assert 4 in (result.source_ids or [])
    assert 3 in (result.source_ids or [])
    assert 7 in (result.source_ids or [])


def test_merge_pair_accumulates_source_uuids() -> None:
    """merge_pair accumulates source_uuids from both entities."""
    a = Entity(id=1, name="A", uuid="uuid-a")
    b = Entity(id=2, name="B", uuid="uuid-b")
    merger = EntityMerger()
    result = merger.merge_pair(a, b)
    assert "uuid-b" in (result.source_uuids or [])


def test_merge_pair_picks_longest_string() -> None:
    """merge_pair picks the longest non-empty string for fields."""
    a = Entity(id=1, name="Acme", description="Short")
    b = Entity(id=2, name="A", description="A longer description")
    merger = EntityMerger()
    result = merger.merge_pair(a, b)
    assert result.name == "Acme"
    assert result.description == "A longer description"


def test_merge_entities_single_returns_unchanged() -> None:
    """merge_entities with one entity returns it unchanged."""
    e = Entity(id=1, name="A")
    merger = EntityMerger()
    assert merger.merge_entities([e]) is e


def test_merge_entities_multiple() -> None:
    """merge_entities merges all entities."""
    entities = [
        Entity(id=5, name="E5"),
        Entity(id=1, name="E1"),
        Entity(id=3, name="E3"),
    ]
    merger = EntityMerger()
    result = merger.merge_entities(entities)
    assert result.id == 1
    assert set(result.source_ids or []) == {3, 5}


def test_merge_entities_empty_raises() -> None:
    """merge_entities with empty list raises ValueError."""
    merger = EntityMerger()
    with pytest.raises(ValueError, match="empty"):
        merger.merge_entities([])
