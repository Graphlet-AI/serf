"""Tests for EntityMerger."""

from unittest.mock import MagicMock, patch

import pytest

from serf.dspy.types import Entity
from serf.merge.merger import EntityMerger
from serf.schema import parse_schema


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


def test_merge_pair_deduplicates_source_ids() -> None:
    """merge_pair deduplicates source_ids."""
    a = Entity(id=1, name="A", source_ids=[3, 5])
    b = Entity(id=2, name="B", source_ids=[3, 7])
    merger = EntityMerger()
    result = merger.merge_pair(a, b)
    assert result.source_ids is not None
    assert len(result.source_ids) == len(set(result.source_ids))
    assert set(result.source_ids) == {2, 3, 5, 7}


def test_merge_pair_excludes_master_id_from_source_ids() -> None:
    """Master id is excluded from source_ids."""
    a = Entity(id=1, name="A")
    b = Entity(id=2, name="B", source_ids=[1])
    merger = EntityMerger()
    result = merger.merge_pair(a, b)
    assert result.id == 1
    assert 1 not in (result.source_ids or [])


def test_merge_pair_deduplicates_source_uuids() -> None:
    """merge_pair deduplicates source_uuids."""
    a = Entity(id=1, name="A", uuid="uuid-a", source_uuids=["uuid-x"])
    b = Entity(id=2, name="B", uuid="uuid-b", source_uuids=["uuid-x"])
    merger = EntityMerger()
    result = merger.merge_pair(a, b)
    assert result.source_uuids is not None
    assert len(result.source_uuids) == len(set(result.source_uuids))
    assert "uuid-x" in result.source_uuids
    assert "uuid-b" in result.source_uuids
    assert "uuid-a" not in result.source_uuids


def test_merger_deduplicates_list_fields_for_addresses_and_names() -> None:
    """List attributes such as addresses and names deduplicate into lists of individual items."""
    a = Entity(
        id=1,
        name="Russell Jurney",
        attributes={
            "addresses": ["123 Main St", "500 Pine Blvd, Suite 400, Austin, TX"],
            "names": ["Russell Jurney", "Russ Journey"],
            "phones": ["(415) 555-0100"],
        },
    )
    b = Entity(
        id=2,
        name="Russell H Jurney",
        attributes={
            "addresses": ["123 Main Street", "900 Madison Ave, New York, NY"],
            "names": ["Russell H Jurney", "Bob Dorf"],
            "phones": ["4155550100", "212-555-0199"],
        },
    )
    merger = EntityMerger(
        field_types={"addresses": "address", "names": "name", "phones": "phone"},
        list_fields={"addresses", "names", "phones"},
    )
    result = merger.merge_pair(a, b)

    assert result.name == "Russell H Jurney"
    assert result.attributes["addresses"] == [
        "500 Pine Blvd, Suite 400, Austin, TX",
        "900 Madison Ave, New York, NY",
        "123 Main Street",
    ]
    assert result.attributes["names"] == ["Russell H Jurney", "Bob Dorf"]
    assert result.attributes["phones"] == ["(415) 555-0100", "212-555-0199"]


def test_merger_promotes_scalar_inputs_for_configured_list_fields() -> None:
    """When a field is configured as a list field, scalar inputs across entities become a deduplicated list."""
    a = Entity(id=1, name="Acme", attributes={"address": "123 Main St"})
    b = Entity(id=2, name="Acme Corp", attributes={"address": "123 Main Street"})
    c = Entity(id=3, name="Acme Inc", attributes={"address": "456 Oak Avenue"})

    merger = EntityMerger(
        field_types={"address": "address"},
        list_fields={"address"},
    )
    result = merger.merge_entities([a, b, c])

    assert result.attributes["address"] == ["123 Main Street", "456 Oak Avenue"]


def test_merger_selects_best_value_and_deduplicates_lists_via_dspy_annotations() -> None:
    """Singular and list fields are merged via the DSPy annotated signature when use_llm is enabled."""
    fake_prediction = MagicMock()
    fake_prediction.best_value = "International Business Machines"
    fake_prediction.legal_name = "International Business Machines Corporation"
    fake_prediction.addresses = [
        "1 New Orchard Road, Armonk, NY",
        "590 Madison Ave, New York, NY",
    ]
    fake_predictor = MagicMock(return_value=fake_prediction)

    schema = parse_schema(
        {
            "entity": "Organization",
            "fields": [
                {"name": "legal_name", "type": "name", "cardinality": "single"},
                {"name": "addresses", "type": "address", "cardinality": "list"},
            ],
        }
    )
    a = Entity(
        id=1,
        name="IBM",
        attributes={
            "legal_name": "IBM Corp.",
            "addresses": ["1 New Orchard Rd, Armonk, NY"],
        },
    )
    b = Entity(
        id=2,
        name="International Business Machines",
        attributes={
            "legal_name": "International Business Machines Corporation",
            "addresses": ["1 New Orchard Road, Armonk, NY", "590 Madison Ave, New York, NY"],
        },
    )

    with patch("serf.merge.semantics.dspy.Predict", return_value=fake_predictor):
        merger = EntityMerger(schema=schema, use_llm=True)
        result = merger.merge_pair(a, b)

    assert result.attributes["legal_name"] == "International Business Machines Corporation"
    assert result.attributes["addresses"] == [
        "1 New Orchard Road, Armonk, NY",
        "590 Madison Ave, New York, NY",
    ]
