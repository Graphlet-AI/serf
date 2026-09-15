"""Tests for entity schema YAML and the Pydantic classes generated from it."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from serf.dspy.schemas.base import EntitySide, field_guide
from serf.merge.canonical import canonicalize_groups
from serf.merge.semantics import DEDUPE_EXACT, DEDUPE_FUZZY, DEDUPE_NONE
from serf.schema import (
    CanonicalEntity,
    EntitySchema,
    canonical_model,
    clear_cache,
    load_schema,
    load_schemas,
    parse_schema,
    source_model,
)

EXAMPLE_SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "person.yml"


@pytest.fixture(autouse=True)
def _fresh_generated_classes() -> None:
    """Generated classes are cached by schema, so rebuilds must start clean."""
    clear_cache()


def _schema() -> EntitySchema:
    return load_schema(EXAMPLE_SCHEMA)


def test_the_shipped_example_schema_loads() -> None:
    schema = _schema()

    assert schema.entity == "Person"
    assert [field.name for field in schema.fields][:3] == ["name", "state", "nation"]


def test_a_schema_directory_loads_by_entity_name() -> None:
    schemas = load_schemas(EXAMPLE_SCHEMA.parent)

    assert "Person" in schemas


def test_field_types_drive_canonicalization() -> None:
    schema = _schema()

    assert schema.field_types()["bio"] == "text"
    assert schema.field_types()["name"] == "name"


def test_a_schema_override_beats_the_field_types_default_policy() -> None:
    """`address` is fuzzy by default; this schema makes state codes exact."""
    schema = _schema()

    assert schema.merge_policies()["state"].dedupe == DEDUPE_EXACT
    assert schema.merge_policies()["employer"].dedupe == DEDUPE_FUZZY
    assert schema.merge_policies()["bio"].dedupe == DEDUPE_NONE


def test_the_override_actually_changes_what_merges() -> None:
    """Without the exact override, two state codes one letter apart would collapse."""
    schema = _schema()
    groups = [
        [
            {"uuid": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "state": "CA"},
            {"uuid": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "state": "GA"},
        ]
    ]

    merged = canonicalize_groups(
        groups, field_types=schema.field_types(), policies=schema.merge_policies()
    )[0]

    assert merged.fields["state"] == ["CA", "GA"]


def test_the_source_model_is_an_entity_side_with_scalar_fields() -> None:
    model = source_model(_schema())

    assert issubclass(model, EntitySide)
    assert model.model_fields["name"].annotation is str
    assert "record_id" in model.model_fields


def test_the_canonical_model_makes_every_field_a_list() -> None:
    model = canonical_model(_schema())

    assert issubclass(model, CanonicalEntity)
    assert model.model_fields["name"].annotation == list[str]
    assert model.model_fields["uuid"].annotation is str
    assert model.model_fields["source_uuids"].annotation == list[str]


def test_a_canonical_record_round_trips_through_the_generated_model() -> None:
    model = canonical_model(_schema())

    record = model(
        uuid="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
        source_uuids=["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"],
        name=["Russell H Jurney"],
        state=["CA", "WA"],
    )

    assert record.model_dump()["source_uuids"] == ["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"]
    assert record.model_dump()["name"] == ["Russell H Jurney"]


def test_descriptions_reach_the_prompt_through_the_field_guide() -> None:
    """Field descriptions are the whole point of declaring a schema."""
    guide = field_guide(source_model(_schema()))

    assert "Two-letter ISO country code" in guide


def test_generating_twice_returns_the_same_class() -> None:
    """DSPy compares signature types by identity, so rebuilds must be stable."""
    schema = _schema()

    assert source_model(schema) is source_model(schema)
    assert canonical_model(schema) is canonical_model(schema)


def test_a_required_field_has_no_default() -> None:
    model = source_model(_schema())

    assert model.model_fields["name"].is_required()
    assert not model.model_fields["state"].is_required()


def test_numeric_fields_default_to_float() -> None:
    schema = parse_schema({"entity": "Thing", "fields": [{"name": "price", "type": "numeric"}]})

    assert source_model(schema).model_fields["price"].annotation == float | None


def test_an_explicit_python_type_wins() -> None:
    schema = parse_schema(
        {"entity": "Thing", "fields": [{"name": "year", "type": "numeric", "python_type": "int"}]}
    )

    assert source_model(schema).model_fields["year"].annotation == int | None


def test_an_unknown_field_type_is_rejected_at_load() -> None:
    """A typo must fail where it is written, not silently merge the wrong way."""
    with pytest.raises(ValidationError, match="type must be one of"):
        parse_schema({"entity": "Thing", "fields": [{"name": "x", "type": "nonsense"}]})


def test_an_unknown_dedupe_strategy_is_rejected_at_load() -> None:
    with pytest.raises(ValidationError, match="dedupe must be one of"):
        parse_schema(
            {"entity": "Thing", "fields": [{"name": "x", "merge": {"dedupe": "nonsense"}}]}
        )


def test_an_unknown_python_type_is_rejected_at_load() -> None:
    with pytest.raises(ValidationError, match="python_type must be one of"):
        parse_schema({"entity": "Thing", "fields": [{"name": "x", "python_type": "complex"}]})


def test_duplicate_field_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate field names"):
        parse_schema({"entity": "Thing", "fields": [{"name": "x"}, {"name": "x"}]})


def test_a_schema_file_that_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_schema(path)


def test_a_schema_written_by_hand_survives_a_yaml_round_trip(tmp_path: Path) -> None:
    schema = _schema()
    path = tmp_path / "person.yml"
    path.write_text(yaml.safe_dump(schema.model_dump(exclude_none=True)), encoding="utf-8")

    assert load_schema(path) == schema
