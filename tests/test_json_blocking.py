"""Tests for JSON record text used by the json blocking strategy."""

import json

from serf.dspy.types import Entity


def test_json_strips_the_source_side_prefix() -> None:
    """Left and right records describe the same field with the same key.

    Every gold pair crosses sources, so a key that encodes which side a record
    came from would push the two sides apart in embedding space.
    """
    left = Entity(
        id=1,
        name="acme widget",
        attributes={"l_title": "acme widget", "l_brand": "acme", "l_price": "9.99"},
    )
    right = Entity(
        id=2,
        name="acme widget xl",
        attributes={"r_title": "acme widget xl", "r_brand": "acme", "r_price": "12.99"},
    )

    assert set(json.loads(left.json_for_embedding())) == {"title", "brand", "price"}
    assert set(json.loads(right.json_for_embedding())) == {"title", "brand", "price"}


def test_json_drops_record_identifiers() -> None:
    """Ids come from unrelated namespaces per source, so they are noise."""
    entity = Entity(
        id=1,
        name="acme widget",
        attributes={"l_id": "journals/sigmod/Mackay99", "l_title": "acme widget"},
    )

    record = json.loads(entity.json_for_embedding())

    assert "id" not in record
    assert "journals/sigmod/Mackay99" not in record.values()


def test_json_drops_empty_values_and_sorts_keys() -> None:
    """Blank fields add nothing, and stable key order keeps runs comparable."""
    entity = Entity(
        id=1,
        name="acme widget",
        attributes={"l_title": "acme widget", "l_brand": "", "l_price": None, "l_year": "1999"},
    )

    text = entity.json_for_embedding()

    assert text == '{"title": "acme widget", "year": "1999"}'


def test_json_includes_the_name_when_attributes_omit_it() -> None:
    """Datasets that carry no title column still get the name embedded."""
    entity = Entity(id=1, name="acme widget", attributes={"l_brand": "acme"})

    record = json.loads(entity.json_for_embedding())

    assert record["name"] == "acme widget"


def test_json_does_not_repeat_the_name_already_present() -> None:
    """A title column holding the name is not duplicated under a second key."""
    entity = Entity(
        id=1, name="acme widget", attributes={"l_title": "acme widget", "l_brand": "acme"}
    )

    record = json.loads(entity.json_for_embedding())

    assert list(record.values()).count("acme widget") == 1


def test_json_field_names_appear_in_the_text() -> None:
    """The point of the strategy is that the model reads the field names."""
    entity = Entity(id=1, name="acme widget", attributes={"l_brand": "acme", "l_price": "9.99"})

    text = entity.json_for_embedding()

    assert '"brand"' in text
    assert '"price"' in text
