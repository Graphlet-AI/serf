"""Tests for RobustXMLAdapter's ampersand-escaping fix."""

from serf.dspy.adapters import escape_bare_ampersands


def test_bare_ampersand_gets_escaped() -> None:
    """A literal '&' with no following entity name is escaped."""
    assert escape_bare_ampersands("Black & White") == "Black &amp; White"


def test_already_escaped_ampersand_is_left_alone() -> None:
    """An already-valid &amp; is not double-escaped."""
    assert escape_bare_ampersands("Black &amp; White") == "Black &amp; White"


def test_named_entities_are_left_alone() -> None:
    """Other standard XML named entities are recognized and untouched."""
    for entity in ("&lt;", "&gt;", "&quot;", "&apos;"):
        text = f"before {entity} after"
        assert escape_bare_ampersands(text) == text


def test_numeric_entities_are_left_alone() -> None:
    """Decimal and hex numeric character references are untouched."""
    assert escape_bare_ampersands("&#65;") == "&#65;"
    assert escape_bare_ampersands("&#x41;") == "&#x41;"


def test_multiple_bare_ampersands_all_get_escaped() -> None:
    """Every bare ampersand in a string is escaped, not just the first."""
    result = escape_bare_ampersands("A & B & C")
    assert result == "A &amp; B &amp; C"


def test_mixed_bare_and_valid_ampersands() -> None:
    """A string mixing already-valid and bare ampersands only fixes the bare ones."""
    result = escape_bare_ampersands("Tom &amp; Jerry & Friends")
    assert result == "Tom &amp; Jerry &amp; Friends"
