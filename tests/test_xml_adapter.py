"""Tests for the XML adapter that repairs unescaped metacharacters."""

import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError

from serf.dspy.adapter import RepairingXMLAdapter, escape_stray_metacharacters
from serf.dspy.signatures import BlockMatch


def test_escapes_bare_ampersand() -> None:
    """A bare ampersand in text becomes an entity."""
    assert escape_stray_metacharacters("<name>Home & Student</name>") == (
        "<name>Home &amp; Student</name>"
    )


def test_leaves_existing_entities_alone() -> None:
    """XML's predefined names and numeric character references survive untouched."""
    text = "<name>A&amp;B &#38; C &#x26; D &lt;E&gt; &quot;F&quot; &apos;G&apos;</name>"
    assert escape_stray_metacharacters(text) == text


def test_escapes_html_entities_xml_does_not_define() -> None:
    """XML predefines five names, so every other named entity is undefined in it.

    ACM titles carry raw HTML entities, and ElementTree rejects the whole
    document with "undefined entity" rather than passing them through.
    """
    assert escape_stray_metacharacters("<title>The VLDB Journal &mdash; Volume 6</title>") == (
        "<title>The VLDB Journal &amp;mdash; Volume 6</title>"
    )
    assert escape_stray_metacharacters("<name>Oliver G&uuml;nther</name>") == (
        "<name>Oliver G&amp;uuml;nther</name>"
    )


def test_parses_completion_with_an_html_entity() -> None:
    """A block whose echoed text carries an HTML entity must not be lost."""
    completion = (
        "<resolution>"
        "<block_key>block_0</block_key>"
        "<matches></matches>"
        "<resolved_entities><item>"
        "<id>0</id><uuid>null</uuid>"
        "<name>The VLDB Journal &mdash; Volume 6</name>"
        "<description>Oliver G&uuml;nther</description>"
        "<entity_type>entity</entity_type>"
        "<attributes>{}</attributes>"
        "<source_ids><item>1</item></source_ids>"
        "<source_uuids></source_uuids>"
        "<match_skip>null</match_skip>"
        "<match_skip_reason></match_skip_reason>"
        "<match_skip_history></match_skip_history>"
        "</item></resolved_entities>"
        "<was_resolved>true</was_resolved>"
        "<original_count>1</original_count>"
        "<resolved_count>1</resolved_count>"
        "</resolution>"
    )

    with pytest.raises(AdapterParseError):
        dspy.XMLAdapter().parse(BlockMatch, completion)

    entity = RepairingXMLAdapter().parse(BlockMatch, completion)["resolution"].resolved_entities[0]
    assert entity.name == "The VLDB Journal &mdash; Volume 6"
    assert entity.description == "Oliver G&uuml;nther"


def test_escapes_bare_less_than_but_not_tags() -> None:
    """A less-than that does not open a tag becomes an entity."""
    assert escape_stray_metacharacters("<desc>weighs < 30 lbs</desc>") == (
        "<desc>weighs &lt; 30 lbs</desc>"
    )
    assert escape_stray_metacharacters("<a><b/></a>") == "<a><b/></a>"


def test_parses_completion_with_unescaped_ampersand() -> None:
    """The adapter recovers a block that ElementTree rejects outright."""
    completion = (
        "<resolution>"
        "<block_key>block_0</block_key>"
        "<matches><item>"
        "<entity_a_id>0</entity_a_id><entity_b_id>1</entity_b_id>"
        "<is_match>true</is_match><confidence>0.95</confidence>"
        "<reasoning>Both are Office Home &amp; Student</reasoning>"
        "</item></matches>"
        "<resolved_entities><item>"
        "<id>0</id><uuid>null</uuid>"
        "<name>Office 2008 Home & Student</name>"
        "<description>AT&T exclusive</description>"
        "<entity_type>entity</entity_type>"
        '<attributes>{"price": "$99.00"}</attributes>'
        "<source_ids><item>1</item></source_ids>"
        "<source_uuids></source_uuids>"
        "<match_skip>null</match_skip>"
        "<match_skip_reason></match_skip_reason>"
        "<match_skip_history></match_skip_history>"
        "</item></resolved_entities>"
        "<was_resolved>true</was_resolved>"
        "<original_count>2</original_count>"
        "<resolved_count>1</resolved_count>"
        "</resolution>"
    )

    with pytest.raises(AdapterParseError):
        dspy.XMLAdapter().parse(BlockMatch, completion)

    resolution = RepairingXMLAdapter().parse(BlockMatch, completion)["resolution"]
    entity = resolution.resolved_entities[0]
    assert entity.name == "Office 2008 Home & Student"
    assert entity.description == "AT&T exclusive"
    assert entity.attributes == {"price": "$99.00"}
    assert entity.uuid is None
    assert entity.match_skip is None
    assert entity.source_ids == [1]


def test_reraises_when_escaping_changes_nothing() -> None:
    """A completion with no stray metacharacters fails with its own error."""
    with pytest.raises(AdapterParseError):
        RepairingXMLAdapter().parse(BlockMatch, "<resolution><matches>")
