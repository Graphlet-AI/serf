"""Tests for the multi-iteration helpers that feed one ER round into the next.

A single matching pass can only pair records that blocking already put in the same
block. Later iterations re-block the entities merged by the previous round, so the
loop needs to merge on the predicted pairs and to read a match against a merged
entity as a match against every record that entity carries.
"""

from serf.dspy.types import Entity
from serf.match.run import entity_members, expand_pairs, merge_matched_entities


def _entity(entity_id: int, name: str, source_ids: list[int] | None = None) -> Entity:
    """Build an entity, optionally one that already absorbed other records.

    Parameters
    ----------
    entity_id : int
        Entity id
    name : str
        Entity name
    source_ids : list[int] | None
        Record ids this entity already stands for

    Returns
    -------
    Entity
        Entity for the tests
    """
    return Entity(
        id=entity_id,
        name=name,
        entity_type="product",
        source_ids=source_ids or [],
    )


def test_entity_members_covers_the_entity_and_everything_it_absorbed() -> None:
    """A merged entity stands for its own id plus its source ids."""
    entities = [_entity(1, "a", [100_001]), _entity(2, "b")]

    assert entity_members(entities) == {1: {1, 100_001}, 2: {2}}


def test_expand_pairs_crosses_every_member_of_both_sides() -> None:
    """Matching two merged entities asserts a match between all their records."""
    members = {1: {1, 100_001}, 2: {2, 100_002}}

    assert expand_pairs({(1, 2)}, members) == {
        (1, 2),
        (1, 100_002),
        (2, 100_001),
        (100_001, 100_002),
    }


def test_expand_pairs_leaves_unmerged_pairs_alone() -> None:
    """A pair of plain records expands to itself."""
    assert expand_pairs({(1, 2)}, {1: {1}, 2: {2}}) == {(1, 2)}


def test_expand_pairs_drops_self_pairs() -> None:
    """Two entities that already share a record produce no self pair."""
    assert expand_pairs({(1, 2)}, {1: {1, 5}, 2: {2, 5}}) == {(1, 2), (1, 5), (2, 5)}


def test_merge_matched_entities_collapses_a_connected_component() -> None:
    """Records joined transitively by predicted pairs become one entity."""
    entities = [_entity(1, "acme"), _entity(2, "acme inc"), _entity(3, "other")]

    merged = merge_matched_entities(entities, {(1, 2)})

    assert [e.id for e in merged] == [1, 3]
    assert set(merged[0].source_ids or []) == {2}


def test_merge_matched_entities_chains_pairs_through_a_shared_record() -> None:
    """A -- B and B -- C collapse into a single entity carrying both."""
    entities = [_entity(1, "a"), _entity(2, "b"), _entity(3, "c")]

    merged = merge_matched_entities(entities, {(1, 2), (2, 3)})

    assert len(merged) == 1
    assert merged[0].id == 1
    assert set(merged[0].source_ids or []) == {2, 3}


def test_merge_matched_entities_keeps_earlier_source_ids() -> None:
    """Merging again does not lose the records absorbed in an earlier round."""
    entities = [_entity(1, "a", [100_001]), _entity(2, "b", [100_002])]

    merged = merge_matched_entities(entities, {(1, 2)})

    assert len(merged) == 1
    assert set(merged[0].source_ids or []) == {2, 100_001, 100_002}


def test_merge_matched_entities_without_pairs_is_a_no_op() -> None:
    """A round that predicted nothing hands the same entities forward."""
    entities = [_entity(1, "a"), _entity(2, "b")]

    merged = merge_matched_entities(entities, set())

    assert [e.id for e in merged] == [1, 2]


def test_merge_matched_entities_ignores_pairs_for_absent_records() -> None:
    """Pairs naming records outside the current entity set are skipped."""
    entities = [_entity(1, "a"), _entity(2, "b")]

    merged = merge_matched_entities(entities, {(1, 999)})

    assert [e.id for e in merged] == [1, 2]
