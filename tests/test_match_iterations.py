"""Tests for the multi-iteration helpers that feed one ER round into the next.

A single matching pass can only pair records that blocking already put in the same
block. Later iterations re-block the entities merged by the previous round, so the
loop needs to merge on the predicted pairs and to read a match against a merged
entity as a match against every record that entity carries.
"""

from serf.dspy.types import Entity
from serf.match.run import entity_members, expand_pairs, merge_matched_entities


def _uuid_for(entity_id: int) -> str:
    """Return a readable stand-in uuid for a source key.

    Parameters
    ----------
    entity_id : int
        Source key the record arrived with

    Returns
    -------
    str
        A uuid-shaped string unique to that key
    """
    return f"{entity_id:08d}-0000-4000-8000-000000000000"


def _entity(entity_id: int, name: str, source_ids: list[int] | None = None) -> Entity:
    """Build an entity, optionally one that already absorbed other records.

    Parameters
    ----------
    entity_id : int
        Source key this record arrived with. Identity is the uuid derived from
        it; the integer stays only so the benchmark gold standard can be
        matched against the record.
    name : str
        Entity name
    source_ids : list[int] | None
        Source keys this entity already stands for

    Returns
    -------
    Entity
        Entity for the tests
    """
    return Entity(
        id=entity_id,
        uuid=_uuid_for(entity_id),
        name=name,
        entity_type="product",
        source_ids=source_ids or [],
        source_uuids=[_uuid_for(value) for value in source_ids or []],
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
    """Records joined transitively become one entity with a minted identity."""
    entities = [_entity(1, "acme"), _entity(2, "acme inc"), _entity(3, "other")]

    merged = merge_matched_entities(entities, {(1, 2)})

    by_uuid = {e.uuid: e for e in merged}
    assert len(merged) == 2
    assert _uuid_for(3) in by_uuid, "an entity that merged with nothing keeps its uuid"
    minted = next(value for value in by_uuid if value != _uuid_for(3))
    assert minted not in {_uuid_for(1), _uuid_for(2)}, "a merge is a new entity"
    assert set(by_uuid[minted].source_uuids or []) == {_uuid_for(1), _uuid_for(2)}


def test_merge_matched_entities_chains_pairs_through_a_shared_record() -> None:
    """A -- B and B -- C collapse into a single entity carrying both."""
    entities = [_entity(1, "a"), _entity(2, "b"), _entity(3, "c")]

    merged = merge_matched_entities(entities, {(1, 2), (2, 3)})

    assert len(merged) == 1
    assert merged[0].uuid not in {_uuid_for(1), _uuid_for(2), _uuid_for(3)}
    assert set(merged[0].source_uuids or []) == {_uuid_for(1), _uuid_for(2), _uuid_for(3)}


def test_merge_matched_entities_keeps_earlier_lineage() -> None:
    """Merging again does not lose the records absorbed in an earlier round."""
    entities = [_entity(1, "a", [100_001]), _entity(2, "b", [100_002])]

    merged = merge_matched_entities(entities, {(1, 2)})

    absorbed = {_uuid_for(1), _uuid_for(2), _uuid_for(100_001), _uuid_for(100_002)}
    assert len(merged) == 1
    assert set(merged[0].source_uuids or []) == absorbed
    assert merged[0].uuid not in absorbed


def test_a_merged_entity_never_takes_an_identity_another_record_holds() -> None:
    """A minted uuid cannot collide, which an integer minted from one block could."""
    entities = [_entity(1, "a"), _entity(2, "a"), _entity(4, "other"), _entity(5, "other too")]

    merged = merge_matched_entities(entities, {(1, 2)})

    identities = {e.uuid for e in merged}
    assert identities & {_uuid_for(1), _uuid_for(2)} == set()
    assert {_uuid_for(4), _uuid_for(5)} <= identities


def test_every_original_record_is_still_reachable_after_a_merge() -> None:
    """Conservation: nothing the pipeline was handed may become unreachable."""
    entities = [_entity(1, "a", [100_001]), _entity(2, "b"), _entity(3, "c")]

    merged = merge_matched_entities(entities, {(1, 2)})

    reachable = {value for e in merged for value in ([e.uuid] + list(e.source_uuids or []))}
    assert {_uuid_for(1), _uuid_for(2), _uuid_for(3), _uuid_for(100_001)} <= reachable


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


def test_minting_across_rounds_never_repeats_an_identity() -> None:
    """No allocator to thread: a uuid is unique without knowing what came before."""
    first = merge_matched_entities(
        [_entity(1, "a"), _entity(2, "a"), _entity(3, "b"), _entity(4, "b")],
        {(1, 2), (3, 4)},
    )
    second = merge_matched_entities(first, {(first[0].id, first[1].id)})

    seen = {e.uuid for e in first} | {e.uuid for e in second}
    assert len(seen) == len([e.uuid for e in first]) + len([e.uuid for e in second])


def test_a_second_round_merge_records_the_intermediate_entity_in_its_lineage() -> None:
    """Lineage is an audit trail, so the entity that existed in between is kept."""
    first = merge_matched_entities([_entity(1, "a"), _entity(2, "a"), _entity(3, "a")], {(1, 2)})
    intermediate = next(e for e in first if e.source_uuids)
    second = merge_matched_entities(first, {(intermediate.id, 3)})

    assert len(second) == 1
    assert set(second[0].source_uuids or []) == {
        _uuid_for(1),
        _uuid_for(2),
        _uuid_for(3),
        intermediate.uuid,
    }
