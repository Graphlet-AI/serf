"""Tests for the identifier-conservation contract (docs/ID_INVARIANTS.md).

Each test is named for the invariant it protects and mirrors one of the
nine required tests in ID_INVARIANTS.md Section 9. These tests must fail
if the contract is broken; several fail against the pre-fix implementation
(see the divergences D1-D8 in that document's Section 8).
"""

from unittest.mock import patch

from serf.dspy.types import BlockResolution, Entity, EntityBlock
from serf.match.matcher import EntityMatcher
from serf.match.uuid_mapper import UUIDMapper


def test_round_trip_identifier_conservation() -> None:
    """Every input identifier appears exactly once in the output: as a
    master id, or inside exactly one resolved entity's source_ids."""
    block = EntityBlock(
        block_key="b1",
        block_size=4,
        entities=[
            Entity(id=10, name="A"),
            Entity(id=20, name="B"),
            Entity(id=30, name="C"),
            Entity(id=40, name="D"),
        ],
    )
    mapper = UUIDMapper()
    mapped = mapper.map_block(block)
    # LLM merges mapped 0+1 into master 0, and returns mapped 2 (D) untouched;
    # mapped 3 is omitted entirely (dropped, recovered via Phase 2).
    resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[
            Entity(id=mapped.entities[0].id, name="A", source_ids=[mapped.entities[1].id]),
            Entity(id=mapped.entities[2].id, name="C"),
        ],
        was_resolved=True,
        original_count=4,
        resolved_count=2,
    )
    restored = mapper.unmap_block(resolution, block)

    # Phase-2-recovered entities are the documented exception to "a master's
    # own id is excluded from its own source_ids" (ID_INVARIANTS.md Section 5,
    # Phase 2, point 2): they self-reference by design, so they're accounted
    # for separately rather than folded into the general master/merge tally.
    recovered_ids = {
        e.id for e in restored.resolved_entities if e.match_skip_reason == "missing_in_match_output"
    }
    normal_ids = {e.id for e in restored.resolved_entities} - recovered_ids
    all_source_ids: list[int] = []
    for e in restored.resolved_entities:
        if e.id in recovered_ids:
            continue
        all_source_ids.extend(e.source_ids or [])

    input_ids = {10, 20, 30, 40}
    for input_id in input_ids:
        if input_id in recovered_ids:
            continue
        in_master = input_id in normal_ids
        occurrences = all_source_ids.count(input_id)
        if in_master:
            assert occurrences == 0, f"master {input_id} must not also appear in a source_ids list"
        else:
            assert occurrences == 1, (
                f"non-master {input_id} must appear in exactly one source_ids list, found {occurrences}"
            )
    # No identifier duplicated across multiple source_ids lists.
    assert len(all_source_ids) == len(set(all_source_ids))


def test_dropped_record_recovered_with_self_provenance() -> None:
    """A mock LLM omits a record entirely. It comes back with
    match_skip_reason == 'missing_in_match_output', its original field
    values, and its own identifier in its own provenance list."""
    block = EntityBlock(
        block_key="b1",
        block_size=2,
        entities=[
            Entity(id=100, name="A", description="original description"),
            Entity(id=200, name="B"),
        ],
    )
    mapper = UUIDMapper()
    mapped = mapper.map_block(block)
    # LLM returns only entity B; A is dropped entirely.
    resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[Entity(id=mapped.entities[1].id, name="B")],
        was_resolved=True,
        original_count=2,
        resolved_count=1,
    )
    restored = mapper.unmap_block(resolution, block)

    recovered = [e for e in restored.resolved_entities if e.id == 100]
    assert len(recovered) == 1
    recovered_entity = recovered[0]
    assert recovered_entity.match_skip_reason == "missing_in_match_output"
    assert recovered_entity.name == "A"
    assert recovered_entity.description == "original description"
    assert 100 in (recovered_entity.source_ids or [])


def test_dropped_provenance_restored_via_phase_one_recovery() -> None:
    """A mock LLM returns a record but truncates its source_ids (drops
    part of its merge history from an earlier round). Phase 1 restores
    the missing entries from the cached input, without a full re-emit."""
    block = EntityBlock(
        block_key="b1",
        block_size=3,
        entities=[
            Entity(id=100, name="A merged-in-round-1", source_ids=[50, 51]),
            Entity(id=200, name="B"),
            Entity(id=300, name="C"),
        ],
    )
    mapper = UUIDMapper()
    mapped = mapper.map_block(block)
    # The LLM merges A (mapped 0) and B (mapped 1) this round, but its
    # source_ids only mentions B (1) -- it has no way to know about A's
    # round-1 provenance [50, 51], since that was stripped before the call.
    resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[
            Entity(id=mapped.entities[0].id, name="A+B", source_ids=[mapped.entities[1].id]),
            Entity(id=mapped.entities[2].id, name="C"),
        ],
        was_resolved=True,
        original_count=3,
        resolved_count=2,
    )
    restored = mapper.unmap_block(resolution, block)

    master = next(e for e in restored.resolved_entities if e.id == 100)
    assert 200 in (master.source_ids or [])
    assert 50 in (master.source_ids or [])
    assert 51 in (master.source_ids or [])


def test_second_round_provenance_survives_without_id_collision() -> None:
    """A record carrying source_ids from round 1 goes through round 2.
    Its round-1 provenance survives, and none of those identifiers is
    resolved against a block-local record from round 2."""
    # Round-2 block: entity 100 already carries round-1 provenance [2],
    # and this round's OTHER entities happen to map to block-local int 2,
    # so a stale, unmapped source_id sent to the LLM would collide with
    # an unrelated real entity.
    block = EntityBlock(
        block_key="b1",
        block_size=4,
        entities=[
            Entity(id=100, name="A", source_ids=[2]),  # round-1 provenance
            Entity(id=500, name="unrelated-1"),
            Entity(id=2, name="unrelated-2"),  # real id 2 -- collision bait
            Entity(id=600, name="unrelated-3"),
        ],
    )
    mapper = UUIDMapper()
    mapped = mapper.map_block(block)

    # The stale source_ids=[2] must never reach the LLM at all.
    mapped_a = mapped.entities[0]
    assert mapped_a.source_ids is None or mapped_a.source_ids == []

    # LLM returns everyone untouched (no merges this round).
    resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[e.model_copy() for e in mapped.entities],
        was_resolved=False,
        original_count=4,
        resolved_count=4,
    )
    restored = mapper.unmap_block(resolution, block)

    restored_a = next(e for e in restored.resolved_entities if e.id == 100)
    restored_unrelated_2 = next(e for e in restored.resolved_entities if e.id == 2)
    # A's round-1 provenance [2] must survive as-is...
    assert 2 in (restored_a.source_ids or [])
    # ...and must NOT be confused with the real, unrelated entity id=2,
    # which must come back untouched with no borrowed provenance.
    assert not (restored_unrelated_2.source_ids or [])


def test_transitive_accumulation_of_source_ids() -> None:
    """Master id=1 source_ids=[3,7] merged with id=22 source_ids=[2,4]
    yields master id=1 with source_ids == {22,3,7,2,4} and 1 is not in
    its own list."""
    block = EntityBlock(
        block_key="b1",
        block_size=2,
        entities=[
            Entity(id=1, name="A", source_ids=[3, 7]),
            Entity(id=22, name="B", source_ids=[2, 4]),
        ],
    )
    mapper = UUIDMapper()
    mapped = mapper.map_block(block)
    resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[
            Entity(id=mapped.entities[0].id, name="A+B", source_ids=[mapped.entities[1].id]),
        ],
        was_resolved=True,
        original_count=2,
        resolved_count=1,
    )
    restored = mapper.unmap_block(resolution, block)
    master = restored.resolved_entities[0]
    assert master.id == 1
    assert set(master.source_ids or []) == {22, 3, 7, 2, 4}
    assert 1 not in (master.source_ids or [])


def test_singleton_short_circuit_skips_llm_call() -> None:
    """A block of one produces no LLM call and one output record with
    match_skip_reason == 'singleton_block' and its original identifier."""
    matcher = EntityMatcher()
    block = EntityBlock(block_key="b1", block_size=1, entities=[Entity(id=100, name="A")])

    with patch.object(EntityMatcher, "_ensure_lm") as mock_ensure_lm:
        resolution = matcher.resolve_block(block)

    mock_ensure_lm.assert_not_called()
    assert len(resolution.resolved_entities) == 1
    assert resolution.resolved_entities[0].id == 100
    assert resolution.resolved_entities[0].match_skip_reason == "singleton_block"
    assert resolution.was_resolved is False


def test_error_recovery_keeps_original_identifiers() -> None:
    """The LLM raises. All input records come back unchanged with
    match_skip_reason == 'error_recovery', was_resolved == False, and
    ORIGINAL identifiers (no new UUID assigned to a block that changed
    nothing)."""
    matcher = EntityMatcher()
    block = EntityBlock(
        block_key="b1",
        block_size=2,
        entities=[
            Entity(id=100, name="A", uuid="uuid-original-100"),
            Entity(id=200, name="B", uuid="uuid-original-200"),
        ],
    )

    with patch.object(EntityMatcher, "_ensure_lm", side_effect=RuntimeError("LLM down")):
        resolution = matcher.resolve_block(block)

    assert resolution.was_resolved is False
    assert len(resolution.resolved_entities) == 2
    for e in resolution.resolved_entities:
        assert e.match_skip_reason == "error_recovery"
    restored_ids = {e.id for e in resolution.resolved_entities}
    assert restored_ids == {100, 200}
    restored_uuids = {e.uuid for e in resolution.resolved_entities}
    assert restored_uuids == {"uuid-original-100", "uuid-original-200"}


def test_skip_history_accumulates_across_iterations() -> None:
    """A record skipped in rounds 1, 2 and 3 ends with
    match_skip_history == [1, 2, 3]."""
    block = EntityBlock(
        block_key="b1",
        block_size=1,
        entities=[Entity(id=100, name="A", match_skip_history=[1, 2])],
    )
    matcher = EntityMatcher()
    resolution = matcher.resolve_block(block, iteration=3)
    assert resolution.resolved_entities[0].match_skip_history == [1, 2, 3]


def test_validation_gate_fails_on_dropped_records() -> None:
    """A synthetic run that drops 1% of records fails the identifier
    coverage gate; a clean run passes all checks."""
    from serf.eval.evaluator import evaluate_er_results

    input_ids = set(range(100))
    clean_resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[Entity(id=i, name=f"E{i}", uuid=f"uuid-{i}") for i in range(100)],
        was_resolved=False,
        original_count=100,
        resolved_count=100,
    )
    clean_report = evaluate_er_results(
        [clean_resolution], original_entity_count=100, input_entity_ids=input_ids
    )
    assert clean_report["overall_status"] == "PASS"
    assert clean_report["identifier_coverage"]["missing_count"] == 0

    # Drop record 0 entirely: absent from output, no match_skip_reason.
    lossy_resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[Entity(id=i, name=f"E{i}", uuid=f"uuid-{i}") for i in range(1, 100)],
        was_resolved=False,
        original_count=100,
        resolved_count=99,
    )
    lossy_report = evaluate_er_results(
        [lossy_resolution], original_entity_count=100, input_entity_ids=input_ids
    )
    assert lossy_report["overall_status"] == "FAIL"
    assert lossy_report["identifier_coverage"]["missing_count"] == 1
    assert 0 in lossy_report["identifier_coverage"]["missing_ids"]
