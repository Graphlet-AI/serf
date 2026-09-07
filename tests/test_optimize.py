"""Tests for GEPA optimization data preparation and scoring (no LLM calls)."""

# isort: off
# numpy must load before dspy: dspy lazily proxies the numpy module, and if
# something later does `from numpy.typing import X` before numpy has been
# imported for real, the lazy proxy re-execs numpy's __init__ into an
# already-partially-loaded module and corrupts its C extension state.
import numpy  # noqa: F401
import dspy

# isort: on

from serf.dspy.optimize import (
    _map_gold_resolution,
    build_gepa_examples,
    er_metric,
    evaluate_program,
    gold_resolution_for_block,
    resolution_to_pairs,
)
from serf.dspy.types import BlockResolution, Entity, EntityBlock
from serf.match.uuid_mapper import UUIDMapper


def _block(ids: list[int]) -> EntityBlock:
    return EntityBlock(
        block_key="b1",
        block_size=len(ids),
        entities=[Entity(id=i, name=f"E{i}") for i in ids],
    )


def test_gold_resolution_merges_true_pairs_transitively() -> None:
    """Entities linked by a chain of true pairs form one component, mastered
    by the lowest id, exactly like an LLM merge would be expected to."""
    block = _block([10, 20, 30, 40])
    ground_truth = {(10, 20), (20, 30)}  # 10-20-30 chain; 40 is a distractor
    gold = gold_resolution_for_block(block, ground_truth)

    merged = next(e for e in gold.resolved_entities if e.id == 10)
    assert set(merged.source_ids or []) == {20, 30}
    standalone = next(e for e in gold.resolved_entities if e.id == 40)
    assert not standalone.source_ids
    assert gold.was_resolved is True


def test_gold_resolution_no_true_pairs_leaves_everyone_standalone() -> None:
    """A block with zero true pairs in it resolves to no merges at all."""
    block = _block([10, 20])
    gold = gold_resolution_for_block(block, ground_truth=set())
    assert gold.was_resolved is False
    assert all(not e.source_ids for e in gold.resolved_entities)


def test_resolution_to_pairs_normalizes_order() -> None:
    """Pairs are always (min, max), matching evaluate_resolution's convention."""
    resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[Entity(id=5, name="A", source_ids=[2])],
        original_count=2,
        resolved_count=1,
    )
    assert resolution_to_pairs(resolution) == {(2, 5)}


def test_map_gold_resolution_translates_to_mapped_ids() -> None:
    """The gold label's ids move into the same mapped-id space the block
    itself was translated into, so the metric compares like with like."""
    block = _block([100, 200, 300])
    mapper = UUIDMapper()
    mapper.map_block(block)
    gold = BlockResolution(
        block_key="b1",
        resolved_entities=[Entity(id=100, name="A", source_ids=[200])],
        original_count=3,
        resolved_count=1,
    )
    mapped = _map_gold_resolution(gold, mapper)
    assert mapped.resolved_entities[0].id == mapper._id_to_int[100]
    assert mapped.resolved_entities[0].source_ids == [mapper._id_to_int[200]]


def test_build_gepa_examples_produces_valid_dspy_examples() -> None:
    """Each example carries the three BlockMatch inputs and a resolution
    label whose ids are all valid mapped ids for that same block."""
    blocks = [_block([10, 20, 30])]
    ground_truth = {(10, 20)}
    examples = build_gepa_examples(blocks, ground_truth)

    assert len(examples) == 1
    ex = examples[0]
    assert set(ex.inputs().keys()) == {"block_records", "schema_info", "few_shot_examples"}
    assert isinstance(ex.resolution, BlockResolution)
    mapped_ids = {1, 2, 3}  # 1-based mapping, per docs/ID_INVARIANTS.md D6
    for e in ex.resolution.resolved_entities:
        assert e.id in mapped_ids
        for sid in e.source_ids or []:
            assert sid in mapped_ids


def test_er_metric_perfect_prediction_scores_one() -> None:
    """An exact match to gold scores 1.0 with no missed/extra feedback."""
    gold_resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[Entity(id=1, name="A", source_ids=[2])],
        original_count=2,
        resolved_count=1,
    )
    gold = dspy.Example(resolution=gold_resolution)
    pred = dspy.Prediction(resolution=gold_resolution.model_copy())
    result = er_metric(gold, pred)
    assert result.score == 1.0
    assert "Missed" not in result.feedback
    assert "Incorrectly merged" not in result.feedback


def test_er_metric_missed_match_scores_less_than_one() -> None:
    """A prediction that fails to merge a true pair is penalized and the
    feedback names the missed pair, for the reflection_lm to read."""
    gold_resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[Entity(id=1, name="A", source_ids=[2])],
        original_count=2,
        resolved_count=1,
    )
    pred_resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[Entity(id=1, name="A"), Entity(id=2, name="B")],
        original_count=2,
        resolved_count=2,
    )
    gold = dspy.Example(resolution=gold_resolution)
    pred = dspy.Prediction(resolution=pred_resolution)
    result = er_metric(gold, pred)
    assert result.score < 1.0
    assert "Missed 1 true match" in result.feedback


def test_er_metric_malformed_prediction_scores_zero() -> None:
    """A prediction missing the resolution field fails gracefully to 0.0
    rather than raising out of the optimizer's evaluation loop."""
    gold = dspy.Example(
        resolution=BlockResolution(block_key="b1", original_count=1, resolved_count=1)
    )
    pred = dspy.Prediction()  # no `resolution` attribute at all
    result = er_metric(gold, pred)
    assert result.score == 0.0


def test_evaluate_program_averages_scores_across_examples() -> None:
    """evaluate_program calls the program per-example and averages er_metric."""
    gold_resolution = BlockResolution(
        block_key="b1",
        resolved_entities=[Entity(id=1, name="A", source_ids=[2])],
        original_count=2,
        resolved_count=1,
    )
    example = dspy.Example(
        block_records="[]",
        schema_info="",
        few_shot_examples="",
        resolution=gold_resolution,
    ).with_inputs("block_records", "schema_info", "few_shot_examples")

    def fake_program(**kwargs: object) -> dspy.Prediction:
        return dspy.Prediction(resolution=gold_resolution.model_copy())

    avg_f1 = evaluate_program(fake_program, [example])
    assert avg_f1 == 1.0


def test_evaluate_program_handles_exceptions_as_zero() -> None:
    """A program call that raises counts as a 0.0, not a crashed evaluation."""

    def failing_program(**kwargs: object) -> dspy.Prediction:
        raise RuntimeError("boom")

    example = dspy.Example(
        block_records="[]",
        schema_info="",
        few_shot_examples="",
        resolution=BlockResolution(block_key="b1", original_count=1, resolved_count=1),
    ).with_inputs("block_records", "schema_info", "few_shot_examples")
    avg_f1 = evaluate_program(failing_program, [example])
    assert avg_f1 == 0.0
