"""Tests for GEPA training of the per-dataset matching prompts."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import dspy
import pytest

from serf.dspy.dataset_signatures import get_dataset_spec
from serf.dspy.train import (
    GOLD_PAIRS_FIELD,
    MAX_ENUMERATED_ERRORS,
    blocks_to_dataset_examples,
    candidate_pairs,
    gold_pairs_in_block,
    make_dataset_metric,
    train_dataset,
)
from serf.dspy.trained import (
    load_trained_predictor,
    predictor_instructions,
    trained_instructions,
    trained_program_path,
)
from serf.dspy.types import Entity, EntityBlock

SPEC = get_dataset_spec("dblp-acm")


def _entity(record_id: int, side: str, title: str, year: int = 1996) -> Entity:
    """Build a benchmark-shaped entity on one side of the join.

    Parameters
    ----------
    record_id : int
        Entity id
    side : str
        ``l`` or ``r``, the prefix the benchmark loader adds
    title : str
        Publication title
    year : int
        Publication year

    Returns
    -------
    Entity
        Entity whose attributes carry the prefixed source columns
    """
    return Entity(
        id=record_id,
        name=title,
        attributes={
            f"{side}_id": f"{side}{record_id}",
            f"{side}_title": title,
            f"{side}_authors": "A. Author",
            f"{side}_venue": "VLDB",
            f"{side}_year": year,
        },
    )


def _block(entities: list[Entity], key: str = "block-0") -> EntityBlock:
    """Wrap entities in a block.

    Parameters
    ----------
    entities : list[Entity]
        Entities in the block
    key : str
        Block key

    Returns
    -------
    EntityBlock
        Block holding those entities
    """
    return EntityBlock(block_key=key, entities=entities, block_size=len(entities))


def _prediction(pairs: list[tuple[int, int]], is_match: bool = True) -> dspy.Prediction:
    """Build a prediction carrying typed candidates for the given pairs.

    Parameters
    ----------
    pairs : list[tuple[int, int]]
        Record id pairs to emit
    is_match : bool
        Whether each candidate is marked as a match

    Returns
    -------
    dspy.Prediction
        Prediction shaped like the per-dataset signature's output
    """
    candidates = [
        SPEC.candidate_type(
            left=SPEC.left_type(record_id=left),
            right=SPEC.right_type(record_id=right),
            is_match=is_match,
        )
        for left, right in pairs
    ]
    return dspy.Prediction(candidates=candidates)


def test_gold_pairs_in_block_keeps_only_pairs_wholly_inside_the_block() -> None:
    """A pair whose other half blocking put elsewhere is not learnable here."""
    block = _block([_entity(1, "l", "A"), _entity(2, "r", "A")])
    pairs = gold_pairs_in_block(block, {(1, 2), (1, 9), (7, 8)})
    assert pairs == [[1, 2]]


def test_examples_are_built_the_way_the_matcher_builds_a_call() -> None:
    """Training inputs must be the typed sides, or the prompt shape would differ."""
    block = _block([_entity(1, "l", "A"), _entity(2, "r", "A")])
    examples = blocks_to_dataset_examples([block], {(1, 2)}, SPEC)
    assert len(examples) == 1
    example = examples[0]
    assert set(example.inputs().keys()) == {SPEC.left_field, SPEC.right_field}
    assert [type(r).__name__ for r in getattr(example, SPEC.left_field)] == [
        SPEC.left_type.__name__
    ]
    assert [type(r).__name__ for r in getattr(example, SPEC.right_field)] == [
        SPEC.right_type.__name__
    ]
    assert getattr(example, GOLD_PAIRS_FIELD) == [[1, 2]]


def test_single_source_blocks_are_not_trainable() -> None:
    """A block the matcher would skip cannot teach it anything."""
    block = _block([_entity(1, "l", "A"), _entity(2, "l", "A")])
    assert blocks_to_dataset_examples([block], {(1, 2)}, SPEC) == []


def test_blocks_without_a_gold_pair_are_not_trainable() -> None:
    """The best answer is the empty list, which every candidate prompt returns."""
    block = _block([_entity(1, "l", "A"), _entity(2, "r", "B")])
    assert blocks_to_dataset_examples([block], set(), SPEC) == []


def test_candidate_pairs_ignores_rejections_and_normalizes_order() -> None:
    """Only the pairs marked as matches count, smaller id first."""
    assert candidate_pairs(_prediction([(4, 2)])) == {(2, 4)}
    assert candidate_pairs(_prediction([(2, 4)], is_match=False)) == set()
    assert candidate_pairs(dspy.Prediction()) == set()


def test_metric_scores_a_perfect_block_and_says_so() -> None:
    """A correct block gets full marks and feedback that names no error."""
    block = _block([_entity(1, "l", "A"), _entity(2, "r", "A")])
    example = blocks_to_dataset_examples([block], {(1, 2)}, SPEC)[0]
    result = make_dataset_metric(SPEC)(example, _prediction([(1, 2)]))
    assert result["score"] == 1.0
    assert "Correct" in result["feedback"]
    assert "Missed" not in result["feedback"]


def test_metric_enumerates_the_records_behind_every_error() -> None:
    """GEPA can only write a rule about an error the reflection LM can see."""
    block = _block([_entity(1, "l", "Mediator Languages"), _entity(2, "r", "Mediator languages")])
    example = blocks_to_dataset_examples([block], {(1, 2)}, SPEC)[0]
    result = make_dataset_metric(SPEC)(example, _prediction([]))
    feedback = result["feedback"]
    assert result["score"] == 0.0
    assert "Missed 1 true pair(s)" in feedback
    assert "(1, 2)" in feedback
    assert "Mediator Languages" in feedback
    assert "Mediator languages" in feedback


def test_metric_separates_missed_pairs_from_invented_ones() -> None:
    """Recall errors and precision errors need different instructions to fix."""
    block = _block(
        [
            _entity(1, "l", "Left one"),
            _entity(2, "r", "Right one"),
            _entity(3, "r", "Right two"),
        ]
    )
    example = blocks_to_dataset_examples([block], {(1, 2)}, SPEC)[0]
    result = make_dataset_metric(SPEC)(example, _prediction([(1, 3)]))
    feedback = result["feedback"]
    assert result["score"] == 0.0
    assert "ARE the same entity" in feedback
    assert "are NOT matches" in feedback
    assert "(1, 2)" in feedback
    assert "(1, 3)" in feedback


def test_metric_bounds_how_many_errors_it_enumerates() -> None:
    """An unbounded error list would crowd the instructions out of the context."""
    left = [_entity(index, "l", f"Left {index}") for index in range(1, 12)]
    right = [_entity(index + 100, "r", f"Right {index}") for index in range(1, 12)]
    gold = {(index, index + 100) for index in range(1, 12)}
    example = blocks_to_dataset_examples([_block(left + right)], gold, SPEC)[0]
    result = make_dataset_metric(SPEC)(example, _prediction([]))
    feedback = result["feedback"]
    assert feedback.count("Left ") == MAX_ENUMERATED_ERRORS
    assert f"...and {len(gold) - MAX_ENUMERATED_ERRORS} more" in feedback


def test_metric_rewards_an_empty_answer_on_a_block_with_no_true_pair() -> None:
    """F1 is undefined on two empty sets; scoring it zero punishes the right answer."""
    example = dspy.Example(
        **{SPEC.left_field: [], SPEC.right_field: [], GOLD_PAIRS_FIELD: []}
    ).with_inputs(SPEC.left_field, SPEC.right_field)
    result = make_dataset_metric(SPEC)(example, _prediction([]))
    assert result["score"] == 1.0
    assert "no matching pair" in result["feedback"]


def test_metric_binds_the_gepa_five_argument_protocol() -> None:
    """GEPA inspects the metric signature and passes these names by keyword."""
    import inspect

    parameters = list(inspect.signature(make_dataset_metric(SPEC)).parameters)
    assert parameters == ["gold", "pred", "trace", "pred_name", "pred_trace"]


def test_trained_program_path_is_keyed_by_dataset(tmp_path: Path) -> None:
    """One program per dataset, named so the matcher can find it without a flag."""
    path = trained_program_path("abt-buy", str(tmp_path))
    assert path == tmp_path / "abt-buy_gepa.json"


def test_loading_a_missing_program_returns_none(tmp_path: Path) -> None:
    """A dataset that was never trained matches with its signature as written."""
    assert load_trained_predictor("abt-buy", str(tmp_path)) is None
    assert trained_instructions("abt-buy", str(tmp_path)) is None


def test_a_trained_program_round_trips_its_instructions(tmp_path: Path) -> None:
    """What training saves is what matching loads, or the run is not reproducible."""
    predictor = dspy.Predict(SPEC.signature)
    predictor.signature = SPEC.signature.with_instructions("Trained instructions.")
    predictor.save(str(trained_program_path("dblp-acm", str(tmp_path))))

    loaded = load_trained_predictor("dblp-acm", str(tmp_path))
    assert loaded is not None
    assert predictor_instructions(loaded) == "Trained instructions."
    signature = loaded.signature
    assert signature is not None
    assert list(signature.input_fields) == [SPEC.left_field, SPEC.right_field]
    assert trained_instructions("dblp-acm", str(tmp_path)) == "Trained instructions."


def test_train_dataset_uses_the_dataset_signature_and_saves_where_matching_looks(
    tmp_path: Path,
) -> None:
    """Training has to optimize the signature that ships, not the generic one."""
    entities = [_entity(1, "l", "Alpha"), _entity(2, "r", "Alpha")]
    benchmark = MagicMock()
    benchmark.to_entities.return_value = ([entities[0]], [entities[1]])
    benchmark.ground_truth = {(1, 2)}

    splits = MagicMock()
    splits.train_records = entities
    splits.val_records = entities

    optimized = dspy.Predict(SPEC.signature)
    optimized.signature = SPEC.signature.with_instructions("Rewritten by GEPA.")
    optimized.detailed_results = MagicMock(val_aggregate_scores=[0.5, 0.9], best_idx=1)

    captured: dict[str, Any] = {}

    def fake_optimize(module: Any, **kwargs: Any) -> Any:
        captured["module"] = module
        captured.update(kwargs)
        return optimized

    with (
        patch("serf.dspy.train.BenchmarkDataset.download", return_value=benchmark),
        patch("serf.dspy.train.sample_random_splits", return_value=splits),
        patch("serf.dspy.train.SemanticBlockingPipeline") as blocker,
        patch("serf.dspy.train.optimize_module", side_effect=fake_optimize),
    ):
        blocker.return_value.run.return_value = (
            [_block(entities)],
            MagicMock(),
        )
        result = train_dataset("dblp-acm", output_dir=str(tmp_path))

    assert captured["module"].signature is SPEC.signature
    assert captured["metric"].__qualname__.startswith("make_dataset_metric")
    assert result.signature_name == SPEC.signature.__name__
    assert result.train_examples == 1
    assert result.val_examples == 1
    assert result.baseline_score == 0.5
    assert result.best_score == 0.9
    assert result.improved is True
    assert result.instructions_after == "Rewritten by GEPA."
    assert Path(result.program_path) == trained_program_path("dblp-acm", str(tmp_path))
    assert Path(result.program_path).exists()


def test_train_dataset_refuses_to_train_on_nothing(tmp_path: Path) -> None:
    """A run with no usable block would report a score it never measured."""
    entities = [_entity(1, "l", "Alpha"), _entity(2, "l", "Alpha")]
    benchmark = MagicMock()
    benchmark.to_entities.return_value = (entities, [])
    benchmark.ground_truth = {(1, 2)}
    splits = MagicMock(train_records=entities, val_records=entities)

    with (
        patch("serf.dspy.train.BenchmarkDataset.download", return_value=benchmark),
        patch("serf.dspy.train.sample_random_splits", return_value=splits),
        patch("serf.dspy.train.SemanticBlockingPipeline") as blocker,
    ):
        blocker.return_value.run.return_value = ([_block(entities)], MagicMock())
        with pytest.raises(ValueError, match="No trainable blocks"):
            train_dataset("dblp-acm", output_dir=str(tmp_path))


def test_train_result_reports_no_improvement_when_gepa_did_not_find_one() -> None:
    """A trained prompt that lost on validation must not look like a win."""
    from serf.dspy.train import TrainResult

    result = TrainResult(
        dataset="abt-buy",
        signature_name="AbtBuyBlockMatch",
        program_path="x.json",
        train_examples=10,
        val_examples=5,
        baseline_score=0.9,
        best_score=0.9,
        instructions_before="a",
        instructions_after="b",
    )
    assert result.improved is False
