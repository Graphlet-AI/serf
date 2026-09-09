"""Tests for GEPA student/teacher optimization wiring."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import dspy

from serf.dspy.optimize import (
    er_metric,
    gold_resolution_for_block,
    optimize_module,
    prepare_dataset_splits,
)
from serf.dspy.types import BlockResolution, Entity, EntityBlock, MatchDecision
from serf.eval.splits import SplitSizes


def _resolution(matches: list[tuple[int, int]]) -> BlockResolution:
    """Build a BlockResolution with the given match pairs."""
    decisions = [
        MatchDecision(
            entity_a_id=a,
            entity_b_id=b,
            is_match=True,
            confidence=1.0,
            reasoning="match",
        )
        for a, b in matches
    ]
    return BlockResolution(block_key="b1", matches=decisions, was_resolved=True)


def test_er_metric_perfect_match() -> None:
    """Metric returns score 1.0 when predicted pairs match gold."""
    gold = dspy.Example(resolution=_resolution([(1, 2)])).with_inputs()
    pred = dspy.Prediction(resolution=_resolution([(1, 2)]))
    result = er_metric(gold, pred)
    assert result.score == 1.0
    assert "perfect" in result.feedback.lower() or "no errors" in result.feedback.lower()


def test_er_metric_missed_and_extra_pairs() -> None:
    """Metric returns partial F1 and names missed/extra pairs in feedback."""
    gold = dspy.Example(resolution=_resolution([(1, 2), (3, 4)])).with_inputs()
    pred = dspy.Prediction(resolution=_resolution([(1, 2), (5, 6)]))
    result = er_metric(gold, pred)
    assert 0.0 < result.score < 1.0
    assert "(3, 4)" in result.feedback or "(3,4)" in result.feedback.replace(" ", "")
    assert "(5, 6)" in result.feedback or "(5,6)" in result.feedback.replace(" ", "")


@patch("serf.dspy.optimize.dspy.GEPA")
@patch("serf.dspy.optimize.create_lm")
def test_optimize_module_uses_student_lm_and_teacher_reflection(
    mock_create_lm: MagicMock, mock_gepa_cls: MagicMock
) -> None:
    """GEPA compiles the student module with Gemini 3.5 Flash-Lite as reflection_lm."""
    student_lm = MagicMock(name="student_lm")
    teacher_lm = MagicMock(name="teacher_lm")

    def _create(model: str | None = None, *, role: str = "student", **kwargs: Any) -> MagicMock:
        if role == "teacher":
            return teacher_lm
        return student_lm

    mock_create_lm.side_effect = _create
    optimizer = MagicMock()
    optimizer.compile.return_value = MagicMock(name="optimized")
    mock_gepa_cls.return_value = optimizer

    module = cast(dspy.Module, dspy.Predict("block_records -> resolution"))
    trainset = [
        dspy.Example(block_records="[]", resolution=_resolution([])).with_inputs("block_records")
    ]
    result = optimize_module(module, trainset=trainset)

    roles = [c.kwargs.get("role") for c in mock_create_lm.call_args_list]
    assert "student" in roles
    assert "teacher" in roles
    teacher_call = next(
        c for c in mock_create_lm.call_args_list if c.kwargs.get("role") == "teacher"
    )
    assert teacher_call.kwargs.get("temperature") == 1.0

    gepa_kwargs = mock_gepa_cls.call_args.kwargs
    assert gepa_kwargs["reflection_lm"] is teacher_lm
    assert gepa_kwargs["auto"] == "light"
    optimizer.compile.assert_called_once()
    compile_kwargs = optimizer.compile.call_args.kwargs
    assert compile_kwargs["student"] is module
    assert compile_kwargs["trainset"] is trainset
    assert result is optimizer.compile.return_value


@patch("serf.dspy.optimize.dspy.GEPA")
@patch("serf.dspy.optimize.create_lm")
def test_optimize_module_forwards_log_dir(
    mock_create_lm: MagicMock, mock_gepa_cls: MagicMock
) -> None:
    """GEPA state goes to the caller's log_dir so runs do not resume each other."""
    mock_create_lm.return_value = MagicMock()
    optimizer = MagicMock()
    mock_gepa_cls.return_value = optimizer

    module = cast(dspy.Module, dspy.Predict("block_records -> resolution"))
    trainset = [
        dspy.Example(block_records="[]", resolution=_resolution([])).with_inputs("block_records")
    ]
    optimize_module(module, trainset=trainset, log_dir="data/gepa_logs/dblp-acm-v2")

    assert mock_gepa_cls.call_args.kwargs["log_dir"] == "data/gepa_logs/dblp-acm-v2"


def test_gold_resolution_keeps_pairs_inside_the_block() -> None:
    """Gold labels only include ground-truth pairs fully inside the block."""
    entities = [Entity(id=i, name=f"e{i}", description="", entity_type="entity") for i in range(4)]
    block = EntityBlock(block_key="b", block_size=3, entities=entities[:3])
    gold = gold_resolution_for_block(block, {(0, 1), (0, 9), (2, 3)})
    pairs = {(m.entity_a_id, m.entity_b_id) for m in gold.matches}
    assert pairs == {(0, 1)}


def test_prepare_dataset_splits_builds_disjoint_examples() -> None:
    """prepare_dataset_splits turns blocked data into train/val examples plus holdout blocks."""
    entities = [Entity(id=i, name=f"e{i}", description="", entity_type="entity") for i in range(30)]
    blocks = [
        EntityBlock(
            block_key=str(i),
            block_size=3,
            entities=entities[i * 3 : i * 3 + 3],
        )
        for i in range(10)
    ]
    train, val, holdout = prepare_dataset_splits(
        ground_truth={(0, 1), (3, 4)},
        blocks=blocks,
        sizes=SplitSizes(train_blocks=2, val_records=4, holdout_records=5),
        seed=3,
    )
    assert len(train) == 2
    assert len(val) == 2
    assert len(holdout) == 2
    assert all(example.resolution is not None for example in train)
    assert all(example.resolution is not None for example in val)


def test_prepare_dataset_splits_val_examples_carry_gold_pairs() -> None:
    """Val examples are real semantic blocks, so they can carry gold match pairs."""
    entities = [Entity(id=i, name=f"e{i}", description="", entity_type="entity") for i in range(40)]
    blocks = [
        EntityBlock(
            block_key=str(i),
            block_size=4,
            entities=entities[i * 4 : i * 4 + 4],
        )
        for i in range(10)
    ]
    ground_truth = {(i * 4, i * 4 + 1) for i in range(10)}
    _train, val, _holdout = prepare_dataset_splits(
        ground_truth=ground_truth,
        blocks=blocks,
        sizes=SplitSizes(train_blocks=5, val_records=8, holdout_records=8),
        seed=3,
    )
    assert val
    assert all(example.resolution.matches for example in val)
