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


def _entities(count: int) -> list[Entity]:
    """Build ``count`` minimal entities."""
    return [Entity(id=i, name=f"e{i}", description="", entity_type="entity") for i in range(count)]


def _fake_blocker(block_size: int = 4) -> MagicMock:
    """Blocking pipeline stub that packs each split into fixed-size blocks."""

    def run(records: list[Entity]) -> tuple[list[EntityBlock], MagicMock]:
        chunks = [records[i : i + block_size] for i in range(0, len(records), block_size)]
        blocks = [
            EntityBlock(block_key=f"b{i}", block_size=len(chunk), entities=chunk)
            for i, chunk in enumerate(chunks)
        ]
        return blocks, MagicMock()

    blocker = MagicMock()
    blocker.run.side_effect = run
    return blocker


@patch("serf.dspy.optimize.SemanticBlockingPipeline")
def test_prepare_dataset_splits_blocks_each_split_separately(mock_blocker_cls: MagicMock) -> None:
    """Records are sampled first, then blocked within each disjoint split."""
    blocker = _fake_blocker()
    mock_blocker_cls.return_value = blocker
    train, val, holdout = prepare_dataset_splits(
        _entities(40),
        ground_truth={(0, 1), (3, 4)},
        sizes=SplitSizes(train_records=16, val_records=8, holdout_records=8),
        seed=3,
    )
    assert len(train) == 4
    assert len(val) == 2
    assert len(holdout) == 2
    assert all(example.resolution is not None for example in train)
    assert all(example.resolution is not None for example in val)

    blocked = [call.args[0] for call in blocker.run.call_args_list]
    assert len(blocked) == 3
    id_sets = [{entity.id for entity in records} for records in blocked]
    assert id_sets[0].isdisjoint(id_sets[1])
    assert id_sets[0].isdisjoint(id_sets[2])
    assert id_sets[1].isdisjoint(id_sets[2])


@patch("serf.dspy.optimize.SemanticBlockingPipeline")
def test_prepare_dataset_splits_val_examples_carry_gold_pairs(mock_blocker_cls: MagicMock) -> None:
    """Match-group sampling keeps gold pairs inside the val blocks."""
    mock_blocker_cls.return_value = _fake_blocker()
    ground_truth = {(i, i + 20) for i in range(20)}
    _train, val, _holdout = prepare_dataset_splits(
        _entities(40),
        ground_truth=ground_truth,
        sizes=SplitSizes(train_records=16, val_records=8, holdout_records=8),
        seed=3,
    )
    assert val
    assert sum(len(example.resolution.matches) for example in val) > 0
