"""Tests for GEPA student/teacher optimization wiring."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import dspy

from serf.dspy.optimize import er_metric, optimize_module
from serf.dspy.types import BlockResolution, MatchDecision


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
    """GEPA compiles the student module with Gemini 3.7 Flash as reflection_lm."""
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
