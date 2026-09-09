"""GEPA optimization of ER signatures with student and teacher LMs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import dspy
from dspy.teleprompt.gepa.gepa import GEPAFeedbackMetric
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from serf.config import config
from serf.dspy.lm import create_lm
from serf.dspy.signatures import BlockMatch, EdgeResolve, EntityMerge
from serf.dspy.types import BlockResolution
from serf.eval.metrics import f1_score
from serf.logs import get_logger

logger = get_logger(__name__)

SIGNATURES: dict[str, type[dspy.Signature]] = {
    "block-match": BlockMatch,
    "entity-merge": EntityMerge,
    "edge-resolve": EdgeResolve,
}

INPUT_FIELDS: dict[str, list[str]] = {
    "block-match": ["block_records", "schema_info", "few_shot_examples"],
    "entity-merge": ["entity_a", "entity_b"],
    "edge-resolve": ["edge_block"],
}


def _match_pairs(resolution: BlockResolution | dict[str, Any] | None) -> set[tuple[int, int]]:
    """Extract normalized match pairs from a block resolution.

    Parameters
    ----------
    resolution : BlockResolution | dict[str, Any] | None
        Predicted or gold block resolution

    Returns
    -------
    set[tuple[int, int]]
        Match pairs with the smaller ID first
    """
    if resolution is None:
        return set()
    if isinstance(resolution, dict):
        resolution = BlockResolution.model_validate(resolution)

    pairs: set[tuple[int, int]] = set()
    for match in resolution.matches:
        if match.is_match:
            left = min(match.entity_a_id, match.entity_b_id)
            right = max(match.entity_a_id, match.entity_b_id)
            pairs.add((left, right))
    for entity in resolution.resolved_entities:
        if entity.source_ids:
            for source_id in entity.source_ids:
                pairs.add((min(entity.id, source_id), max(entity.id, source_id)))
    return pairs


def er_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
    program_trace: Any = None,
) -> ScoreWithFeedback:
    """Score a BlockMatch prediction and explain errors for GEPA reflection.

    Parameters
    ----------
    gold : dspy.Example
        Gold example with a ``resolution`` field
    pred : dspy.Prediction
        Model prediction with a ``resolution`` field
    trace : Any
        Optional DSPy trace (unused)
    pred_name : str | None
        Optional predictor name (unused)
    pred_trace : Any
        Optional predictor trace (unused)
    program_trace : Any
        Optional full program trace (unused)

    Returns
    -------
    ScoreWithFeedback
        ``score`` in [0, 1] and natural-language ``feedback``
    """
    gold_pairs = _match_pairs(getattr(gold, "resolution", None))
    pred_pairs = _match_pairs(getattr(pred, "resolution", None))
    score = f1_score(pred_pairs, gold_pairs)

    missed = gold_pairs - pred_pairs
    extra = pred_pairs - gold_pairs
    if not missed and not extra:
        feedback = "No errors. Perfect match against ground truth."
    else:
        parts: list[str] = [f"F1={score:.3f}."]
        if missed:
            parts.append(f"Missed true pairs: {sorted(missed)}.")
        if extra:
            parts.append(f"Extra predicted pairs: {sorted(extra)}.")
        feedback = " ".join(parts)

    return ScoreWithFeedback(score=score, feedback=feedback)


def load_jsonl_examples(path: str, input_fields: list[str]) -> list[dspy.Example]:
    """Load DSPy examples from a JSONL file.

    Parameters
    ----------
    path : str
        Path to a JSONL file of labeled examples
    input_fields : list[str]
        Field names marked as inputs

    Returns
    -------
    list[dspy.Example]
        Examples with inputs bound
    """
    examples: list[dspy.Example] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if isinstance(data.get("resolution"), dict):
                data["resolution"] = BlockResolution.model_validate(data["resolution"])
            examples.append(dspy.Example(**data).with_inputs(*input_fields))
    return examples


def optimize_module(
    module: dspy.Module,
    trainset: list[dspy.Example],
    valset: list[dspy.Example] | None = None,
    *,
    student_model: str | None = None,
    teacher_model: str | None = None,
    auto: str | None = None,
    log_dir: str | None = None,
) -> dspy.Module:
    """Optimize a DSPy module with GEPA using student and teacher LMs.

    The student LM (GPT OSS 120b by default) executes the task program.
    The teacher LM (Gemini 3.7 Flash by default) is the GEPA ``reflection_lm``.

    Parameters
    ----------
    module : dspy.Module
        Unoptimized DSPy module
    trainset : list[dspy.Example]
        Labeled training examples
    valset : list[dspy.Example] | None
        Optional validation examples
    student_model : str | None
        Override for the student/task LM
    teacher_model : str | None
        Override for the teacher/reflection LM
    auto : str | None
        GEPA budget preset (``light``, ``medium``, ``heavy``)
    log_dir : str | None
        Directory for GEPA checkpoints

    Returns
    -------
    dspy.Module
        Optimized module
    """
    student_name = student_model or config.get("models.llm")
    teacher_name = teacher_model or config.get("models.teacher")
    student_lm = create_lm(student_model, role="student")
    teacher_lm = create_lm(teacher_model, role="teacher", temperature=1.0)
    auto_budget = cast(
        Literal["light", "medium", "heavy"],
        auto or config.get("optimize.auto", "light"),
    )
    log_dir = log_dir or config.get("optimize.log_dir", "data/gepa_logs")
    num_threads = int(config.get("optimize.num_threads", 4))

    logger.info(
        f"GEPA optimize: student={student_name} teacher={teacher_name} "
        f"auto={auto_budget} train={len(trainset)}"
    )
    metric: GEPAFeedbackMetric = er_metric
    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=teacher_lm,
        auto=auto_budget,
        num_threads=num_threads,
        track_stats=True,
        log_dir=log_dir,
    )
    with dspy.context(lm=student_lm, adapter=dspy.XMLAdapter()):
        return optimizer.compile(student=module, trainset=trainset, valset=valset)
