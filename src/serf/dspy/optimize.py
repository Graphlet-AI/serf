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
from serf.dspy.types import BlockResolution, Entity, EntityBlock, MatchDecision
from serf.eval.metrics import f1_score
from serf.eval.splits import SplitSizes, chunk_records, sample_blocked_splits
from serf.logs import get_logger
from serf.match.few_shot import get_default_few_shot_examples
from serf.match.matcher import SCHEMA_INFO

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


def gold_resolution_for_block(
    block: EntityBlock,
    ground_truth: set[tuple[int, int]],
) -> BlockResolution:
    """Build a gold BlockResolution from ground-truth pairs inside a block.

    Parameters
    ----------
    block : EntityBlock
        Block of entities
    ground_truth : set[tuple[int, int]]
        True matching pairs

    Returns
    -------
    BlockResolution
        Gold matches fully contained in the block
    """
    ids = {entity.id for entity in block.entities}
    matches = [
        MatchDecision(
            entity_a_id=left,
            entity_b_id=right,
            is_match=True,
            confidence=1.0,
            reasoning="ground truth",
        )
        for left, right in ground_truth
        if left in ids and right in ids
    ]
    return BlockResolution(
        block_key=block.block_key,
        matches=matches,
        resolved_entities=list(block.entities),
        was_resolved=bool(matches),
        original_count=block.block_size,
        resolved_count=block.block_size,
    )


def blocks_to_examples(
    blocks: list[EntityBlock],
    ground_truth: set[tuple[int, int]],
) -> list[dspy.Example]:
    """Convert entity blocks into labeled DSPy examples.

    Parameters
    ----------
    blocks : list[EntityBlock]
        Blocks to convert
    ground_truth : set[tuple[int, int]]
        True matching pairs

    Returns
    -------
    list[dspy.Example]
        BlockMatch examples with gold resolutions
    """
    few_shot = get_default_few_shot_examples()
    examples: list[dspy.Example] = []
    for block in blocks:
        records = json.dumps(
            [entity.model_dump(mode="json") for entity in block.entities], indent=2
        )
        examples.append(
            dspy.Example(
                block_records=records,
                schema_info=SCHEMA_INFO,
                few_shot_examples=few_shot,
                resolution=gold_resolution_for_block(block, ground_truth),
            ).with_inputs("block_records", "schema_info", "few_shot_examples")
        )
    return examples


def prepare_dataset_splits(
    entities: list[Entity],
    ground_truth: set[tuple[int, int]],
    blocks: list[EntityBlock],
    sizes: SplitSizes | None = None,
    seed: int | None = None,
) -> tuple[list[dspy.Example], list[dspy.Example], list[Entity]]:
    """Sample train blocks, val records, and holdout from already-blocked data.

    Parameters
    ----------
    entities : list[Entity]
        Full entity list
    ground_truth : set[tuple[int, int]]
        True matching pairs
    blocks : list[EntityBlock]
        Semantic blocks over the full entity list
    sizes : SplitSizes | None
        Split sizes. Defaults to global benchmark config.
    seed : int | None
        RNG seed

    Returns
    -------
    tuple[list[dspy.Example], list[dspy.Example], list[Entity]]
        Train examples, val examples, holdout records
    """
    if sizes is None:
        sizes = SplitSizes(
            train_blocks=int(config.get("benchmarks.train_blocks", 1000)),
            val_records=int(config.get("benchmarks.val_records", 500)),
            holdout_records=int(config.get("benchmarks.holdout_records", 1000)),
        )
    splits = sample_blocked_splits(
        entities,
        blocks,
        train_blocks=sizes.train_blocks,
        val_records=sizes.val_records,
        holdout_records=sizes.holdout_records,
        seed=seed,
    )
    target_block_size = int(config.get("er.blocking.target_block_size", 30))
    val_blocks = chunk_records(splits.val_records, target_block_size, prefix="val")
    train_examples = blocks_to_examples(splits.train_blocks, ground_truth)
    val_examples = blocks_to_examples(val_blocks, ground_truth)
    return train_examples, val_examples, splits.holdout_records


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
