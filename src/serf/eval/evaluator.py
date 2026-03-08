"""Post-match evaluation with Abzu-level rigor.

Performs comprehensive validation of entity resolution results:
- Explodes resolved entities from block resolutions
- Deduplicates exact copies
- Splits LLM-processed vs skipped entities
- Validates source_uuids against all historical UUIDs
- Computes detailed metrics with PASS/FAIL checks
- Tracks match_skip_reason distribution
"""

import json
from typing import Any

from serf.config import config
from serf.dspy.types import BlockResolution, Entity
from serf.eval.metrics import validate_source_uuids
from serf.logs import get_logger

logger = get_logger(__name__)

# Thresholds for PASS/FAIL assessment (all values are percentages: 0–100).
# Previously ERROR_THRESHOLD used a fraction scale (0–1) with a * 100 multiplier
# in the comparison. All three are now consistently percentage-based.
COVERAGE_THRESHOLD: float = config.get("er.eval.coverage_threshold", 99.99)
ERROR_THRESHOLD: float = config.get("er.eval.error_threshold", 1.0)
OVERLAP_THRESHOLD: float = config.get("er.eval.overlap_threshold", 1.0)


def evaluate_er_results(
    resolutions: list[BlockResolution],
    original_entity_count: int,
    iteration: int = 1,
    historical_uuids: set[str] | None = None,
    previous_entity_count: int | None = None,
) -> dict[str, Any]:
    """Comprehensive evaluation of entity resolution results.

    Mirrors Abzu's er_eval.py with iteration-aware validation,
    match_skip_reason analysis, and PASS/FAIL assessment.

    Parameters
    ----------
    resolutions : list[BlockResolution]
        Block resolution results from matching
    original_entity_count : int
        Number of entities in the original (iteration 0) dataset
    iteration : int
        Current iteration number
    historical_uuids : set[str] | None
        All UUIDs from all previous iterations (for validation)
    previous_entity_count : int | None
        Entity count from previous iteration (for per-round reduction)

    Returns
    -------
    dict[str, Any]
        Comprehensive evaluation metrics
    """
    # Step 1: Explode resolved entities from all blocks
    all_resolved: list[Entity] = []
    for r in resolutions:
        all_resolved.extend(r.resolved_entities)

    # Step 2: Deduplicate exact copies (by id + uuid)
    seen_keys: set[tuple[int, str | None]] = set()
    unique_entities: list[Entity] = []
    duplicates = 0
    for e in all_resolved:
        key = (e.id, e.uuid)
        if key in seen_keys:
            duplicates += 1
            continue
        seen_keys.add(key)
        unique_entities.append(e)

    # Step 3: Split into LLM-processed vs skipped
    llm_processed: list[Entity] = []
    skipped: list[Entity] = []
    for e in unique_entities:
        if e.match_skip:
            skipped.append(e)
        else:
            llm_processed.append(e)

    # Step 4: Analyze match_skip_reasons
    skip_reasons: dict[str, int] = {}
    for e in skipped:
        reason = e.match_skip_reason or "unknown"
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    singleton_count = skip_reasons.get("singleton_block", 0)
    error_recovery_count = skip_reasons.get("error_recovery", 0)
    missing_count = skip_reasons.get("missing_in_match_output", 0)

    # Step 5: Count source tracking
    total_source_ids = sum(len(e.source_ids or []) for e in unique_entities)
    total_source_uuids = sum(len(e.source_uuids or []) for e in unique_entities)
    entities_with_merges = sum(1 for e in unique_entities if e.source_ids)

    # Step 6: Validate source_uuids if historical data provided
    uuid_validation: dict[str, Any] = {"skipped": True}
    if historical_uuids is not None:
        uuid_validation = validate_source_uuids(unique_entities, historical_uuids)

    # Step 7: Compute reduction metrics
    iteration_input = previous_entity_count or original_entity_count
    reduction_from_matching = iteration_input - len(unique_entities)
    reduction_pct = reduction_from_matching / iteration_input * 100 if iteration_input > 0 else 0.0
    overall_reduction_pct = (
        (original_entity_count - len(unique_entities)) / original_entity_count * 100
        if original_entity_count > 0
        else 0.0
    )

    # Step 8: PASS/FAIL assessment
    error_rate = error_recovery_count / len(unique_entities) * 100 if unique_entities else 0.0
    duplicate_rate = duplicates / len(all_resolved) * 100 if all_resolved else 0.0
    uuid_coverage = uuid_validation.get("coverage_pct", 100.0)

    checks: dict[str, dict[str, Any]] = {
        "uuid_coverage": {
            "value": uuid_coverage,
            "threshold": COVERAGE_THRESHOLD,
            "passed": uuid_coverage >= COVERAGE_THRESHOLD or uuid_validation.get("skipped", False),
            "description": f"source_uuid coverage >= {COVERAGE_THRESHOLD}%",
        },
        "error_rate": {
            "value": error_rate,
            "threshold": ERROR_THRESHOLD,
            "passed": error_rate < ERROR_THRESHOLD,
            "description": f"error_recovery rate < {ERROR_THRESHOLD}%",
        },
        "duplicate_rate": {
            "value": duplicate_rate,
            "threshold": OVERLAP_THRESHOLD,
            "passed": duplicate_rate < OVERLAP_THRESHOLD,
            "description": f"duplicate entity rate < {OVERLAP_THRESHOLD}%",
        },
    }
    overall_passed = all(c["passed"] for c in checks.values())

    metrics: dict[str, Any] = {
        "iteration": iteration,
        "original_entity_count": original_entity_count,
        "iteration_input_entities": iteration_input,
        "total_resolved_raw": len(all_resolved),
        "duplicates_removed": duplicates,
        "unique_entities": len(unique_entities),
        "llm_processed": len(llm_processed),
        "skipped_entities": len(skipped),
        "entities_with_merges": entities_with_merges,
        "total_source_ids": total_source_ids,
        "total_source_uuids": total_source_uuids,
        "reduction_from_matching": reduction_from_matching,
        "reduction_pct": round(reduction_pct, 2),
        "overall_reduction_pct": round(overall_reduction_pct, 2),
        "skip_reasons": {
            "singleton_block": singleton_count,
            "error_recovery": error_recovery_count,
            "missing_in_match_output": missing_count,
            "other": sum(
                v
                for k, v in skip_reasons.items()
                if k not in ("singleton_block", "error_recovery", "missing_in_match_output")
            ),
        },
        "uuid_validation": uuid_validation,
        "checks": checks,
        "overall_status": "PASS" if overall_passed else "FAIL",
    }

    logger.info(f"Evaluation (iteration {iteration}): {metrics['overall_status']}")
    logger.info(
        f"  {iteration_input} → {len(unique_entities)} entities "
        f"({reduction_pct:.1f}% reduction, {overall_reduction_pct:.1f}% overall)"
    )
    logger.info(
        f"  LLM processed: {len(llm_processed)}, skipped: {len(skipped)}, "
        f"merged: {entities_with_merges}"
    )
    if skip_reasons:
        logger.info(f"  Skip reasons: {skip_reasons}")
    if not uuid_validation.get("skipped"):
        logger.info(f"  UUID validation: {uuid_validation.get('coverage_pct', 0):.2f}% coverage")

    return metrics


def format_evaluation_report(metrics: dict[str, Any]) -> str:
    """Format evaluation metrics as a human-readable report.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics from evaluate_er_results

    Returns
    -------
    str
        Formatted multi-line report
    """
    lines = []
    status = metrics["overall_status"]
    lines.append(f"\n{'=' * 60}")
    lines.append(f"  ER Evaluation Report — Iteration {metrics['iteration']}  [{status}]")
    lines.append(f"{'=' * 60}")

    lines.append("\n  Entity Counts:")
    lines.append(f"    Original (iteration 0):    {metrics['original_entity_count']}")
    lines.append(f"    Input (this iteration):    {metrics['iteration_input_entities']}")
    lines.append(f"    Output (unique resolved):  {metrics['unique_entities']}")
    lines.append(f"    Duplicates removed:        {metrics['duplicates_removed']}")

    lines.append("\n  Processing Breakdown:")
    lines.append(f"    LLM processed:             {metrics['llm_processed']}")
    lines.append(f"    Skipped:                   {metrics['skipped_entities']}")
    lines.append(f"    Entities with merges:       {metrics['entities_with_merges']}")

    lines.append("\n  Reduction:")
    lines.append(f"    This iteration:            {metrics['reduction_pct']:.1f}%")
    lines.append(f"    Overall (from original):   {metrics['overall_reduction_pct']:.1f}%")

    lines.append("\n  Source Tracking:")
    lines.append(f"    Total source_ids:          {metrics['total_source_ids']}")
    lines.append(f"    Total source_uuids:        {metrics['total_source_uuids']}")

    skip = metrics["skip_reasons"]
    if any(v > 0 for v in skip.values()):
        lines.append("\n  Skip Reasons:")
        if skip["singleton_block"]:
            lines.append(f"    Singleton block:           {skip['singleton_block']}")
        if skip["error_recovery"]:
            lines.append(f"    Error recovery:            {skip['error_recovery']}")
        if skip["missing_in_match_output"]:
            lines.append(f"    Missing in LLM output:     {skip['missing_in_match_output']}")
        if skip["other"]:
            lines.append(f"    Other:                     {skip['other']}")

    uv = metrics["uuid_validation"]
    if not uv.get("skipped"):
        lines.append("\n  UUID Validation:")
        lines.append(f"    Total source_uuids:        {uv['total_source_uuids']}")
        lines.append(f"    Valid:                     {uv['valid_source_uuids']}")
        lines.append(f"    Invalid:                   {uv['invalid_source_uuids']}")
        lines.append(f"    Coverage:                  {uv['coverage_pct']:.2f}%")

    lines.append("\n  Checks:")
    for _name, check in metrics["checks"].items():
        mark = "✓" if check["passed"] else "✗"
        lines.append(f"    {mark} {check['description']}: {check['value']:.2f}")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def save_evaluation(metrics: dict[str, Any], output_path: str) -> None:
    """Save evaluation metrics to a JSON file.

    Parameters
    ----------
    metrics : dict[str, Any]
        Evaluation metrics
    output_path : str
        Path to write JSON file
    """
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Evaluation metrics saved to {output_path}")
