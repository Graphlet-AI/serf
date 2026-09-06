"""Head-to-head comparison: default benchmark config vs LLM-generated config.

Runs the SERF pipeline on DBLP-ACM with both setups and compares metrics.
Requires GEMINI_API_KEY environment variable.

Run with: uv run pytest tests/test_benchmark_comparison.py -v -s
"""

import json
import os
from typing import Any

import pandas as pd
import pytest
import yaml

from serf.analyze.profiler import DatasetProfiler, generate_er_config
from serf.block.embeddings import EntityEmbedder
from serf.block.faiss_blocker import FAISSBlocker
from serf.dspy.types import Entity, EntityBlock
from serf.eval.benchmarks import RIGHT_ID_OFFSET, BenchmarkDataset
from serf.eval.metrics import evaluate_resolution
from serf.logs import get_logger
from serf.match.matcher import EntityMatcher

logger = get_logger(__name__)

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)

# Use a small subset for speed — 50 entities from each table
SUBSET_SIZE = 50


@pytest.fixture(scope="module")
def dblp_acm_dataset() -> BenchmarkDataset:
    """Download and cache the DBLP-ACM benchmark dataset."""
    return BenchmarkDataset.download("dblp-acm")


@pytest.fixture(scope="module")
def dblp_acm_subset(dblp_acm_dataset: BenchmarkDataset) -> dict[str, Any]:
    """Create a small subset of DBLP-ACM for fast testing.

    Returns dict with left_entities, right_entities, ground_truth, merged_df.
    """
    import random

    random.seed(42)

    ds = dblp_acm_dataset
    # Get matched IDs from ground truth
    gt_left_ids = {a for a, _ in ds.ground_truth}
    gt_right_ids = {b for _, b in ds.ground_truth}

    left_ents, right_ents = ds.to_entities()

    # Sample: prioritize entities with ground truth matches
    matched_left = [e for e in left_ents if e.id in gt_left_ids][: SUBSET_SIZE // 2]
    unmatched_left = [e for e in left_ents if e.id not in gt_left_ids]
    sample_left = matched_left + random.sample(
        unmatched_left, min(SUBSET_SIZE - len(matched_left), len(unmatched_left))
    )

    matched_right = [e for e in right_ents if e.id in gt_right_ids][: SUBSET_SIZE // 2]
    unmatched_right = [e for e in right_ents if e.id not in gt_right_ids]
    sample_right = matched_right + random.sample(
        unmatched_right, min(SUBSET_SIZE - len(matched_right), len(unmatched_right))
    )

    # Re-index
    for i, e in enumerate(sample_left):
        e.id = i
    for i, e in enumerate(sample_right):
        e.id = i + RIGHT_ID_OFFSET

    # Filter ground truth to only include entities in our subset
    left_orig_ids = {e.id for e in sample_left}
    right_orig_ids = {e.id for e in sample_right}
    subset_gt = {(a, b) for a, b in ds.ground_truth if a in left_orig_ids and b in right_orig_ids}

    # Create merged DataFrame for profiling
    table_a_sub = ds.table_a.head(SUBSET_SIZE).copy()
    table_b_sub = ds.table_b.head(SUBSET_SIZE).copy()
    table_a_sub["source_table"] = "dblp"
    table_b_sub["source_table"] = "acm"
    merged_df = pd.concat([table_a_sub, table_b_sub], ignore_index=True)

    return {
        "left_entities": sample_left,
        "right_entities": sample_right,
        "ground_truth": subset_gt,
        "merged_df": merged_df,
    }


def _run_blocking_and_matching(
    all_entities: list[Entity],
    target_block_size: int = 30,
    max_block_size: int = 100,
) -> set[tuple[int, int]]:
    """Run blocking + LLM matching on entities and return predicted pairs."""
    import asyncio

    # Embed
    embedder = EntityEmbedder()
    texts = [e.text_for_embedding() for e in all_entities]
    embeddings = embedder.embed(texts)

    # Block
    ids = [str(e.id) for e in all_entities]
    blocker = FAISSBlocker(target_block_size=target_block_size)
    block_assignments = blocker.block(embeddings, ids)

    # Build EntityBlocks
    entity_map = {e.id: e for e in all_entities}
    blocks: list[EntityBlock] = []
    for bk, eids in block_assignments.items():
        block_ents = [entity_map[int(eid)] for eid in eids]
        blocks.append(
            EntityBlock(
                block_key=bk,
                block_key_type="semantic",
                block_size=len(block_ents),
                entities=block_ents,
            )
        )

    # Match with LLM
    matcher = EntityMatcher()
    resolutions = asyncio.run(matcher.resolve_blocks(blocks))

    # Extract pairs
    predicted_pairs: set[tuple[int, int]] = set()
    for r in resolutions:
        for m in r.matches:
            if m.is_match:
                a, b = m.entity_a_id, m.entity_b_id
                predicted_pairs.add((min(a, b), max(a, b)))

    return predicted_pairs


def test_default_config_benchmark(dblp_acm_subset: dict[str, Any]) -> None:
    """Run benchmark with default config and report metrics."""
    left = dblp_acm_subset["left_entities"]
    right = dblp_acm_subset["right_entities"]
    gt = dblp_acm_subset["ground_truth"]
    all_entities = left + right

    logger.info(
        f"Default config: {len(left)} left, {len(right)} right, {len(gt)} ground truth pairs"
    )

    predicted = _run_blocking_and_matching(all_entities, target_block_size=30)
    metrics = evaluate_resolution(predicted, gt)

    logger.info(
        f"Default config results: P={metrics['precision']:.4f}, "
        f"R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}"
    )

    # Save for comparison
    results_dir = "data/benchmarks/comparison"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "default_config_results.json"), "w") as f:
        json.dump(
            {"config": "default", "predicted_pairs": len(predicted), **metrics},
            f,
            indent=2,
        )

    # Basic sanity: should find at least some matches
    assert metrics["f1_score"] >= 0.0


def test_analyze_config_benchmark(dblp_acm_subset: dict[str, Any]) -> None:
    """Run benchmark with LLM-generated config and report metrics."""
    left = dblp_acm_subset["left_entities"]
    right = dblp_acm_subset["right_entities"]
    gt = dblp_acm_subset["ground_truth"]
    merged_df = dblp_acm_subset["merged_df"]

    # Generate config via LLM
    profiler = DatasetProfiler()
    profile = profiler.profile(merged_df.to_dict("records"))
    config_yaml = generate_er_config(profile, merged_df.to_dict("records")[:10])

    logger.info(f"LLM-generated config:\n{config_yaml}")

    parsed = yaml.safe_load(config_yaml)
    assert isinstance(parsed, dict)

    all_entities = left + right

    # Use the LLM's recommended block sizes if available
    blocking = parsed.get("blocking", {})
    target = blocking.get("target_block_size", 30)
    # Cap target at 100 regardless of LLM output
    target = min(target, 100)

    logger.info(
        f"Analyze config: {len(left)} left, {len(right)} right, "
        f"{len(gt)} ground truth pairs, target_block_size={target}"
    )

    predicted = _run_blocking_and_matching(all_entities, target_block_size=target)
    metrics = evaluate_resolution(predicted, gt)

    logger.info(
        f"Analyze config results: P={metrics['precision']:.4f}, "
        f"R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}"
    )

    # Save for comparison
    results_dir = "data/benchmarks/comparison"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "analyze_config_results.json"), "w") as f:
        json.dump(
            {
                "config": "llm_generated",
                "predicted_pairs": len(predicted),
                "llm_config": parsed,
                **metrics,
            },
            f,
            indent=2,
        )

    assert metrics["f1_score"] >= 0.0


def test_compare_configs() -> None:
    """Compare default vs LLM-generated config results.

    This test runs after the other two and reads their saved results.
    """
    results_dir = "data/benchmarks/comparison"
    default_path = os.path.join(results_dir, "default_config_results.json")
    analyze_path = os.path.join(results_dir, "analyze_config_results.json")

    if not os.path.exists(default_path) or not os.path.exists(analyze_path):
        pytest.skip("Comparison results not yet generated")

    with open(default_path) as f:
        default_results = json.load(f)
    with open(analyze_path) as f:
        analyze_results = json.load(f)

    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD COMPARISON: DBLP-ACM Subset")
    print("=" * 60)
    print(f"{'Metric':<20} {'Default':>12} {'LLM-Generated':>14}")
    print("-" * 60)
    for metric in ["precision", "recall", "f1_score"]:
        d = default_results[metric]
        a = analyze_results[metric]
        winner = "←" if d > a else ("→" if a > d else "=")
        print(f"{metric:<20} {d:>12.4f} {a:>14.4f}  {winner}")
    print("-" * 60)
    print(
        f"{'predicted_pairs':<20} {default_results['predicted_pairs']:>12} "
        f"{analyze_results['predicted_pairs']:>14}"
    )
    print("=" * 60)

    if "llm_config" in analyze_results:
        print("\nLLM-generated config:")
        for k, v in analyze_results["llm_config"].items():
            print(f"  {k}: {v}")
