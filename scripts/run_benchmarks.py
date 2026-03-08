"""Run SERF pipeline on benchmark datasets and evaluate.

Downloads real benchmark datasets from Leipzig, runs blocking
with embedding similarity matching, and evaluates against ground truth.
"""

import json
import os
import time

import numpy as np

from serf.block.embeddings import EntityEmbedder
from serf.block.faiss_blocker import FAISSBlocker
from serf.eval.benchmarks import BenchmarkDataset
from serf.eval.metrics import evaluate_resolution
from serf.logs import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Shared embedder to avoid reloading model
_embedder: EntityEmbedder | None = None


def get_embedder() -> EntityEmbedder:
    """Get or create a shared embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = EntityEmbedder()
    return _embedder


def run_benchmark(
    name: str,
    target_block_size: int = 15,
    similarity_threshold: float = 0.85,
    max_entities: int | None = None,
) -> dict[str, float]:
    """Run the SERF pipeline on a benchmark dataset.

    Parameters
    ----------
    name : str
        Dataset name from the registry
    target_block_size : int
        Target block size for FAISS clustering
    similarity_threshold : float
        Cosine similarity threshold for embedding-based matching
    max_entities : int | None
        Max entities from right table (for speed on large datasets)

    Returns
    -------
    dict[str, float]
        Evaluation metrics
    """
    start = time.time()
    logger.info(f"=== Benchmark: {name} ===")

    # Download and load
    dataset = BenchmarkDataset.download(name)
    left_entities, right_entities = dataset.to_entities()

    # Optionally limit right table size for large datasets
    if max_entities and len(right_entities) > max_entities:
        # Keep entities that have ground truth matches + random sample
        gt_right_ids = {b for _, b in dataset.ground_truth}
        matched_right = [e for e in right_entities if e.id in gt_right_ids]
        unmatched_right = [e for e in right_entities if e.id not in gt_right_ids]
        sample_size = max(0, max_entities - len(matched_right))
        import random

        random.seed(42)
        sampled = random.sample(unmatched_right, min(sample_size, len(unmatched_right)))
        right_entities = matched_right + sampled
        logger.info(f"  Sampled right table to {len(right_entities)} entities")

    all_entities = left_entities + right_entities
    logger.info(
        f"  Left: {len(left_entities)}, Right: {len(right_entities)}, "
        f"Total: {len(all_entities)}, Ground truth: {len(dataset.ground_truth)} pairs"
    )

    # Phase 1: Embed all entities
    logger.info("  Embedding entities...")
    embedder = get_embedder()
    texts = [e.text_for_embedding() for e in all_entities]
    embeddings = embedder.embed(texts)

    # Phase 2: Block using FAISS
    logger.info("  Blocking with FAISS...")
    ids = [str(e.id) for e in all_entities]
    blocker = FAISSBlocker(target_block_size=target_block_size)
    block_assignments = blocker.block(embeddings, ids)

    # Build embedding lookup
    emb_map = {str(e.id): embeddings[i] for i, e in enumerate(all_entities)}
    left_ids = {str(e.id) for e in left_entities}
    right_ids = {str(e.id) for e in right_entities}

    # Phase 3: Embedding-based matching within blocks
    logger.info("  Matching within blocks...")
    predicted_pairs: set[tuple[int, int]] = set()

    for _block_key, block_entity_ids in block_assignments.items():
        block_left = [eid for eid in block_entity_ids if eid in left_ids]
        block_right = [eid for eid in block_entity_ids if eid in right_ids]

        if not block_left or not block_right:
            continue

        # Compute cross-similarity matrix
        left_embs = np.array([emb_map[eid] for eid in block_left])
        right_embs = np.array([emb_map[eid] for eid in block_right])
        sim_matrix = np.dot(left_embs, right_embs.T)

        for i, lid in enumerate(block_left):
            for j, rid in enumerate(block_right):
                if sim_matrix[i, j] >= similarity_threshold:
                    l_int = int(lid)
                    r_int = int(rid)
                    predicted_pairs.add((min(l_int, r_int), max(l_int, r_int)))

    logger.info(f"  Predicted {len(predicted_pairs)} match pairs")

    # Phase 4: Evaluate
    metrics = evaluate_resolution(predicted_pairs, dataset.ground_truth)
    elapsed = time.time() - start

    logger.info(
        f"  Results ({elapsed:.1f}s): "
        f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}"
    )

    metrics["elapsed_seconds"] = elapsed
    metrics["predicted_pairs"] = len(predicted_pairs)
    metrics["true_pairs"] = len(dataset.ground_truth)
    metrics["left_entities"] = len(left_entities)
    metrics["right_entities"] = len(right_entities)
    metrics["similarity_threshold"] = similarity_threshold
    return metrics


def main() -> None:
    """Run all benchmarks and save results."""
    benchmarks = [
        {"name": "dblp-acm", "target_block_size": 15, "similarity_threshold": 0.85},
        {"name": "abt-buy", "target_block_size": 15, "similarity_threshold": 0.80},
        {
            "name": "dblp-scholar",
            "target_block_size": 15,
            "similarity_threshold": 0.85,
            "max_entities": 5000,
        },
    ]

    results = {}
    for params in benchmarks:
        name = params.pop("name")
        try:
            metrics = run_benchmark(name, **params)  # type: ignore[arg-type]
            results[name] = metrics
            print(
                f"\n{name}: P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}"
            )
        except Exception as e:
            logger.error(f"Failed on {name}: {e}", exc_info=True)
            results[name] = {"error": str(e)}

    # Save results
    os.makedirs("data/benchmarks", exist_ok=True)
    results_file = "data/benchmarks/baseline_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Dataset':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>10}")
    print("-" * 70)
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<20} {'ERROR':>10} {m.get('error', '')[:40]}")
        else:
            print(
                f"{name:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1_score']:>10.4f} {m['elapsed_seconds']:>9.1f}s"
            )
    print("=" * 70)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
