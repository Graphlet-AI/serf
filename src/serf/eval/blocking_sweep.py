"""Compare embedding models on name-only blocking recall.

Blocking decides which record pairs the matcher is ever allowed to see, so the
share of gold pairs landing in a shared block is a hard ceiling on end-to-end
recall. This module measures that ceiling for a set of candidate embedding
models without making a single LLM call, which makes an embedding sweep cheap
enough to run over whole datasets.
"""

import time
from dataclasses import asdict, dataclass

from serf.block.pipeline import SemanticBlockingPipeline
from serf.config import config
from serf.dspy.types import Entity
from serf.eval.benchmarks import BenchmarkDataset
from serf.eval.sample import sample_records
from serf.logs import get_logger

logger = get_logger(__name__)


@dataclass
class BlockingSweepResult:
    """Blocking quality for one embedding model on one dataset.

    Parameters
    ----------
    dataset : str
        Benchmark name
    model : str
        Embedding model name
    prompt : str
        Instruction prefix applied to every text
    records : int
        Records blocked
    gold_pairs : int
        Gold pairs present among those records
    co_blocked : int
        Gold pairs whose two records share a block
    blocking_recall : float
        ``co_blocked / gold_pairs``, the recall ceiling for any matcher
    blocks : int
        Blocks produced
    avg_block_size : float
        Mean entities per block
    max_block_size : int
        Largest block
    reduction_ratio : float
        Share of all possible pairs that blocking eliminated
    elapsed_seconds : float
        Wall clock for embedding plus clustering
    """

    dataset: str
    model: str
    prompt: str
    records: int
    gold_pairs: int
    co_blocked: int
    blocking_recall: float
    blocks: int
    avg_block_size: float
    max_block_size: int
    reduction_ratio: float
    elapsed_seconds: float

    def as_dict(self) -> dict[str, object]:
        """Return the result as a plain dict."""
        return asdict(self)


def evaluate_blocking(
    entities: list[Entity],
    ground_truth: set[tuple[int, int]],
    dataset: str,
    model_name: str,
    prompt: str,
    target_block_size: int,
    max_block_size: int,
) -> BlockingSweepResult:
    """Block a record set with one embedding model and score pair completeness.

    Parameters
    ----------
    entities : list[Entity]
        Records to block, both sources concatenated
    ground_truth : set[tuple[int, int]]
        Gold pairs restricted to these records
    dataset : str
        Benchmark name, recorded on the result
    model_name : str
        Embedding model to test
    prompt : str
        Instruction prefix prepended to every text
    target_block_size : int
        Target entities per block
    max_block_size : int
        Oversized blocks are split at this size

    Returns
    -------
    BlockingSweepResult
        Blocking quality for this model
    """
    pipeline = SemanticBlockingPipeline(
        model_name=model_name,
        target_block_size=target_block_size,
        max_block_size=max_block_size,
        auto_scale=False,
        embedding_prompt=prompt,
    )

    start = time.time()
    blocks, metrics = pipeline.run(entities)
    elapsed = time.time() - start

    block_of: dict[int, int] = {}
    for index, block in enumerate(blocks):
        for entity in block.entities:
            block_of[int(entity.id)] = index

    co_blocked = sum(
        1 for left, right in ground_truth if block_of.get(left, -1) == block_of.get(right, -2)
    )
    recall = co_blocked / len(ground_truth) if ground_truth else 0.0

    logger.info(
        f"{dataset} {model_name} prompt={prompt!r}: "
        f"blocking recall {recall:.4f} ({co_blocked}/{len(ground_truth)}) "
        f"in {elapsed:.1f}s"
    )

    return BlockingSweepResult(
        dataset=dataset,
        model=model_name,
        prompt=prompt,
        records=len(entities),
        gold_pairs=len(ground_truth),
        co_blocked=co_blocked,
        blocking_recall=recall,
        blocks=metrics.total_blocks,
        avg_block_size=metrics.avg_block_size,
        max_block_size=metrics.max_block_size,
        reduction_ratio=metrics.reduction_ratio,
        elapsed_seconds=elapsed,
    )


def sweep_dataset(
    dataset: str,
    candidates: list[tuple[str, str]],
    output_dir: str | None = None,
    sample: int = 0,
    seed: int = 42,
    target_block_size: int | None = None,
    max_block_size: int | None = None,
) -> list[BlockingSweepResult]:
    """Score every candidate embedding model on one benchmark.

    All candidates see the identical record set, so their recalls are directly
    comparable.

    Parameters
    ----------
    dataset : str
        Benchmark name
    candidates : list[tuple[str, str]]
        ``(model_name, prompt)`` pairs to test
    output_dir : str | None
        Directory holding the downloaded benchmark. Defaults to config.
    sample : int
        Record budget, or 0 to use the whole dataset
    seed : int
        Sampling seed
    target_block_size : int | None
        Target entities per block. Defaults to config.
    max_block_size : int | None
        Split threshold. Defaults to config.

    Returns
    -------
    list[BlockingSweepResult]
        One result per candidate, in the order given
    """
    if target_block_size is None:
        target_block_size = int(config.get("er.blocking.target_block_size", 30))
    if max_block_size is None:
        max_block_size = int(config.get("er.blocking.max_block_size", 100))

    benchmark = BenchmarkDataset.download(dataset, output_dir)
    left, right = benchmark.to_entities()
    entities = left + right
    ground_truth = benchmark.ground_truth

    if sample and sample < len(entities):
        drawn = sample_records(entities, ground_truth, sample, seed=seed)
        entities = drawn.records
        ground_truth = drawn.ground_truth

    logger.info(
        f"Sweeping {len(candidates)} embeddings on {dataset}: "
        f"{len(entities)} records, {len(ground_truth)} gold pairs"
    )

    results: list[BlockingSweepResult] = []
    for model_name, prompt in candidates:
        results.append(
            evaluate_blocking(
                entities=entities,
                ground_truth=ground_truth,
                dataset=dataset,
                model_name=model_name,
                prompt=prompt,
                target_block_size=target_block_size,
                max_block_size=max_block_size,
            )
        )
    return results
