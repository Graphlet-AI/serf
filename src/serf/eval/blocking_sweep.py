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
from serf.match.run import entity_members, merge_matched_entities

logger = get_logger(__name__)


@dataclass(frozen=True)
class EmbeddingCandidate:
    """One embedding configuration to score.

    Parameters
    ----------
    model : str
        Hugging Face model name
    prompt : str
        Instruction prefix prepended to every text
    trust_remote_code : bool
        Execute the modelling code shipped in the model repository. Several
        architectures will not load without it, and it runs third-party code,
        so it is opt-in per candidate rather than global.
    """

    model: str
    prompt: str = ""
    trust_remote_code: bool = False

    @property
    def label(self) -> str:
        """Return a short display name."""
        return self.model.split("/")[-1] + (" +prefix" if self.prompt else "")


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
    round_number : int
        ER round this result describes, counting from one
    cumulative_co_blocked : int
        Gold pairs co-blocked in this round or any earlier one
    cumulative_recall : float
        ``cumulative_co_blocked / gold_pairs``, the ceiling a multi-round run
        can reach with a perfect matcher
    trust_remote_code : bool
        Whether the model repository's own code was executed to load it
    strategy : str
        Text fed to the embedding: ``"name"`` or ``"json"``
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
    round_number: int = 1
    cumulative_co_blocked: int = 0
    cumulative_recall: float = 0.0
    trust_remote_code: bool = False
    strategy: str = "name"

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
    trust_remote_code: bool = False,
    strategy: str = "name",
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
    trust_remote_code : bool
        Execute the modelling code shipped in the model repository
    strategy : str
        ``"name"`` or ``"json"``

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
        embedding_trust_remote_code=trust_remote_code,
        blocking_strategy=strategy,
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
        cumulative_co_blocked=co_blocked,
        cumulative_recall=recall,
        trust_remote_code=trust_remote_code,
        strategy=strategy,
    )


def evaluate_blocking_rounds(
    entities: list[Entity],
    ground_truth: set[tuple[int, int]],
    dataset: str,
    model_name: str,
    prompt: str,
    target_block_size: int,
    max_block_size: int,
    rounds: int,
    trust_remote_code: bool = False,
    strategy: str = "name",
) -> list[BlockingSweepResult]:
    """Measure how much blocking recall extra ER rounds recover.

    Splitting an oversized block separates pairs that clustering had put
    together, but a later round re-blocks the entities the previous round merged,
    so a separated pair can still meet. This models a perfect matcher: every gold
    pair that lands in a shared block is merged before the next round. The
    cumulative recall is therefore the ceiling iteration can reach, and any
    shortfall against a single unsplit round is recall that rounds cannot recover.

    Parameters
    ----------
    entities : list[Entity]
        Records to block, both sources concatenated
    ground_truth : set[tuple[int, int]]
        Gold pairs restricted to these records
    dataset : str
        Benchmark name, recorded on each result
    model_name : str
        Embedding model to test
    prompt : str
        Instruction prefix prepended to every text
    target_block_size : int
        Target entities per block
    max_block_size : int
        Oversized blocks are split at this size
    rounds : int
        ER rounds to simulate
    trust_remote_code : bool
        Execute the modelling code shipped in the model repository
    strategy : str
        ``"name"`` or ``"json"``

    Returns
    -------
    list[BlockingSweepResult]
        One result per round, carrying both that round's recall and the
        cumulative recall through it
    """
    current = entities
    covered: set[tuple[int, int]] = set()
    results: list[BlockingSweepResult] = []

    for round_number in range(1, rounds + 1):
        pipeline = SemanticBlockingPipeline(
            model_name=model_name,
            target_block_size=target_block_size,
            max_block_size=max_block_size,
            iteration=round_number,
            embedding_prompt=prompt,
            embedding_trust_remote_code=trust_remote_code,
            blocking_strategy=strategy,
        )

        start = time.time()
        blocks, metrics = pipeline.run(current)
        elapsed = time.time() - start

        members = entity_members(current)
        owner = {record: entity for entity, records in members.items() for record in records}
        block_of = {
            int(entity.id): index for index, block in enumerate(blocks) for entity in block.entities
        }

        matched: set[tuple[int, int]] = set()
        for left, right in ground_truth:
            left_entity = owner.get(left)
            right_entity = owner.get(right)
            if left_entity is None or right_entity is None:
                continue
            if left_entity == right_entity:
                covered.add((left, right))
                continue
            if block_of.get(left_entity, -1) == block_of.get(right_entity, -2):
                covered.add((left, right))
                matched.add((left_entity, right_entity))

        this_round = sum(
            1
            for left, right in ground_truth
            if block_of.get(owner.get(left, -1), -1) == block_of.get(owner.get(right, -2), -2)
        )
        recall = this_round / len(ground_truth) if ground_truth else 0.0
        cumulative = len(covered) / len(ground_truth) if ground_truth else 0.0

        logger.info(
            f"{dataset} {model_name} round {round_number}: "
            f"recall {recall:.4f}, cumulative {cumulative:.4f} "
            f"({len(covered)}/{len(ground_truth)}) over {len(current)} entities in {elapsed:.1f}s"
        )

        results.append(
            BlockingSweepResult(
                dataset=dataset,
                model=model_name,
                prompt=prompt,
                records=len(current),
                gold_pairs=len(ground_truth),
                co_blocked=this_round,
                blocking_recall=recall,
                blocks=metrics.total_blocks,
                avg_block_size=metrics.avg_block_size,
                max_block_size=metrics.max_block_size,
                reduction_ratio=metrics.reduction_ratio,
                elapsed_seconds=elapsed,
                round_number=round_number,
                cumulative_co_blocked=len(covered),
                cumulative_recall=cumulative,
                trust_remote_code=trust_remote_code,
                strategy=strategy,
            )
        )

        if not matched:
            logger.info(f"{dataset} round {round_number} merged nothing; stopping early")
            break
        current = merge_matched_entities(current, matched)

    return results


def sweep_dataset(
    dataset: str,
    candidates: list[EmbeddingCandidate],
    output_dir: str | None = None,
    sample: int = 0,
    seed: int = 42,
    target_block_size: int | None = None,
    max_block_size: int | None = None,
    rounds: int = 1,
    strategy: str = "name",
) -> list[BlockingSweepResult]:
    """Score every candidate embedding model on one benchmark.

    All candidates see the identical record set, so their recalls are directly
    comparable.

    Parameters
    ----------
    dataset : str
        Benchmark name
    candidates : list[EmbeddingCandidate]
        Embedding configurations to test
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
    rounds : int
        ER rounds to simulate per candidate. One means a single blocking pass.
    strategy : str
        Text fed to the embedding: ``"name"`` or ``"json"``

    Returns
    -------
    list[BlockingSweepResult]
        One result per candidate per round, in the order given
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
    for candidate in candidates:
        if rounds > 1:
            results.extend(
                evaluate_blocking_rounds(
                    entities=entities,
                    ground_truth=ground_truth,
                    dataset=dataset,
                    model_name=candidate.model,
                    prompt=candidate.prompt,
                    target_block_size=target_block_size,
                    max_block_size=max_block_size,
                    rounds=rounds,
                    trust_remote_code=candidate.trust_remote_code,
                    strategy=strategy,
                )
            )
        else:
            results.append(
                evaluate_blocking(
                    entities=entities,
                    ground_truth=ground_truth,
                    dataset=dataset,
                    model_name=candidate.model,
                    prompt=candidate.prompt,
                    target_block_size=target_block_size,
                    max_block_size=max_block_size,
                    trust_remote_code=candidate.trust_remote_code,
                    strategy=strategy,
                )
            )
    return results
