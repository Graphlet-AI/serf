"""Train/val/holdout splits for GEPA from fully blocked benchmark data."""

from __future__ import annotations

import random
from dataclasses import dataclass

from serf.config import config
from serf.dspy.types import Entity, EntityBlock
from serf.logs import get_logger

logger = get_logger(__name__)

_DEFAULT_TRAIN_BLOCKS = 1000
_DEFAULT_VAL_RECORDS = 500
_DEFAULT_HOLDOUT_RECORDS = 1000
_DEFAULT_SEED = 42


@dataclass(frozen=True)
class SplitSizes:
    """Configured sample sizes for one benchmark dataset."""

    train_blocks: int
    val_records: int
    holdout_records: int


@dataclass
class BenchmarkSplits:
    """Blocked train sample plus disjoint val and holdout records."""

    train_blocks: list[EntityBlock]
    val_records: list[Entity]
    holdout_records: list[Entity]


def get_split_sizes(dataset: str) -> SplitSizes:
    """Return train/val/holdout sizes for a benchmark dataset.

    Parameters
    ----------
    dataset : str
        Benchmark dataset name (e.g. ``dblp-acm``)

    Returns
    -------
    SplitSizes
        Configured block and record counts
    """
    return SplitSizes(
        train_blocks=_dataset_int(dataset, "train_blocks", _DEFAULT_TRAIN_BLOCKS),
        val_records=_dataset_int(dataset, "val_records", _DEFAULT_VAL_RECORDS),
        holdout_records=_dataset_int(dataset, "holdout_records", _DEFAULT_HOLDOUT_RECORDS),
    )


def get_all_split_sizes() -> dict[str, SplitSizes]:
    """Return split sizes for every configured benchmark dataset.

    Returns
    -------
    dict[str, SplitSizes]
        Mapping of dataset name to split sizes
    """
    datasets = config.get("benchmarks.datasets", {})
    return {name: get_split_sizes(name) for name in datasets}


def sample_blocked_splits(
    entities: list[Entity],
    blocks: list[EntityBlock],
    *,
    train_blocks: int | None = None,
    val_records: int | None = None,
    holdout_records: int | None = None,
    seed: int | None = None,
    min_block_size: int | None = None,
) -> BenchmarkSplits:
    """Block-all then sample: 1K train blocks, 500 val records, 1K holdout.

    Trains on a random sample of semantic blocks after blocking the full
    dataset. Validation and holdout records are drawn from entities that
    are not in the sampled train blocks, so the three sets are disjoint.

    Parameters
    ----------
    entities : list[Entity]
        Full entity list that was blocked
    blocks : list[EntityBlock]
        Blocks from running semantic blocking on *all* entities
    train_blocks : int | None
        Number of blocks to sample for GEPA training
    val_records : int | None
        Number of leftover records for GEPA validation
    holdout_records : int | None
        Number of leftover records held out for final evaluation
    seed : int | None
        RNG seed
    min_block_size : int | None
        Skip blocks smaller than this when sampling train blocks

    Returns
    -------
    BenchmarkSplits
        Disjoint train blocks, val records, and holdout records
    """
    train_n = (
        train_blocks
        if train_blocks is not None
        else int(config.get("benchmarks.train_blocks", _DEFAULT_TRAIN_BLOCKS))
    )
    val_n = (
        val_records
        if val_records is not None
        else int(config.get("benchmarks.val_records", _DEFAULT_VAL_RECORDS))
    )
    holdout_n = (
        holdout_records
        if holdout_records is not None
        else int(config.get("benchmarks.holdout_records", _DEFAULT_HOLDOUT_RECORDS))
    )
    seed = seed if seed is not None else int(config.get("optimize.seed", _DEFAULT_SEED))
    min_size = (
        min_block_size
        if min_block_size is not None
        else int(config.get("er.blocking.min_block_size", 2))
    )

    rng = random.Random(seed)
    candidates = [block for block in blocks if block.block_size >= min_size]
    rng.shuffle(candidates)
    selected = candidates[: min(train_n, len(candidates))]

    train_ids = {entity.id for block in selected for entity in block.entities}
    remaining = [entity for entity in entities if entity.id not in train_ids]
    rng.shuffle(remaining)

    n_val = min(val_n, len(remaining))
    val = remaining[:n_val]
    leftover = remaining[n_val:]
    n_holdout = min(holdout_n, len(leftover))
    holdout = leftover[:n_holdout]

    logger.info(
        f"Sampled splits: {len(selected)} train blocks "
        f"({len(train_ids)} records), {len(val)} val records, "
        f"{len(holdout)} holdout records"
    )
    return BenchmarkSplits(
        train_blocks=selected,
        val_records=val,
        holdout_records=holdout,
    )


def chunk_records(records: list[Entity], block_size: int, prefix: str = "val") -> list[EntityBlock]:
    """Pack records into fixed-size blocks for GEPA val examples.

    Parameters
    ----------
    records : list[Entity]
        Records to pack
    block_size : int
        Target entities per block
    prefix : str
        Block key prefix

    Returns
    -------
    list[EntityBlock]
        Chunks of ``block_size`` (last chunk may be smaller)
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    chunks: list[EntityBlock] = []
    for start in range(0, len(records), block_size):
        chunk = records[start : start + block_size]
        chunks.append(
            EntityBlock(
                block_key=f"{prefix}_{start // block_size}",
                block_key_type="sample",
                block_size=len(chunk),
                entities=chunk,
            )
        )
    return chunks


def _dataset_int(dataset: str, key: str, default: int) -> int:
    """Read an integer split size from dataset config with a global fallback.

    Parameters
    ----------
    dataset : str
        Dataset name
    key : str
        Config key (``train_blocks``, ``val_records``, ``holdout_records``)
    default : int
        Fallback if neither dataset nor global key is set

    Returns
    -------
    int
        Configured size
    """
    global_default = int(config.get(f"benchmarks.{key}", default))
    return int(config.get(f"benchmarks.datasets.{dataset}.{key}", global_default))
