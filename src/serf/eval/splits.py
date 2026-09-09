"""Train/val/holdout splits for GEPA from fully blocked benchmark data."""

from __future__ import annotations

import random
from dataclasses import dataclass

from serf.config import config
from serf.dspy.types import EntityBlock
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
    """Disjoint train, val, and holdout partitions of the same semantic blocks."""

    train_blocks: list[EntityBlock]
    val_blocks: list[EntityBlock]
    holdout_blocks: list[EntityBlock]

    @property
    def train_record_count(self) -> int:
        """Number of records in the train blocks.

        Returns
        -------
        int
            Total entities across train blocks
        """
        return sum(block.block_size for block in self.train_blocks)

    @property
    def val_record_count(self) -> int:
        """Number of records in the val blocks.

        Returns
        -------
        int
            Total entities across val blocks
        """
        return sum(block.block_size for block in self.val_blocks)

    @property
    def holdout_record_count(self) -> int:
        """Number of records in the holdout blocks.

        Returns
        -------
        int
            Total entities across holdout blocks
        """
        return sum(block.block_size for block in self.holdout_blocks)


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
    blocks: list[EntityBlock],
    *,
    train_blocks: int | None = None,
    val_records: int | None = None,
    holdout_records: int | None = None,
    seed: int | None = None,
    min_block_size: int | None = None,
) -> BenchmarkSplits:
    """Partition semantic blocks into disjoint train, val, and holdout splits.

    Val and holdout budgets are filled first so they are never starved when the
    dataset produces far fewer blocks than ``train_blocks``. All three splits
    hold real semantic blocks, so val and holdout contain true duplicate pairs
    and are therefore scoreable. Blocks are assigned whole, so the splits are
    disjoint by entity id.

    Parameters
    ----------
    blocks : list[EntityBlock]
        Blocks from running semantic blocking on *all* entities
    train_blocks : int | None
        Maximum number of blocks for GEPA training
    val_records : int | None
        Record budget for GEPA validation (whole blocks are added until met)
    holdout_records : int | None
        Record budget held out for final evaluation
    seed : int | None
        RNG seed
    min_block_size : int | None
        Skip blocks smaller than this

    Returns
    -------
    BenchmarkSplits
        Disjoint train, val, and holdout blocks
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

    train: list[EntityBlock] = []
    val: list[EntityBlock] = []
    holdout: list[EntityBlock] = []
    used_ids: set[int] = set()
    val_count = 0
    holdout_count = 0

    for block in candidates:
        block_ids = {entity.id for entity in block.entities}
        if block_ids & used_ids:
            continue
        if val_count < val_n:
            val.append(block)
            val_count += block.block_size
        elif holdout_count < holdout_n:
            holdout.append(block)
            holdout_count += block.block_size
        elif len(train) < train_n:
            train.append(block)
        else:
            break
        used_ids |= block_ids

    splits = BenchmarkSplits(train_blocks=train, val_blocks=val, holdout_blocks=holdout)
    logger.info(
        f"Sampled splits from {len(candidates)} eligible blocks of {len(blocks)} total: "
        f"train {len(splits.train_blocks)} blocks / {splits.train_record_count} records, "
        f"val {len(splits.val_blocks)} blocks / {splits.val_record_count} records, "
        f"holdout {len(splits.holdout_blocks)} blocks / {splits.holdout_record_count} records"
    )
    if not splits.val_blocks:
        logger.warning("Val split is empty: GEPA has no signal to select candidate programs")
    if not splits.train_blocks:
        logger.warning("Train split is empty: val and holdout budgets consumed every block")
    return splits


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
