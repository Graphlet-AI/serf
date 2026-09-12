"""Random train/val/holdout record splits for GEPA over benchmark data."""

from __future__ import annotations

import random
from dataclasses import dataclass

from serf.config import config
from serf.dspy.types import Entity
from serf.logs import get_logger

logger = get_logger(__name__)

_DEFAULT_TRAIN_RECORDS = 2000
_DEFAULT_VAL_RECORDS = 1000
_DEFAULT_HOLDOUT_RECORDS = 1000
_DEFAULT_SEED = 42


@dataclass(frozen=True)
class SplitSizes:
    """Configured record budgets for one benchmark dataset."""

    train_records: int
    val_records: int
    holdout_records: int


@dataclass
class BenchmarkSplits:
    """Disjoint train, val, and holdout records sampled from one dataset."""

    train_records: list[Entity]
    val_records: list[Entity]
    holdout_records: list[Entity]


def get_split_sizes(dataset: str) -> SplitSizes:
    """Return train/val/holdout record budgets for a benchmark dataset.

    Parameters
    ----------
    dataset : str
        Benchmark dataset name (e.g. ``dblp-acm``)

    Returns
    -------
    SplitSizes
        Configured record counts
    """
    return SplitSizes(
        train_records=_dataset_int(dataset, "train_records", _DEFAULT_TRAIN_RECORDS),
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


def match_groups(
    entities: list[Entity],
    ground_truth: set[tuple[int, int]],
) -> dict[int, list[Entity]]:
    """Group records into connected components of the ground-truth pair graph.

    A record and everything transitively matched to it share one group, so a
    group can be sampled as a unit and its gold pairs always survive.

    Parameters
    ----------
    entities : list[Entity]
        Records to group
    ground_truth : set[tuple[int, int]]
        True matching pairs; ids outside ``entities`` are ignored

    Returns
    -------
    dict[int, list[Entity]]
        Entity id mapped to its group, sorted by id and shared by group members
    """
    by_id = {entity.id: entity for entity in entities}
    parent = {entity_id: entity_id for entity_id in by_id}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in ground_truth:
        if left not in parent or right not in parent:
            continue
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    members: dict[int, list[Entity]] = {}
    for entity_id in by_id:
        members.setdefault(find(entity_id), []).append(by_id[entity_id])

    groups: dict[int, list[Entity]] = {}
    for group in members.values():
        group.sort(key=lambda entity: entity.id)
        for entity in group:
            groups[entity.id] = group
    return groups


def count_gold_pairs(records: list[Entity], ground_truth: set[tuple[int, int]]) -> int:
    """Count ground-truth pairs with both records inside a split.

    Parameters
    ----------
    records : list[Entity]
        Records in one split
    ground_truth : set[tuple[int, int]]
        True matching pairs

    Returns
    -------
    int
        Pairs fully contained in ``records``
    """
    ids = {record.id for record in records}
    return sum(1 for left, right in ground_truth if left in ids and right in ids)


def sample_random_splits(
    entities: list[Entity],
    ground_truth: set[tuple[int, int]],
    *,
    train_records: int | None = None,
    val_records: int | None = None,
    holdout_records: int | None = None,
    seed: int | None = None,
) -> BenchmarkSplits:
    """Randomly sample disjoint train, val, and holdout records by match group.

    Records are drawn in random order; drawing a record also pulls in every
    record transitively matched to it, so gold pairs are never split across two
    splits and never lost to independent uniform sampling. Val is filled first,
    then holdout, then train, so val is never starved. When the dataset is
    smaller than the requested budgets, all three are scaled down proportionally.

    Parameters
    ----------
    entities : list[Entity]
        Full entity list
    ground_truth : set[tuple[int, int]]
        True matching pairs, used to keep match groups whole
    train_records : int | None
        Record budget for GEPA training
    val_records : int | None
        Record budget for GEPA validation
    holdout_records : int | None
        Record budget held out for final evaluation
    seed : int | None
        RNG seed

    Returns
    -------
    BenchmarkSplits
        Disjoint train, val, and holdout records
    """
    train_n = (
        train_records
        if train_records is not None
        else int(config.get("benchmarks.train_records", _DEFAULT_TRAIN_RECORDS))
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

    val_n, holdout_n, train_n = _scaled_budgets(len(entities), val_n, holdout_n, train_n)
    groups = match_groups(entities, ground_truth)

    order = list(entities)
    random.Random(seed).shuffle(order)

    buckets: list[list[Entity]] = [[], [], []]
    budgets = [val_n, holdout_n, train_n]
    assigned: set[int] = set()
    current = 0
    for entity in order:
        if entity.id in assigned:
            continue
        while current < len(buckets) and len(buckets[current]) >= budgets[current]:
            current += 1
        if current >= len(buckets):
            break
        group = groups[entity.id]
        buckets[current].extend(group)
        assigned.update(member.id for member in group)

    val, holdout, train = buckets
    splits = BenchmarkSplits(train_records=train, val_records=val, holdout_records=holdout)
    logger.info(
        f"Random match-group splits over {len(entities)} records "
        f"(budgets train={train_n} val={val_n} holdout={holdout_n}): "
        f"train {len(train)} records / {count_gold_pairs(train, ground_truth)} gold pairs, "
        f"val {len(val)} records / {count_gold_pairs(val, ground_truth)} gold pairs, "
        f"holdout {len(holdout)} records / {count_gold_pairs(holdout, ground_truth)} gold pairs"
    )
    for name, records in (("train", train), ("val", val), ("holdout", holdout)):
        if ground_truth and not count_gold_pairs(records, ground_truth):
            logger.warning(f"Split {name} contains no gold pairs and cannot be scored")
    return splits


def _scaled_budgets(total: int, *budgets: int) -> tuple[int, ...]:
    """Scale record budgets down proportionally when the dataset is too small.

    Parameters
    ----------
    total : int
        Records available
    *budgets : int
        Requested record budgets in fill order

    Returns
    -------
    tuple[int, ...]
        Budgets that fit within ``total``, keeping their relative ratio
    """
    requested = sum(budgets)
    if requested <= total or requested == 0:
        return budgets
    scale = total / requested
    scaled = tuple(int(budget * scale) for budget in budgets)
    logger.warning(
        f"Dataset has {total} records but {requested} were requested; scaling budgets to {scaled}"
    )
    return scaled


def _dataset_int(dataset: str, key: str, default: int) -> int:
    """Read an integer split size from dataset config with a global fallback.

    Parameters
    ----------
    dataset : str
        Dataset name
    key : str
        Config key (``train_records``, ``val_records``, ``holdout_records``)
    default : int
        Fallback if neither dataset nor global key is set

    Returns
    -------
    int
        Configured size
    """
    global_default = int(config.get(f"benchmarks.{key}", default))
    return int(config.get(f"benchmarks.datasets.{dataset}.{key}", global_default))
