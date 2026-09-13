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
    smaller than the requested budgets, val and holdout are still honoured in
    full and train takes whatever is left.

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

    val_n, holdout_n, train_n = _fit_budgets(len(entities), val_n, holdout_n, train_n)
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


EVAL_SPLIT_HOLDOUT = "holdout"
EVAL_SPLIT_VAL = "val"
EVAL_SPLIT_TRAIN = "train"
EVAL_SPLIT_ALL = "all"
EVAL_SPLITS = (EVAL_SPLIT_HOLDOUT, EVAL_SPLIT_VAL, EVAL_SPLIT_TRAIN, EVAL_SPLIT_ALL)

# The splits a prompt is fitted to. Training reads train, GEPA selects on val,
# so a score on either is reading the fit back. Only holdout answers the
# question a benchmark is asked.
FITTED_SPLITS = (EVAL_SPLIT_VAL, EVAL_SPLIT_TRAIN, EVAL_SPLIT_ALL)


@dataclass
class EvalSelection:
    """The records one benchmark run will score, and where they came from.

    Parameters
    ----------
    split : str
        Which split was selected
    records : list[Entity]
        Records to score, sorted by id
    ground_truth : set[tuple[int, int]]
        Ground truth restricted to pairs with both records inside ``records``
    total : int
        Records in the whole dataset
    gold_pairs : int
        Gold pairs that survived the restriction
    overlaps_training : bool
        Whether these records are ones ``serf train`` reads or selects on
    """

    split: str
    records: list[Entity]
    ground_truth: set[tuple[int, int]]
    total: int
    gold_pairs: int
    overlaps_training: bool

    def describe(self) -> str:
        """Render the selection as one line for a CLI.

        Returns
        -------
        str
            Split, record count, gold pairs, and whether training saw it
        """
        seen = "training saw these records" if self.overlaps_training else "unseen by training"
        return (
            f"{self.split}: {len(self.records)} of {self.total} records, "
            f"{self.gold_pairs} gold pairs, {seen}"
        )


def select_eval_split(
    dataset: str,
    entities: list[Entity],
    ground_truth: set[tuple[int, int]],
    split: str | None = None,
    seed: int | None = None,
    sizes: SplitSizes | None = None,
) -> EvalSelection:
    """Pick the records a benchmark run should score.

    The partition is the one ``serf train`` draws, from the same budgets and
    the same seed, so the holdout selected here is exactly the holdout training
    left alone. That is the whole point: evaluation data and training data are
    separated once, by construction, rather than being kept apart by whoever
    remembers to pass the right flags.

    Parameters
    ----------
    dataset : str
        Benchmark dataset name, used to look up the configured budgets
    entities : list[Entity]
        Every record in the dataset
    ground_truth : set[tuple[int, int]]
        True matching pairs over the full dataset
    split : str | None
        One of ``EVAL_SPLITS``. Defaults to config ``benchmarks.eval_split``.
    seed : int | None
        RNG seed. Defaults to config ``optimize.seed``, which is what
        ``serf train`` uses.
    sizes : SplitSizes | None
        Record budgets. Defaults to this dataset's configured budgets.

    Returns
    -------
    EvalSelection
        The records to score and their provenance

    Raises
    ------
    ValueError
        If the split name is not one SERF knows
    """
    split = split or str(config.get("benchmarks.eval_split", EVAL_SPLIT_HOLDOUT))
    if split not in EVAL_SPLITS:
        raise ValueError(f"Unknown eval split {split!r}. Available: {EVAL_SPLITS}")

    if split == EVAL_SPLIT_ALL:
        return EvalSelection(
            split=split,
            records=sorted(entities, key=lambda entity: entity.id),
            ground_truth=set(ground_truth),
            total=len(entities),
            gold_pairs=count_gold_pairs(entities, ground_truth),
            overlaps_training=True,
        )

    sizes = sizes or get_split_sizes(dataset)
    splits = sample_random_splits(
        entities,
        ground_truth,
        train_records=sizes.train_records,
        val_records=sizes.val_records,
        holdout_records=sizes.holdout_records,
        seed=seed,
    )
    chosen = {
        EVAL_SPLIT_HOLDOUT: splits.holdout_records,
        EVAL_SPLIT_VAL: splits.val_records,
        EVAL_SPLIT_TRAIN: splits.train_records,
    }[split]

    ids = {entity.id for entity in chosen}
    kept = {(left, right) for left, right in ground_truth if left in ids and right in ids}
    selection = EvalSelection(
        split=split,
        records=sorted(chosen, key=lambda entity: entity.id),
        ground_truth=kept,
        total=len(entities),
        gold_pairs=len(kept),
        overlaps_training=split in FITTED_SPLITS,
    )
    logger.info(f"Evaluation split selected for {dataset}: {selection.describe()}")
    if not selection.records:
        logger.warning(f"Split {split} of {dataset} is empty; raise benchmarks.{split}_records")
    elif not kept:
        logger.warning(f"Split {split} of {dataset} holds no gold pair and cannot be scored")
    return selection


def training_overlap(
    dataset: str,
    scored: list[Entity],
    all_entities: list[Entity],
    ground_truth: set[tuple[int, int]],
    seed: int | None = None,
) -> float:
    """Share of ``scored`` that ``serf train`` would put in train or val.

    A trained prompt measured on records GEPA trained or selected on is reading
    its own fit, not a result. This reports how much of that has happened, so a
    benchmark run can say so. The default 1,000-record sample overlaps
    validation entirely, because ``sample_records`` and ``sample_random_splits``
    shuffle the same list with the same seed and splits fill validation first.

    The partition is drawn over ``all_entities``, since that is what training
    sees; comparing against a partition of the sample alone would answer a
    different question.

    Parameters
    ----------
    dataset : str
        Benchmark dataset name, used to look up the configured budgets
    scored : list[Entity]
        Records about to be scored
    all_entities : list[Entity]
        Every record in the dataset, which is what ``serf train`` partitions
    ground_truth : set[tuple[int, int]]
        True matching pairs over the full dataset, used to keep match groups
        whole as training does
    seed : int | None
        RNG seed. Defaults to config ``optimize.seed``, which is what
        ``serf train`` uses, so the partition matches the one it trained on.

    Returns
    -------
    float
        Fraction of ``scored`` in the train or val split, 0.0 when none are
    """
    if not scored:
        return 0.0
    sizes = get_split_sizes(dataset)
    splits = sample_random_splits(
        all_entities,
        ground_truth,
        train_records=sizes.train_records,
        val_records=sizes.val_records,
        holdout_records=sizes.holdout_records,
        seed=seed,
    )
    seen = {entity.id for entity in splits.train_records} | {
        entity.id for entity in splits.val_records
    }
    return len({entity.id for entity in scored} & seen) / len(scored)


def _fit_budgets(total: int, *budgets: int) -> tuple[int, ...]:
    """Trim record budgets to the records available, honouring fill order.

    Each budget is satisfied in full before the next one gets anything, so the
    shortfall always lands on the last split rather than being spread across
    all of them. Fill order is val, holdout, train, which makes train the split
    that absorbs a small dataset.

    Scaling all three proportionally, which is what this used to do, keeps
    their ratio but lets a large train budget push validation towards zero -
    the same starvation that once left a GEPA run with a single validation
    record and no way to tell its candidates apart.

    Parameters
    ----------
    total : int
        Records available
    *budgets : int
        Requested record budgets in fill order

    Returns
    -------
    tuple[int, ...]
        Budgets that fit within ``total``
    """
    requested = sum(budgets)
    if requested <= total:
        return budgets

    remaining = total
    fitted: list[int] = []
    for budget in budgets:
        taken = min(budget, remaining)
        fitted.append(taken)
        remaining -= taken
    logger.warning(
        f"Dataset has {total} records but {requested} were requested; filling in order to "
        f"{tuple(fitted)}, so the shortfall falls on the last split rather than on evaluation"
    )
    return tuple(fitted)


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
