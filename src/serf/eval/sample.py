"""Match-group-aware random record sampling for benchmark runs.

Sampling benchmark records independently at uniform random destroys the gold
pairs: a pair survives only when both of its records are drawn, so retention
falls with the square of the sampling fraction. A 1,000-record sample of
DBLP-Scholar (66,879 records, 5,347 pairs) would retain about one gold pair and
score nothing meaningful.

This module draws whole ground-truth match groups instead, reusing the same match
group logic as the GEPA splits in ``serf.eval.splits``, and returns the ground
truth restricted to the sample so precision and recall are computed against the
pairs that are actually present.
"""

import random
from dataclasses import dataclass, field

from serf.dspy.types import Entity
from serf.eval.splits import count_gold_pairs, match_groups
from serf.logs import get_logger

logger = get_logger(__name__)

_DEFAULT_SEED = 42


@dataclass(frozen=True)
class RecordSample:
    """A sampled subset of a benchmark dataset and its surviving ground truth.

    Parameters
    ----------
    records : list[Entity]
        Sampled records, sorted by id
    ground_truth : set[tuple[int, int]]
        Ground-truth pairs with both records inside the sample
    requested : int
        Record budget that was asked for
    total : int
        Records available in the full dataset
    """

    records: list[Entity]
    ground_truth: set[tuple[int, int]] = field(default_factory=set)
    requested: int = 0
    total: int = 0

    @property
    def gold_pairs(self) -> int:
        """Number of gold pairs retained in the sample."""
        return len(self.ground_truth)


def sample_records(
    entities: list[Entity],
    ground_truth: set[tuple[int, int]],
    count: int,
    seed: int | None = None,
) -> RecordSample:
    """Randomly sample records while keeping ground-truth match groups whole.

    Records are visited in seeded random order, and drawing a record also pulls in
    every record transitively matched to it, so a drawn pair is never half
    sampled. The sample can therefore overshoot the budget slightly, by at most
    the size of the last match group drawn.

    Parameters
    ----------
    entities : list[Entity]
        Full record list, both sources concatenated
    ground_truth : set[tuple[int, int]]
        True matching pairs over the full dataset
    count : int
        Record budget. Values at or above the dataset size return everything.
    seed : int | None
        RNG seed

    Returns
    -------
    RecordSample
        Sampled records and the ground truth restricted to them
    """
    seed = _DEFAULT_SEED if seed is None else seed
    if count >= len(entities):
        logger.info(
            f"Requested {count} records but the dataset has {len(entities)}; using all of them"
        )
        return RecordSample(
            records=sorted(entities, key=lambda entity: entity.id),
            ground_truth=set(ground_truth),
            requested=count,
            total=len(entities),
        )

    groups = match_groups(entities, ground_truth)
    order = list(entities)
    random.Random(seed).shuffle(order)

    sampled: list[Entity] = []
    drawn: set[int] = set()
    for entity in order:
        if len(sampled) >= count:
            break
        if entity.id in drawn:
            continue
        group = groups[entity.id]
        sampled.extend(group)
        drawn.update(member.id for member in group)

    sampled.sort(key=lambda entity: entity.id)
    sampled_ids = {entity.id for entity in sampled}
    kept = {pair for pair in ground_truth if pair[0] in sampled_ids and pair[1] in sampled_ids}
    logger.info(
        f"Sampled {len(sampled)} of {len(entities)} records (budget {count}, seed {seed}) "
        f"retaining {len(kept)} of {len(ground_truth)} gold pairs"
    )
    if ground_truth and not kept:
        logger.warning("Sample retained no gold pairs and cannot be scored")
    if count_gold_pairs(sampled, ground_truth) != len(kept):
        logger.warning("Gold pair accounting disagrees between the sample and the ground truth")
    return RecordSample(
        records=sampled,
        ground_truth=kept,
        requested=count,
        total=len(entities),
    )
