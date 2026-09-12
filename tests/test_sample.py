"""Tests for match-group-aware benchmark record sampling."""

from serf.dspy.types import Entity
from serf.eval.sample import sample_records

RIGHT_ID_OFFSET = 100000


def _dataset(
    left_count: int, right_count: int, pair_count: int
) -> tuple[list[Entity], set[tuple[int, int]]]:
    """Build a synthetic two-source dataset with ground-truth pairs.

    Parameters
    ----------
    left_count : int
        Records on the left side
    right_count : int
        Records on the right side
    pair_count : int
        Number of ground-truth pairs, matching left id i to right id i

    Returns
    -------
    tuple[list[Entity], set[tuple[int, int]]]
        Records and ground-truth pairs
    """
    left = [Entity(id=i, name=f"left {i}", attributes={"l_id": str(i)}) for i in range(left_count)]
    right = [
        Entity(id=RIGHT_ID_OFFSET + i, name=f"right {i}", attributes={"r_id": str(i)})
        for i in range(right_count)
    ]
    pairs = {(i, RIGHT_ID_OFFSET + i) for i in range(pair_count)}
    return left + right, pairs


def test_sample_keeps_match_groups_whole() -> None:
    """Every gold partner of a sampled record is sampled too."""
    entities, ground_truth = _dataset(500, 500, 200)

    sample = sample_records(entities, ground_truth, 100, seed=42)

    ids = {record.id for record in sample.records}
    for left, right in ground_truth:
        assert (left in ids) == (right in ids)


def test_sample_retains_far_more_pairs_than_independent_sampling() -> None:
    """Group-aware sampling beats the square-law retention of naive sampling."""
    entities, ground_truth = _dataset(500, 500, 200)

    sample = sample_records(entities, ground_truth, 100, seed=42)

    naive_expectation = len(ground_truth) * (100 / len(entities)) ** 2
    assert sample.gold_pairs > 10 * naive_expectation


def test_sample_respects_the_record_budget() -> None:
    """The sample overshoots the budget by at most one match group."""
    entities, ground_truth = _dataset(500, 500, 200)

    sample = sample_records(entities, ground_truth, 100, seed=42)

    assert 100 <= len(sample.records) <= 102
    assert sample.requested == 100
    assert sample.total == 1000


def test_sample_is_deterministic_for_a_seed() -> None:
    """The same seed draws the same records, so both A/B arms see one sample."""
    entities, ground_truth = _dataset(300, 300, 100)

    first = sample_records(entities, ground_truth, 80, seed=42)
    second = sample_records(entities, ground_truth, 80, seed=42)
    other = sample_records(entities, ground_truth, 80, seed=7)

    assert [r.id for r in first.records] == [r.id for r in second.records]
    assert [r.id for r in first.records] != [r.id for r in other.records]


def test_sample_ground_truth_is_restricted_to_the_sample() -> None:
    """Metrics must score against the pairs that survived sampling."""
    entities, ground_truth = _dataset(300, 300, 100)

    sample = sample_records(entities, ground_truth, 80, seed=42)

    ids = {record.id for record in sample.records}
    assert sample.ground_truth
    assert sample.ground_truth <= ground_truth
    for left, right in sample.ground_truth:
        assert left in ids and right in ids
    assert sample.gold_pairs == len(sample.ground_truth)


def test_sample_larger_than_the_dataset_returns_everything() -> None:
    """A budget at or above the dataset size keeps all records and pairs."""
    entities, ground_truth = _dataset(50, 50, 20)

    sample = sample_records(entities, ground_truth, 5000, seed=42)

    assert len(sample.records) == len(entities)
    assert sample.ground_truth == ground_truth


def test_sample_records_are_sorted_by_id() -> None:
    """Sorted output makes the sampled block input order reproducible."""
    entities, ground_truth = _dataset(200, 200, 50)

    sample = sample_records(entities, ground_truth, 60, seed=42)

    assert [r.id for r in sample.records] == sorted(r.id for r in sample.records)


def test_sample_without_ground_truth_still_draws_records() -> None:
    """A dataset with no pairs samples plain records without failing."""
    entities, _ = _dataset(100, 100, 0)

    sample = sample_records(entities, set(), 40, seed=42)

    assert len(sample.records) == 40
    assert sample.gold_pairs == 0
