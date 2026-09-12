"""Tests for random match-group train/val/holdout sampling."""

import pytest

from serf.config import config
from serf.dspy.types import Entity
from serf.eval.splits import (
    count_gold_pairs,
    get_all_split_sizes,
    get_split_sizes,
    match_groups,
    sample_random_splits,
)


def _entity(entity_id: int) -> Entity:
    """Build a minimal entity."""
    return Entity(id=entity_id, name=f"e{entity_id}", description="", entity_type="entity")


def _entities(count: int) -> list[Entity]:
    """Build ``count`` entities with IDs 0..count-1."""
    return [_entity(i) for i in range(count)]


def _ids(records: list[Entity]) -> set[int]:
    """Collect entity IDs."""
    return {record.id for record in records}


def test_split_sizes_are_2k_train_records_per_dataset() -> None:
    """Every benchmark dataset asks for 2000 train, 1000 val, 1000 holdout records."""
    expected = get_all_split_sizes()
    assert set(expected) == {
        "walmart-amazon",
        "abt-buy",
        "amazon-google",
        "dblp-acm",
        "dblp-scholar",
    }
    for name, sizes in expected.items():
        assert sizes.train_records == 2000, name
        assert sizes.val_records == 1000, name
        assert sizes.holdout_records == 1000, name
        assert get_split_sizes(name) == sizes
    assert config.get("benchmarks.train_records") == 2000
    assert config.get("benchmarks.val_records") == 1000
    assert config.get("benchmarks.holdout_records") == 1000
    with pytest.raises(KeyError):
        config.get("benchmarks.train_blocks")


def test_match_groups_join_transitive_matches() -> None:
    """Records linked transitively by ground truth land in one group."""
    entities = _entities(6)
    groups = match_groups(entities, {(0, 1), (1, 2), (4, 5)})
    assert _ids(groups[0]) == {0, 1, 2}
    assert groups[0] is groups[1] is groups[2]
    assert _ids(groups[4]) == {4, 5}
    assert _ids(groups[3]) == {3}


def test_match_groups_ignore_pairs_outside_the_entity_list() -> None:
    """Ground-truth partners that are not in the entity list are dropped."""
    entities = _entities(3)
    groups = match_groups(entities, {(0, 99)})
    assert _ids(groups[0]) == {0}


def test_count_gold_pairs_counts_pairs_fully_inside_the_split() -> None:
    """Only pairs with both records present are counted."""
    records = [_entity(0), _entity(1), _entity(5)]
    assert count_gold_pairs(records, {(0, 1), (1, 5), (1, 9)}) == 2


def test_sample_random_splits_is_disjoint_and_sized() -> None:
    """Splits hit their record budgets and share no entity."""
    entities = _entities(1000)
    splits = sample_random_splits(
        entities,
        ground_truth=set(),
        train_records=200,
        val_records=100,
        holdout_records=100,
        seed=0,
    )
    assert len(splits.train_records) == 200
    assert len(splits.val_records) == 100
    assert len(splits.holdout_records) == 100

    train_ids = _ids(splits.train_records)
    val_ids = _ids(splits.val_records)
    holdout_ids = _ids(splits.holdout_records)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(holdout_ids)
    assert val_ids.isdisjoint(holdout_ids)


def test_sample_random_splits_keeps_match_groups_whole() -> None:
    """A record and everything transitively matched to it stay in one split."""
    entities = _entities(600)
    ground_truth = {(i, i + 300) for i in range(300)}
    splits = sample_random_splits(
        entities,
        ground_truth=ground_truth,
        train_records=200,
        val_records=100,
        holdout_records=100,
        seed=1,
    )
    for records in (splits.train_records, splits.val_records, splits.holdout_records):
        ids = _ids(records)
        for left, right in ground_truth:
            assert (left in ids) == (right in ids)


def test_sample_random_splits_preserves_gold_pairs_in_a_sparse_dataset() -> None:
    """The dblp-scholar shape: a tiny sample of a huge table still carries gold pairs."""
    entities = _entities(60000)
    ground_truth = {(i, i + 30000) for i in range(2500)}
    splits = sample_random_splits(
        entities,
        ground_truth=ground_truth,
        train_records=2000,
        val_records=1000,
        holdout_records=1000,
        seed=7,
    )
    assert count_gold_pairs(splits.val_records, ground_truth) > 0
    assert count_gold_pairs(splits.train_records, ground_truth) > 0
    # Independent uniform sampling would retain about 1 pair; match groups keep far more.
    assert count_gold_pairs(splits.val_records, ground_truth) >= 20


def test_sample_random_splits_scales_budgets_for_small_datasets() -> None:
    """abt-buy has 2173 records, so 2000/1000/1000 scales down keeping the 2:1:1 ratio."""
    entities = _entities(2173)
    splits = sample_random_splits(
        entities,
        ground_truth=set(),
        train_records=2000,
        val_records=1000,
        holdout_records=1000,
        seed=3,
    )
    assert len(splits.train_records) == 1086
    assert len(splits.val_records) == 543
    assert len(splits.holdout_records) == 543
    total = len(splits.train_records) + len(splits.val_records) + len(splits.holdout_records)
    assert total <= len(entities)


def test_sample_random_splits_fills_val_before_holdout_and_train() -> None:
    """A tiny dataset keeps the 2:1:1 ratio and still fills val first."""
    entities = _entities(40)
    splits = sample_random_splits(
        entities,
        ground_truth=set(),
        train_records=2000,
        val_records=1000,
        holdout_records=1000,
        seed=4,
    )
    assert len(splits.val_records) == 10
    assert len(splits.holdout_records) == 10
    assert len(splits.train_records) == 20


def test_sample_random_splits_fills_val_when_records_run_out() -> None:
    """With fewer records than the val budget, everything goes to val."""
    entities = _entities(12)
    splits = sample_random_splits(
        entities,
        ground_truth=set(),
        train_records=0,
        val_records=100,
        holdout_records=0,
        seed=4,
    )
    assert len(splits.val_records) == 12
    assert splits.holdout_records == []
    assert splits.train_records == []


def test_sample_random_splits_is_deterministic() -> None:
    """The same seed yields the same split."""
    entities = _entities(500)
    ground_truth = {(i, i + 250) for i in range(120)}
    a = sample_random_splits(
        entities,
        ground_truth=ground_truth,
        train_records=100,
        val_records=50,
        holdout_records=50,
        seed=11,
    )
    b = sample_random_splits(
        entities,
        ground_truth=ground_truth,
        train_records=100,
        val_records=50,
        holdout_records=50,
        seed=11,
    )
    assert [e.id for e in a.train_records] == [e.id for e in b.train_records]
    assert [e.id for e in a.val_records] == [e.id for e in b.val_records]
    assert [e.id for e in a.holdout_records] == [e.id for e in b.holdout_records]


def test_sample_random_splits_uses_config_defaults() -> None:
    """Omitted budgets fall back to the configured record counts."""
    entities = _entities(8000)
    splits = sample_random_splits(entities, ground_truth=set(), seed=2)
    assert len(splits.train_records) == 2000
    assert len(splits.val_records) == 1000
    assert len(splits.holdout_records) == 1000
