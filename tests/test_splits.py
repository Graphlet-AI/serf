"""Tests for random match-group train/val/holdout sampling."""

import pytest

from serf.config import config
from serf.dspy.types import Entity
from serf.eval.sample import sample_records
from serf.eval.splits import (
    EVAL_SPLIT_ALL,
    EVAL_SPLIT_HOLDOUT,
    EVAL_SPLIT_TRAIN,
    EVAL_SPLIT_VAL,
    SplitSizes,
    count_gold_pairs,
    get_all_split_sizes,
    get_split_sizes,
    match_groups,
    sample_random_splits,
    select_eval_split,
    training_overlap,
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


def test_the_default_benchmark_sample_is_entirely_training_data() -> None:
    """A plain sample at the default seed and size is the validation split itself.

    `sample_records` and `sample_random_splits` shuffle the same list with the
    same seed and walk it in order, and splits fill validation first, so a
    1000-record sample draws exactly the records GEPA selected its prompt on.
    """
    entities = _entities(4000)
    ground_truth = {(i, i + 1) for i in range(0, 4000, 2)}
    val_budget = get_split_sizes("dblp-acm").val_records

    sample = sample_records(entities, ground_truth, val_budget, seed=42)

    assert training_overlap("dblp-acm", sample.records, entities, ground_truth, seed=42) == 1.0


def test_records_outside_train_and_val_report_no_overlap() -> None:
    """The holdout split is the part training never saw, so it reports 0.0."""
    entities = _entities(4000)
    ground_truth = {(i, i + 1) for i in range(0, 4000, 2)}
    sizes = get_split_sizes("dblp-acm")
    splits = sample_random_splits(
        entities,
        ground_truth,
        train_records=sizes.train_records,
        val_records=sizes.val_records,
        holdout_records=sizes.holdout_records,
        seed=42,
    )

    overlap = training_overlap("dblp-acm", splits.holdout_records, entities, ground_truth, seed=42)

    assert splits.holdout_records
    assert overlap == 0.0


def test_training_overlap_of_nothing_is_zero() -> None:
    """An empty scored set cannot be contaminated."""
    assert training_overlap("dblp-acm", [], _entities(100), set(), seed=42) == 0.0


def test_split_sizes_are_1k_train_200_val_1k_holdout_per_dataset() -> None:
    """Train and val sit at roughly 80/20, with an equally large holdout GEPA never sees."""
    expected = get_all_split_sizes()
    assert set(expected) == {
        "walmart-amazon",
        "abt-buy",
        "amazon-google",
        "dblp-acm",
        "dblp-scholar",
    }
    for name, sizes in expected.items():
        assert sizes.train_records == 1000, name
        assert sizes.val_records == 200, name
        assert sizes.holdout_records == 1000, name
        assert get_split_sizes(name) == sizes
    assert config.get("benchmarks.train_records") == 1000
    assert config.get("benchmarks.val_records") == 200
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


def test_a_small_dataset_shorts_train_and_never_evaluation() -> None:
    """abt-buy has 2173 records, so val and holdout are paid in full and train takes the rest."""
    entities = _entities(2173)
    splits = sample_random_splits(
        entities,
        ground_truth=set(),
        train_records=1000,
        val_records=200,
        holdout_records=1000,
        seed=3,
    )
    assert len(splits.val_records) == 200
    assert len(splits.holdout_records) == 1000
    assert len(splits.train_records) == 973
    total = len(splits.train_records) + len(splits.val_records) + len(splits.holdout_records)
    assert total <= len(entities)


def test_an_oversized_train_budget_cannot_starve_evaluation() -> None:
    """The failure that once left GEPA one validation record, now impossible by construction."""
    entities = _entities(4910)
    splits = sample_random_splits(
        entities,
        ground_truth=set(),
        train_records=10**9,
        val_records=200,
        holdout_records=200,
        seed=5,
    )
    assert len(splits.val_records) == 200
    assert len(splits.holdout_records) == 200
    assert len(splits.train_records) == 4510


def test_sample_random_splits_fills_val_before_holdout_and_train() -> None:
    """A dataset too small for any budget spends itself on val, then holdout, then train."""
    entities = _entities(40)
    splits = sample_random_splits(
        entities,
        ground_truth=set(),
        train_records=1000,
        val_records=25,
        holdout_records=10,
        seed=4,
    )
    assert len(splits.val_records) == 25
    assert len(splits.holdout_records) == 10
    assert len(splits.train_records) == 5


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
    assert len(splits.train_records) == 1000
    assert len(splits.val_records) == 200
    assert len(splits.holdout_records) == 1000


def test_the_eval_split_defaults_to_the_one_training_never_reads() -> None:
    """Separation by construction: the default is holdout, not whatever is convenient."""
    assert config.get("benchmarks.eval_split") == EVAL_SPLIT_HOLDOUT
    assert config.get("benchmarks.require_disjoint_eval") is True


def test_selecting_the_holdout_gives_records_training_never_sees() -> None:
    entities = _entities(4000)
    ground_truth = {(i, i + 2000) for i in range(400)}
    sizes = SplitSizes(train_records=1000, val_records=200, holdout_records=1000)

    selection = select_eval_split(
        "dblp-acm", entities, ground_truth, split=EVAL_SPLIT_HOLDOUT, seed=11, sizes=sizes
    )
    splits = sample_random_splits(
        entities,
        ground_truth,
        train_records=1000,
        val_records=200,
        holdout_records=1000,
        seed=11,
    )

    scored = {entity.id for entity in selection.records}
    trained = {e.id for e in splits.train_records} | {e.id for e in splits.val_records}
    assert scored & trained == set()
    assert selection.overlaps_training is False


def test_the_holdout_selected_for_evaluation_is_the_one_training_reserved() -> None:
    """Both sides draw the same partition from the same budgets and seed."""
    entities = _entities(4000)
    sizes = SplitSizes(train_records=1000, val_records=200, holdout_records=1000)

    selection = select_eval_split(
        "dblp-acm", entities, set(), split=EVAL_SPLIT_HOLDOUT, seed=11, sizes=sizes
    )
    splits = sample_random_splits(
        entities, set(), train_records=1000, val_records=200, holdout_records=1000, seed=11
    )

    assert [e.id for e in selection.records] == sorted(e.id for e in splits.holdout_records)


def test_ground_truth_is_restricted_to_pairs_inside_the_split() -> None:
    """Scoring a pair whose partner is in another split would count a guaranteed miss."""
    entities = _entities(4000)
    ground_truth = {(i, i + 2000) for i in range(400)}
    sizes = SplitSizes(train_records=1000, val_records=200, holdout_records=1000)

    selection = select_eval_split(
        "dblp-acm", entities, ground_truth, split=EVAL_SPLIT_HOLDOUT, seed=11, sizes=sizes
    )

    ids = {entity.id for entity in selection.records}
    assert selection.gold_pairs == len(selection.ground_truth)
    assert all(left in ids and right in ids for left, right in selection.ground_truth)


def test_the_fitted_splits_declare_that_training_saw_them() -> None:
    entities = _entities(4000)
    sizes = SplitSizes(train_records=1000, val_records=200, holdout_records=1000)

    for split in (EVAL_SPLIT_TRAIN, EVAL_SPLIT_VAL, EVAL_SPLIT_ALL):
        selection = select_eval_split(
            "dblp-acm", entities, set(), split=split, seed=11, sizes=sizes
        )
        assert selection.overlaps_training is True, split


def test_selecting_everything_returns_the_whole_dataset() -> None:
    entities = _entities(100)

    selection = select_eval_split("dblp-acm", entities, set(), split=EVAL_SPLIT_ALL, seed=1)

    assert len(selection.records) == 100
    assert selection.total == 100


def test_an_unknown_split_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown eval split"):
        select_eval_split("dblp-acm", _entities(10), set(), split="nonsense")


def test_the_three_splits_partition_the_records_they_cover() -> None:
    """No record is scored by one split and trained on by another."""
    entities = _entities(4000)
    sizes = SplitSizes(train_records=1000, val_records=200, holdout_records=1000)
    chosen = {
        split: {
            e.id
            for e in select_eval_split(
                "dblp-acm", entities, set(), split=split, seed=11, sizes=sizes
            ).records
        }
        for split in (EVAL_SPLIT_TRAIN, EVAL_SPLIT_VAL, EVAL_SPLIT_HOLDOUT)
    }

    assert chosen[EVAL_SPLIT_TRAIN] & chosen[EVAL_SPLIT_VAL] == set()
    assert chosen[EVAL_SPLIT_TRAIN] & chosen[EVAL_SPLIT_HOLDOUT] == set()
    assert chosen[EVAL_SPLIT_VAL] & chosen[EVAL_SPLIT_HOLDOUT] == set()
