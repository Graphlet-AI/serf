"""Tests for repairing a model's answer into a partition that loses nothing."""

import pytest

from serf.match.partition import build_partition, partition_from_pairs


def test_a_valid_answer_passes_through_unchanged() -> None:
    outcome = build_partition([[1, 3], [2]], known_ids={1, 2, 3})

    assert outcome.groups == [[1, 3], [2]]
    assert outcome.is_clean


def test_a_record_the_model_forgot_comes_back_as_unmatched() -> None:
    """The central guarantee: a record left out of every group is not deleted."""
    outcome = build_partition([[1, 3]], known_ids={1, 2, 3})

    assert outcome.groups == [[1, 3], [2]]
    assert outcome.recovered_ids == [2]
    assert not outcome.is_clean


def test_a_record_named_in_two_groups_stays_in_the_first() -> None:
    """A partition cannot hold a record twice, and guessing which one is meant loses data."""
    outcome = build_partition([[1, 2], [2, 3]], known_ids={1, 2, 3})

    assert outcome.groups == [[1, 2], [3]]
    assert outcome.duplicate_ids == [2]


def test_an_invented_record_id_is_discarded() -> None:
    outcome = build_partition([[1, 999]], known_ids={1, 2})

    assert outcome.groups == [[1], [2]]
    assert outcome.unknown_ids == [999]


def test_every_known_record_is_covered_exactly_once_whatever_the_model_returned() -> None:
    """The invariant that makes the rest of the pipeline safe."""
    outcome = build_partition([[1, 1, 2], [2, 3], [99], []], known_ids={1, 2, 3, 4, 5})

    assert outcome.covered() == {1, 2, 3, 4, 5}
    assert sum(len(group) for group in outcome.groups) == 5


def test_an_empty_answer_leaves_every_record_unmatched() -> None:
    """A failed call must not silently merge or silently drop anything."""
    outcome = build_partition([], known_ids={1, 2, 3})

    assert outcome.groups == [[1], [2], [3]]
    assert outcome.recovered_ids == [1, 2, 3]


def test_groups_are_ordered_and_sorted_so_output_is_reproducible() -> None:
    outcome = build_partition([[5, 4], [1, 3]], known_ids={1, 3, 4, 5})

    assert outcome.groups == [[1, 3], [4, 5]]


def test_pairs_come_from_the_grouping_not_from_what_the_model_listed() -> None:
    """Grouping a, b and c asserts all three pairs, including the one nobody wrote down."""
    outcome = build_partition([[1, 2, 3]], known_ids={1, 2, 3})

    assert outcome.pairs() == {(1, 2), (1, 3), (2, 3)}


def test_a_partition_of_singletons_asserts_no_pair() -> None:
    outcome = build_partition([[1], [2]], known_ids={1, 2})

    assert outcome.pairs() == set()
    assert outcome.matched_groups == []


def test_matched_groups_ignores_the_records_that_merged_with_nothing() -> None:
    outcome = build_partition([[1, 2], [3]], known_ids={1, 2, 3})

    assert outcome.matched_groups == [[1, 2]]


def test_pairs_recover_the_partition_they_came_from() -> None:
    """Round trip, because the gold standard is stated as pairs and the pipeline groups."""
    groups = [[1, 2, 3], [4], [5, 6]]
    pairs = build_partition(groups, known_ids={1, 2, 3, 4, 5, 6}).pairs()

    assert partition_from_pairs(pairs, known_ids={1, 2, 3, 4, 5, 6}) == groups


def test_pairs_chain_transitively_into_one_group() -> None:
    assert partition_from_pairs([(1, 2), (2, 3)], known_ids={1, 2, 3}) == [[1, 2, 3]]


def test_a_pair_naming_an_unknown_record_is_ignored() -> None:
    assert partition_from_pairs([(1, 99)], known_ids={1, 2}) == [[1], [2]]


def test_recovery_cannot_leave_a_gap() -> None:
    """The assertion guarding the invariant should never be reachable."""
    outcome = build_partition([[2]], known_ids={1, 2})

    assert outcome.covered() == {1, 2}


def test_a_block_of_one_record_partitions_into_one_group() -> None:
    outcome = build_partition([], known_ids={7})

    assert outcome.groups == [[7]]


@pytest.mark.parametrize(
    "raw_groups",
    [
        [[1, 2], [3]],
        [[3], [2, 1]],
        [[2, 1], [3]],
    ],
)
def test_the_same_partition_written_differently_gives_the_same_answer(
    raw_groups: list[list[int]],
) -> None:
    outcome = build_partition(raw_groups, known_ids={1, 2, 3})

    assert outcome.groups == [[1, 2], [3]]
