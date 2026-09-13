"""Tests for canonical records: minted ids, lineage, and array-valued fields."""

from typing import Any

import pytest

from serf.merge.canonical import (
    STRATEGY_SEQUENTIAL,
    STRATEGY_SMALLEST_UNUSED,
    CanonicalRecord,
    IdAllocator,
    canonicalize,
    canonicalize_groups,
    known_ids,
)

SPEC_RECORDS: list[dict[str, Any]] = [
    {"id": 1, "name": "Russell Jurney"},
    {"id": 2, "name": "Bob Dorf"},
    {"id": 3, "name": "Russell H Jurney", "source_ids": [5, 6], "state": "WA", "nation": "US"},
    {"id": 7, "name": "Russ Journey", "state": "CA"},
]


def _by_id() -> dict[int, dict[str, Any]]:
    return {record["id"]: record for record in SPEC_RECORDS}


def test_the_worked_example_from_the_specification() -> None:
    """The documented input must produce the documented output, field for field."""
    records = _by_id()
    groups = [[records[1], records[3], records[7]], [records[2]]]

    merged, untouched = canonicalize_groups(groups)

    assert merged.to_dict() == {
        "id": 4,
        "name": ["Russell H Jurney"],
        "state": ["CA", "WA"],
        "nation": ["US"],
        "source_ids": [1, 3, 5, 6, 7],
    }
    assert untouched.to_dict() == {"id": 2, "name": ["Bob Dorf"]}


def test_a_merge_mints_an_id_no_record_and_no_lineage_entry_claims() -> None:
    """Reusing an id that appears in some source_ids would make lineage ambiguous."""
    records = _by_id()

    merged = canonicalize_groups(
        [[records[1], records[3], records[7]]], reserved=known_ids(SPEC_RECORDS)
    )[0]

    assert merged.id not in {1, 2, 3, 5, 6, 7}


def test_minting_from_a_slice_alone_can_collide_with_a_record_outside_it() -> None:
    """Why `reserved` exists: a block does not know the ids of the blocks beside it."""
    records = _by_id()
    group = [records[1], records[3], records[7]]

    unaware = canonicalize_groups([group])[0]
    aware = canonicalize_groups([group], reserved=known_ids(SPEC_RECORDS))[0]

    assert unaware.id == 2
    assert aware.id == 4


def test_lineage_absorbs_both_the_matched_ids_and_their_own_source_ids() -> None:
    """Record 3 already stood for 5 and 6, so the merge has to stand for them too."""
    records = _by_id()

    merged = canonicalize_groups(
        [[records[1], records[3], records[7]]], reserved=known_ids(SPEC_RECORDS)
    )[0]

    assert merged.source_ids == [1, 3, 5, 6, 7]
    assert merged.covers() == {1, 3, 4, 5, 6, 7}


def test_an_unmatched_record_keeps_its_id_and_gains_no_lineage() -> None:
    """Nothing merged, so there is no new entity and nothing to record."""
    untouched = canonicalize_groups([[_by_id()[2]]])[0]

    assert untouched.id == 2
    assert untouched.source_ids == []


def test_a_group_of_one_that_already_had_lineage_keeps_it() -> None:
    """A record standing for 5 and 6 still stands for them when it matches nothing."""
    alone = canonicalize_groups([[_by_id()[3]]])[0]

    assert alone.id == 3
    assert alone.source_ids == [5, 6]


def test_every_field_of_every_input_survives_into_the_output() -> None:
    """Rule 2: the merged record carries all the fields of its inputs."""
    groups = [[_by_id()[1], _by_id()[3], _by_id()[7]]]

    merged = canonicalize_groups(groups)[0]

    assert set(merged.fields) == {"name", "state", "nation"}


def test_every_field_except_id_is_a_list() -> None:
    """Rule 3, including the single-valued fields that read like scalars."""
    merged = canonicalize_groups([[_by_id()[1], _by_id()[3], _by_id()[7]]])[0]

    rendered = merged.to_dict()
    assert isinstance(rendered["id"], int)
    assert all(isinstance(value, list) for key, value in rendered.items() if key != "id")


def test_values_already_held_as_lists_are_absorbed_not_nested() -> None:
    """A merged record is itself a valid input, so merging has to be idempotent in shape."""
    groups = [[{"id": 1, "state": ["CA", "WA"]}, {"id": 2, "state": ["OR"]}]]

    merged = canonicalize_groups(groups)[0]

    assert merged.fields["state"] == ["CA", "OR", "WA"]


def test_field_order_follows_first_appearance_across_the_inputs() -> None:
    """Readable output beats alphabetised output, and the order must be deterministic."""
    groups = [[{"id": 1, "zeta": "a"}, {"id": 2, "alpha": "b"}]]

    merged = canonicalize_groups(groups)[0]

    assert list(merged.to_dict()) == ["id", "zeta", "alpha", "source_ids"]


def test_canonicalizing_nothing_is_an_error_rather_than_an_empty_record() -> None:
    """An empty group means the caller lost records, which must not pass silently."""
    with pytest.raises(ValueError, match="empty group"):
        canonicalize([], IdAllocator([]))


def test_smallest_unused_fills_the_gaps_in_the_id_space() -> None:
    allocator = IdAllocator({1, 2, 3, 5, 6, 7}, strategy=STRATEGY_SMALLEST_UNUSED)

    assert [allocator.mint() for _ in range(3)] == [4, 8, 9]


def test_sequential_never_reuses_an_id_below_the_high_water_mark() -> None:
    allocator = IdAllocator({1, 2, 3, 5, 6, 7}, strategy=STRATEGY_SEQUENTIAL)

    assert [allocator.mint() for _ in range(3)] == [8, 9, 10]


def test_an_unknown_strategy_falls_back_rather_than_raising() -> None:
    """A typo in config must not take down a pipeline mid-run."""
    allocator = IdAllocator({4}, strategy="nonsense")

    assert allocator.strategy == STRATEGY_SEQUENTIAL


def test_one_allocator_across_groups_cannot_mint_the_same_id_twice() -> None:
    groups = [
        [{"id": 10, "name": "a"}, {"id": 11, "name": "a"}],
        [{"id": 12, "name": "b"}, {"id": 13, "name": "b"}],
    ]

    minted = [record.id for record in canonicalize_groups(groups)]

    assert len(set(minted)) == len(minted)


def test_known_ids_counts_lineage_as_used() -> None:
    assert known_ids(SPEC_RECORDS) == {1, 2, 3, 5, 6, 7}


def test_a_field_type_override_beats_detection() -> None:
    """A schema knows what detection can only guess from values."""
    groups = [[{"id": 1, "blob": "Main Street"}, {"id": 2, "blob": "Main St"}]]

    detected = canonicalize_groups(groups)[0]
    overridden = canonicalize_groups(groups, field_types={"blob": "address"})[0]

    assert overridden.fields["blob"] == ["Main Street"]
    assert len(detected.fields["blob"]) >= len(overridden.fields["blob"])


def test_blank_values_never_reach_the_output() -> None:
    groups = [[{"id": 1, "name": "Russell", "note": ""}, {"id": 2, "name": None}]]

    merged = canonicalize_groups(groups)[0]

    assert merged.fields == {"name": ["Russell"]}


def test_covers_includes_the_records_own_id() -> None:
    record = CanonicalRecord(id=4, source_ids=[1, 3])

    assert record.covers() == {1, 3, 4}
