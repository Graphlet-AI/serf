"""Tests for canonical records: minted uuids, lineage, and array-valued fields."""

from typing import Any

import pytest

from serf.merge.canonical import (
    CanonicalRecord,
    canonicalize,
    canonicalize_groups,
    known_uuids,
    mint_uuid,
)

# The worked example from the specification, with uuids standing in for the
# integers it was written with. Integer identity lives only inside UUIDMapper.
RUSSELL = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"
RUSSELL_H = "33333333-3333-4333-8333-333333333333"
ABSORBED_A = "55555555-5555-4555-8555-555555555555"
ABSORBED_B = "66666666-6666-4666-8666-666666666666"
RUSS = "77777777-7777-4777-8777-777777777777"

SPEC_RECORDS: list[dict[str, Any]] = [
    {"uuid": RUSSELL, "name": "Russell Jurney"},
    {"uuid": BOB, "name": "Bob Dorf"},
    {
        "uuid": RUSSELL_H,
        "name": "Russell H Jurney",
        "source_uuids": [ABSORBED_A, ABSORBED_B],
        "state": "WA",
        "nation": "US",
    },
    {"uuid": RUSS, "name": "Russ Journey", "state": "CA"},
]


def _by_uuid() -> dict[str, dict[str, Any]]:
    return {record["uuid"]: record for record in SPEC_RECORDS}


def test_the_worked_example_from_the_specification() -> None:
    """The documented input must produce the documented output, field for field."""
    records = _by_uuid()
    groups = [[records[RUSSELL], records[RUSSELL_H], records[RUSS]], [records[BOB]]]

    merged, untouched = canonicalize_groups(groups)

    rendered = merged.to_dict()
    assert rendered["name"] == ["Russell H Jurney"]
    assert rendered["state"] == ["CA", "WA"]
    assert rendered["nation"] == ["US"]
    assert rendered["source_uuids"] == sorted([RUSSELL, RUSSELL_H, ABSORBED_A, ABSORBED_B, RUSS])
    assert untouched.to_dict() == {"uuid": BOB, "name": ["Bob Dorf"]}


def test_a_merge_mints_an_identity_no_input_holds() -> None:
    """A merge is a new entity, so it cannot answer to one of its inputs' uuids."""
    records = _by_uuid()

    merged = canonicalize_groups([[records[RUSSELL], records[RUSSELL_H], records[RUSS]]])[0]

    assert merged.uuid not in {RUSSELL, BOB, RUSSELL_H, ABSORBED_A, ABSORBED_B, RUSS}


def test_a_minted_identity_cannot_collide_with_a_record_outside_the_group() -> None:
    """Why uuids: an integer minted from one block's ids can hit another block's record."""
    records = _by_uuid()
    group = [records[RUSSELL], records[RUSSELL_H], records[RUSS]]

    first = canonicalize_groups([group])[0]
    second = canonicalize_groups([group])[0]

    assert first.uuid != second.uuid
    assert BOB not in {first.uuid, second.uuid}


def test_lineage_absorbs_both_the_matched_uuids_and_their_own_lineage() -> None:
    """Russell H already stood for two records, so the merge has to stand for them too."""
    records = _by_uuid()

    merged = canonicalize_groups([[records[RUSSELL], records[RUSSELL_H], records[RUSS]]])[0]

    assert merged.source_uuids == sorted([RUSSELL, RUSSELL_H, ABSORBED_A, ABSORBED_B, RUSS])
    assert merged.covers() == {
        merged.uuid,
        RUSSELL,
        RUSSELL_H,
        ABSORBED_A,
        ABSORBED_B,
        RUSS,
    }


def test_an_unmatched_record_keeps_its_uuid_and_gains_no_lineage() -> None:
    """Nothing merged, so there is no new entity and nothing to record."""
    untouched = canonicalize_groups([[_by_uuid()[BOB]]])[0]

    assert untouched.uuid == BOB
    assert untouched.source_uuids == []


def test_a_group_of_one_that_already_had_lineage_keeps_it() -> None:
    alone = canonicalize_groups([[_by_uuid()[RUSSELL_H]]])[0]

    assert alone.uuid == RUSSELL_H
    assert alone.source_uuids == sorted([ABSORBED_A, ABSORBED_B])


def test_every_field_of_every_input_survives_into_the_output() -> None:
    """Rule 2: the merged record carries all the fields of its inputs."""
    records = _by_uuid()
    groups = [[records[RUSSELL], records[RUSSELL_H], records[RUSS]]]

    merged = canonicalize_groups(groups)[0]

    assert set(merged.fields) == {"name", "state", "nation"}


def test_every_field_except_the_identity_is_a_list() -> None:
    """Rule 3, including the single-valued fields that read like scalars."""
    records = _by_uuid()
    merged = canonicalize_groups([[records[RUSSELL], records[RUSSELL_H], records[RUSS]]])[0]

    rendered = merged.to_dict()
    assert isinstance(rendered["uuid"], str)
    assert all(isinstance(value, list) for key, value in rendered.items() if key != "uuid")


def test_values_already_held_as_lists_are_absorbed_not_nested() -> None:
    """A merged record is itself a valid input, so merging has to be idempotent in shape."""
    groups = [[{"uuid": RUSSELL, "state": ["CA", "WA"]}, {"uuid": BOB, "state": ["OR"]}]]

    merged = canonicalize_groups(groups)[0]

    assert merged.fields["state"] == ["CA", "OR", "WA"]


def test_field_order_follows_first_appearance_across_the_inputs() -> None:
    """Readable output beats alphabetised output, and the order must be deterministic."""
    groups = [[{"uuid": RUSSELL, "zeta": "a"}, {"uuid": BOB, "alpha": "b"}]]

    merged = canonicalize_groups(groups)[0]

    assert list(merged.to_dict()) == ["uuid", "zeta", "alpha", "source_uuids"]


def test_canonicalizing_nothing_is_an_error_rather_than_an_empty_record() -> None:
    """An empty group means the caller lost records, which must not pass silently."""
    with pytest.raises(ValueError, match="empty group"):
        canonicalize([])


def test_known_uuids_counts_lineage_as_used() -> None:
    assert known_uuids(SPEC_RECORDS) == {
        RUSSELL,
        BOB,
        RUSSELL_H,
        ABSORBED_A,
        ABSORBED_B,
        RUSS,
    }


def test_a_field_type_override_beats_detection() -> None:
    """A schema knows what detection can only guess from values."""
    groups = [[{"uuid": RUSSELL, "blob": "Main Street"}, {"uuid": BOB, "blob": "Main St"}]]

    detected = canonicalize_groups(groups)[0]
    overridden = canonicalize_groups(groups, field_types={"blob": "address"})[0]

    assert overridden.fields["blob"] == ["Main Street"]
    assert len(detected.fields["blob"]) >= len(overridden.fields["blob"])


def test_blank_values_never_reach_the_output() -> None:
    groups = [[{"uuid": RUSSELL, "name": "Russell", "note": ""}, {"uuid": BOB, "name": None}]]

    merged = canonicalize_groups(groups)[0]

    assert merged.fields == {"name": ["Russell"]}


def test_covers_includes_the_records_own_identity() -> None:
    record = CanonicalRecord(uuid=RUSSELL, source_uuids=[ABSORBED_A, ABSORBED_B])

    assert record.covers() == {RUSSELL, ABSORBED_A, ABSORBED_B}


def test_minted_identities_are_distinct() -> None:
    assert len({mint_uuid() for _ in range(100)}) == 100
