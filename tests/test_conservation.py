"""Tests for proving resolution lost nothing, and recovering it when it did."""

from typing import Any

import pytest

from serf.dspy.types import Entity
from serf.merge.canonical import CanonicalRecord
from serf.merge.conservation import (
    MISSING_IN_OUTPUT_REASON,
    check_conservation,
    coverage_of,
    recover_dropped_records,
)

A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
C = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
C_ABSORBED_1 = "c1111111-cccc-4ccc-8ccc-cccccccccccc"
C_ABSORBED_2 = "c2222222-cccc-4ccc-8ccc-cccccccccccc"
D = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
MERGED = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
OTHER = "ffffffff-ffff-4fff-8fff-ffffffffffff"
GHOST = "99999999-9999-4999-8999-999999999999"


def _inputs() -> list[dict[str, Any]]:
    return [
        {"uuid": A},
        {"uuid": B},
        {"uuid": C, "source_uuids": [C_ABSORBED_1, C_ABSORBED_2]},
        {"uuid": D},
    ]


def _all_but_b() -> list[str]:
    return [A, C, C_ABSORBED_1, C_ABSORBED_2, D]


def test_a_merge_that_keeps_everything_passes() -> None:
    outputs = [CanonicalRecord(uuid=MERGED, source_uuids=_all_but_b()), CanonicalRecord(uuid=B)]

    report = check_conservation(_inputs(), outputs)

    assert report.coverage_pct == 100.0
    assert report.passes
    assert report.missing_ids == []


def test_a_dropped_record_is_reported_as_missing() -> None:
    outputs = [CanonicalRecord(uuid=MERGED, source_uuids=_all_but_b())]

    report = check_conservation(_inputs(), outputs)

    assert report.missing_ids == [B]
    assert not report.passes
    assert not report.coverage_passes


def test_lineage_counts_as_coverage_even_though_the_record_is_gone() -> None:
    """An absorbed record has no output of its own; being inside source_uuids is enough."""
    outputs = [CanonicalRecord(uuid=MERGED, source_uuids=_all_but_b()), CanonicalRecord(uuid=B)]

    report = check_conservation(_inputs(), outputs)

    assert C_ABSORBED_1 in coverage_of(outputs[0])
    assert report.covered_ids == report.input_records


def test_a_source_id_pointing_at_nothing_is_a_dangling_reference() -> None:
    outputs = [
        CanonicalRecord(uuid=MERGED, source_uuids=[A, B, C, C_ABSORBED_1, C_ABSORBED_2, D, GHOST])
    ]

    report = check_conservation(_inputs(), outputs)

    assert report.invalid_reference_ids == [GHOST]
    assert not report.references_pass


def test_a_record_claimed_by_two_entities_fails() -> None:
    """One input record cannot belong to two resolved entities."""
    outputs = [
        CanonicalRecord(uuid=MERGED, source_uuids=[A, B]),
        CanonicalRecord(uuid=OTHER, source_uuids=[B, C]),
    ]

    report = check_conservation([{"uuid": A}, {"uuid": B}, {"uuid": C}], outputs)

    assert report.duplicated_ids == [B]
    assert not report.passes


def test_recovery_puts_the_dropped_record_back() -> None:
    """The anti-join Abzu's evaluation stops short of."""
    inputs = _inputs()
    outputs: list[Any] = [CanonicalRecord(uuid=MERGED, source_uuids=_all_but_b())]

    repaired, report = recover_dropped_records(inputs, outputs)

    assert len(repaired) == 2
    assert report.coverage_pct == 100.0
    assert report.passes
    assert report.recovered_records == 1
    assert report.missing_ids == []


def test_a_recovered_record_says_it_was_recovered() -> None:
    """A reader has to be able to tell a recovered record from a resolved one."""
    inputs = _inputs()
    outputs: list[Any] = [CanonicalRecord(uuid=MERGED, source_uuids=_all_but_b())]

    repaired, _ = recover_dropped_records(inputs, outputs)

    assert repaired[-1]["match_skip_reason"] == MISSING_IN_OUTPUT_REASON


def test_recovery_of_an_entity_marks_the_skip_reason_on_the_model() -> None:
    inputs = [Entity(id=1, uuid=A, name="a"), Entity(id=2, uuid=B, name="b")]
    outputs: list[Any] = [Entity(id=3, uuid=MERGED, name="a", source_uuids=[A])]

    repaired, report = recover_dropped_records(inputs, outputs)

    assert report.passes
    assert repaired[-1].match_skip_reason == MISSING_IN_OUTPUT_REASON
    assert repaired[-1].uuid == B


def test_recovery_leaves_a_complete_output_alone() -> None:
    inputs = _inputs()
    outputs: list[Any] = [
        CanonicalRecord(uuid=MERGED, source_uuids=_all_but_b()),
        CanonicalRecord(uuid=B),
    ]

    repaired, report = recover_dropped_records(inputs, outputs)

    assert repaired is outputs
    assert report.recovered_records == 0


def test_recovering_one_record_can_cover_several_missing_ids() -> None:
    """Adding record 3 back accounts for 5 and 6 as well, so it is added once."""
    inputs = _inputs()
    outputs: list[Any] = [
        CanonicalRecord(uuid=A),
        CanonicalRecord(uuid=B),
        CanonicalRecord(uuid=D),
    ]

    repaired, report = recover_dropped_records(inputs, outputs)

    assert report.recovered_records == 1
    assert report.passes
    assert len(repaired) == 4


def test_recovery_that_cannot_fix_the_data_raises() -> None:
    """Data integrity outranks finishing the run."""
    inputs = [{"uuid": A}, {"uuid": B}]
    outputs: list[Any] = [
        CanonicalRecord(uuid=MERGED, source_uuids=[A, B]),
        CanonicalRecord(uuid=OTHER, source_uuids=[B]),
    ]

    with pytest.raises(ValueError, match="Conservation failed after recovery"):
        recover_dropped_records(inputs, outputs)


def test_an_empty_run_conserves_trivially() -> None:
    report = check_conservation([], [])

    assert report.passes
    assert report.coverage_pct == 100.0


def test_the_summary_names_the_verdict() -> None:
    report = check_conservation(
        _inputs(), [CanonicalRecord(uuid=MERGED, source_uuids=_all_but_b())]
    )

    assert report.summary().startswith("FAIL")
    assert "input records reachable" in report.summary()


def test_coverage_of_accepts_both_records_and_mappings() -> None:
    expected = {C, C_ABSORBED_1, C_ABSORBED_2}
    assert coverage_of({"uuid": C, "source_uuids": [C_ABSORBED_1, C_ABSORBED_2]}) == expected
    assert (
        coverage_of(CanonicalRecord(uuid=C, source_uuids=[C_ABSORBED_1, C_ABSORBED_2])) == expected
    )
    assert (
        coverage_of(Entity(id=3, uuid=C, name="x", source_uuids=[C_ABSORBED_1, C_ABSORBED_2]))
        == expected
    )


def test_an_intermediate_entitys_minted_id_is_real_lineage_not_a_dangling_reference() -> None:
    """The bug the abt-buy run found: a two-round merge leaves a minted id in source_ids."""
    inputs = [{"uuid": A}, {"uuid": B}, {"uuid": C}]
    # Round one merged A and B into minted entity MERGED; round two merged that
    # with C into minted entity OTHER, whose lineage names MERGED.
    outputs: list[Any] = [CanonicalRecord(uuid=OTHER, source_uuids=[A, B, C, MERGED])]

    unaware = check_conservation(inputs, outputs)
    aware = check_conservation(inputs, outputs, minted_uuids={MERGED, OTHER})

    assert unaware.invalid_reference_ids == [MERGED]
    assert not unaware.passes
    assert aware.invalid_reference_ids == []
    assert aware.passes


def test_recovery_also_takes_the_minted_ids_into_account() -> None:
    inputs = [{"uuid": A}, {"uuid": B}, {"uuid": C}]
    outputs: list[Any] = [CanonicalRecord(uuid=OTHER, source_uuids=[A, B, MERGED])]

    repaired, report = recover_dropped_records(inputs, outputs, minted_uuids={MERGED, OTHER})

    assert report.passes
    assert report.recovered_records == 1
    assert len(repaired) == 2


def test_a_reference_to_an_id_nobody_ever_issued_still_dangles() -> None:
    """The check must stay able to catch a genuinely invented reference."""
    inputs = [{"uuid": A}, {"uuid": B}]
    outputs: list[Any] = [CanonicalRecord(uuid=OTHER, source_uuids=[A, B, GHOST])]

    report = check_conservation(inputs, outputs, minted_uuids={OTHER})

    assert report.invalid_reference_ids == [GHOST]
    assert not report.passes
