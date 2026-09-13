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


def _inputs() -> list[dict[str, Any]]:
    return [{"id": 1}, {"id": 2}, {"id": 3, "source_ids": [5, 6]}, {"id": 7}]


def test_a_merge_that_keeps_everything_passes() -> None:
    outputs = [CanonicalRecord(id=4, source_ids=[1, 3, 5, 6, 7]), CanonicalRecord(id=2)]

    report = check_conservation(_inputs(), outputs)

    assert report.coverage_pct == 100.0
    assert report.passes
    assert report.missing_ids == []


def test_a_dropped_record_is_reported_as_missing() -> None:
    outputs = [CanonicalRecord(id=4, source_ids=[1, 3, 5, 6, 7])]

    report = check_conservation(_inputs(), outputs)

    assert report.missing_ids == [2]
    assert not report.passes
    assert not report.coverage_passes


def test_lineage_counts_as_coverage_even_though_the_record_is_gone() -> None:
    """Record 5 has no output of its own; being inside source_ids is enough."""
    outputs = [CanonicalRecord(id=4, source_ids=[1, 3, 5, 6, 7]), CanonicalRecord(id=2)]

    report = check_conservation(_inputs(), outputs)

    assert 5 in coverage_of(outputs[0])
    assert report.covered_ids == report.input_records


def test_a_source_id_pointing_at_nothing_is_a_dangling_reference() -> None:
    outputs = [CanonicalRecord(id=4, source_ids=[1, 2, 3, 5, 6, 7, 9999])]

    report = check_conservation(_inputs(), outputs)

    assert report.invalid_reference_ids == [9999]
    assert not report.references_pass


def test_a_record_claimed_by_two_entities_fails() -> None:
    """One input record cannot belong to two resolved entities."""
    outputs = [CanonicalRecord(id=10, source_ids=[1, 2]), CanonicalRecord(id=11, source_ids=[2, 3])]

    report = check_conservation([{"id": 1}, {"id": 2}, {"id": 3}], outputs)

    assert report.duplicated_ids == [2]
    assert not report.passes


def test_recovery_puts_the_dropped_record_back() -> None:
    """The anti-join Abzu's evaluation stops short of."""
    inputs = _inputs()
    outputs: list[Any] = [CanonicalRecord(id=4, source_ids=[1, 3, 5, 6, 7])]

    repaired, report = recover_dropped_records(inputs, outputs)

    assert len(repaired) == 2
    assert report.coverage_pct == 100.0
    assert report.passes
    assert report.recovered_records == 1
    assert report.missing_ids == []


def test_a_recovered_record_says_it_was_recovered() -> None:
    """A reader has to be able to tell a recovered record from a resolved one."""
    inputs = _inputs()
    outputs: list[Any] = [CanonicalRecord(id=4, source_ids=[1, 3, 5, 6, 7])]

    repaired, _ = recover_dropped_records(inputs, outputs)

    assert repaired[-1]["match_skip_reason"] == MISSING_IN_OUTPUT_REASON


def test_recovery_of_an_entity_marks_the_skip_reason_on_the_model() -> None:
    inputs = [Entity(id=1, name="a"), Entity(id=2, name="b")]
    outputs: list[Any] = [Entity(id=3, name="a", source_ids=[1])]

    repaired, report = recover_dropped_records(inputs, outputs)

    assert report.passes
    assert repaired[-1].match_skip_reason == MISSING_IN_OUTPUT_REASON
    assert repaired[-1].id == 2


def test_recovery_leaves_a_complete_output_alone() -> None:
    inputs = _inputs()
    outputs: list[Any] = [CanonicalRecord(id=4, source_ids=[1, 3, 5, 6, 7]), CanonicalRecord(id=2)]

    repaired, report = recover_dropped_records(inputs, outputs)

    assert repaired is outputs
    assert report.recovered_records == 0


def test_recovering_one_record_can_cover_several_missing_ids() -> None:
    """Adding record 3 back accounts for 5 and 6 as well, so it is added once."""
    inputs = _inputs()
    outputs: list[Any] = [CanonicalRecord(id=1), CanonicalRecord(id=2), CanonicalRecord(id=7)]

    repaired, report = recover_dropped_records(inputs, outputs)

    assert report.recovered_records == 1
    assert report.passes
    assert len(repaired) == 4


def test_recovery_that_cannot_fix_the_data_raises() -> None:
    """Data integrity outranks finishing the run."""
    inputs = [{"id": 1}, {"id": 2}]
    outputs: list[Any] = [
        CanonicalRecord(id=10, source_ids=[1, 2]),
        CanonicalRecord(id=11, source_ids=[2]),
    ]

    with pytest.raises(ValueError, match="Conservation failed after recovery"):
        recover_dropped_records(inputs, outputs)


def test_an_empty_run_conserves_trivially() -> None:
    report = check_conservation([], [])

    assert report.passes
    assert report.coverage_pct == 100.0


def test_the_summary_names_the_verdict() -> None:
    report = check_conservation(_inputs(), [CanonicalRecord(id=4, source_ids=[1, 3, 5, 6, 7])])

    assert report.summary().startswith("FAIL")
    assert "input records reachable" in report.summary()


def test_coverage_of_accepts_both_records_and_mappings() -> None:
    assert coverage_of({"id": 3, "source_ids": [5, 6]}) == {3, 5, 6}
    assert coverage_of(CanonicalRecord(id=3, source_ids=[5, 6])) == {3, 5, 6}
    assert coverage_of(Entity(id=3, name="x", source_ids=[5, 6])) == {3, 5, 6}


def test_an_intermediate_entitys_minted_id_is_real_lineage_not_a_dangling_reference() -> None:
    """The bug the abt-buy run found: a two-round merge leaves a minted id in source_ids."""
    inputs = [{"id": 1}, {"id": 2}, {"id": 3}]
    # Round one merged 1 and 2 into minted entity 4; round two merged 4 with 3
    # into minted entity 5, whose lineage names 4.
    outputs: list[Any] = [CanonicalRecord(id=5, source_ids=[1, 2, 3, 4])]

    unaware = check_conservation(inputs, outputs)
    aware = check_conservation(inputs, outputs, minted_ids={4, 5})

    assert unaware.invalid_reference_ids == [4]
    assert not unaware.passes
    assert aware.invalid_reference_ids == []
    assert aware.passes


def test_recovery_also_takes_the_minted_ids_into_account() -> None:
    inputs = [{"id": 1}, {"id": 2}, {"id": 3}]
    outputs: list[Any] = [CanonicalRecord(id=5, source_ids=[1, 2, 4])]

    repaired, report = recover_dropped_records(inputs, outputs, minted_ids={4, 5})

    assert report.passes
    assert report.recovered_records == 1
    assert len(repaired) == 2


def test_a_reference_to_an_id_nobody_ever_issued_still_dangles() -> None:
    """The check must stay able to catch a genuinely invented reference."""
    inputs = [{"id": 1}, {"id": 2}]
    outputs: list[Any] = [CanonicalRecord(id=5, source_ids=[1, 2, 9999])]

    report = check_conservation(inputs, outputs, minted_ids={5})

    assert report.invalid_reference_ids == [9999]
    assert not report.passes
