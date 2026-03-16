"""Tests for the comprehensive ER evaluator."""

from serf.dspy.types import BlockResolution, Entity
from serf.eval.evaluator import evaluate_er_results, format_evaluation_report


def _make_entity(
    eid: int,
    name: str = "Test",
    uuid: str | None = None,
    source_ids: list[int] | None = None,
    source_uuids: list[str] | None = None,
    match_skip: bool | None = None,
    match_skip_reason: str | None = None,
) -> Entity:
    return Entity(
        id=eid,
        name=name,
        uuid=uuid,
        source_ids=source_ids,
        source_uuids=source_uuids,
        match_skip=match_skip,
        match_skip_reason=match_skip_reason,
    )


def test_basic_evaluation() -> None:
    """Test evaluation with simple merged and unmerged entities."""
    resolutions = [
        BlockResolution(
            block_key="b1",
            resolved_entities=[
                _make_entity(1, uuid="u1", source_ids=[2], source_uuids=["u2"]),
                _make_entity(3, uuid="u3"),
            ],
            was_resolved=True,
            original_count=3,
            resolved_count=2,
        ),
    ]
    metrics = evaluate_er_results(resolutions, original_entity_count=3)
    assert metrics["unique_entities"] == 2
    assert metrics["entities_with_merges"] == 1
    assert metrics["reduction_pct"] > 0
    assert metrics["overall_status"] in ("PASS", "FAIL")


def test_duplicate_removal() -> None:
    """Test that exact duplicate entities are removed."""
    entity = _make_entity(1, uuid="u1")
    resolutions = [
        BlockResolution(
            block_key="b1",
            resolved_entities=[entity, entity],
            original_count=2,
            resolved_count=2,
        ),
    ]
    metrics = evaluate_er_results(resolutions, original_entity_count=2)
    assert metrics["unique_entities"] == 1
    assert metrics["duplicates_removed"] == 1


def test_skip_reason_analysis() -> None:
    """Test match_skip_reason distribution."""
    resolutions = [
        BlockResolution(
            block_key="b1",
            resolved_entities=[
                _make_entity(1, uuid="u1"),
                _make_entity(2, uuid="u2", match_skip=True, match_skip_reason="error_recovery"),
                _make_entity(3, uuid="u3", match_skip=True, match_skip_reason="singleton_block"),
                _make_entity(
                    4, uuid="u4", match_skip=True, match_skip_reason="missing_in_match_output"
                ),
            ],
            original_count=4,
            resolved_count=4,
        ),
    ]
    metrics = evaluate_er_results(resolutions, original_entity_count=4)
    assert metrics["skip_reasons"]["error_recovery"] == 1
    assert metrics["skip_reasons"]["singleton_block"] == 1
    assert metrics["skip_reasons"]["missing_in_match_output"] == 1
    assert metrics["skipped_entities"] == 3
    assert metrics["llm_processed"] == 1


def test_uuid_validation_pass() -> None:
    """Test UUID validation passes when all source_uuids are known."""
    resolutions = [
        BlockResolution(
            block_key="b1",
            resolved_entities=[
                _make_entity(1, uuid="u1", source_uuids=["u2", "u3"]),
            ],
            original_count=3,
            resolved_count=1,
        ),
    ]
    historical = {"u1", "u2", "u3"}
    metrics = evaluate_er_results(resolutions, original_entity_count=3, historical_uuids=historical)
    assert metrics["uuid_validation"]["passed"] is True
    assert metrics["uuid_validation"]["coverage_pct"] == 100.0


def test_uuid_validation_fail() -> None:
    """Test UUID validation fails when source_uuids reference unknown UUIDs."""
    resolutions = [
        BlockResolution(
            block_key="b1",
            resolved_entities=[
                _make_entity(1, uuid="u1", source_uuids=["u2", "unknown"]),
            ],
            original_count=2,
            resolved_count=1,
        ),
    ]
    historical = {"u1", "u2"}
    metrics = evaluate_er_results(resolutions, original_entity_count=2, historical_uuids=historical)
    assert metrics["uuid_validation"]["invalid_source_uuids"] == 1
    assert "unknown" in metrics["uuid_validation"]["missing_uuids"]


def test_overall_reduction() -> None:
    """Test overall reduction from original baseline."""
    resolutions = [
        BlockResolution(
            block_key="b1",
            resolved_entities=[_make_entity(1, uuid="u1", source_ids=[2])],
            original_count=4,
            resolved_count=1,
        ),
    ]
    metrics = evaluate_er_results(resolutions, original_entity_count=100, previous_entity_count=4)
    assert metrics["iteration_input_entities"] == 4
    assert metrics["original_entity_count"] == 100
    assert metrics["overall_reduction_pct"] == 99.0


def test_format_report() -> None:
    """Test that format_evaluation_report produces readable output."""
    resolutions = [
        BlockResolution(
            block_key="b1",
            resolved_entities=[
                _make_entity(1, uuid="u1", source_ids=[2], source_uuids=["u2"]),
                _make_entity(3, uuid="u3"),
            ],
            was_resolved=True,
            original_count=3,
            resolved_count=2,
        ),
    ]
    metrics = evaluate_er_results(resolutions, original_entity_count=3)
    report = format_evaluation_report(metrics)
    assert "Evaluation Report" in report
    assert "Entity Counts" in report
    assert "Reduction" in report
    assert "Checks" in report


def test_no_historical_uuids_skips_validation() -> None:
    """Test that UUID validation is skipped when no historical UUIDs provided."""
    resolutions = [
        BlockResolution(
            block_key="b1",
            resolved_entities=[_make_entity(1, uuid="u1")],
            original_count=1,
            resolved_count=1,
        ),
    ]
    metrics = evaluate_er_results(resolutions, original_entity_count=1)
    assert metrics["uuid_validation"]["skipped"] is True
