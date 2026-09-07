"""Tests for EntityMatcher's retry and LM configuration behavior."""

from unittest.mock import MagicMock, PropertyMock, patch

from serf.dspy.types import BlockResolution, Entity, EntityBlock
from serf.match.matcher import EntityMatcher


def test_retries_recover_from_a_transient_failure() -> None:
    """A predictor call that fails once and succeeds on retry returns the
    successful resolution rather than falling back to error_recovery."""
    matcher = EntityMatcher()
    block = EntityBlock(
        block_key="b1",
        block_size=2,
        entities=[Entity(id=100, name="A"), Entity(id=200, name="B")],
    )
    success = BlockResolution(
        block_key="b1", resolved_entities=[], was_resolved=False, original_count=2
    )
    mock_predictor = MagicMock(
        side_effect=[RuntimeError("transient"), MagicMock(resolution=success)]
    )

    with (
        patch.object(EntityMatcher, "_ensure_lm", return_value=MagicMock()),
        patch.object(EntityMatcher, "predictor", new_callable=PropertyMock) as mock_prop,
        patch("time.sleep"),
    ):
        mock_prop.return_value = mock_predictor
        resolution = matcher.resolve_block(block)

    assert mock_predictor.call_count == 2
    assert all(e.match_skip_reason != "error_recovery" for e in resolution.resolved_entities)


def test_exhausting_retries_falls_back_to_error_recovery() -> None:
    """A predictor that always fails exhausts max_retries and falls back
    to error_recovery, rather than raising out of resolve_block."""
    matcher = EntityMatcher()
    block = EntityBlock(
        block_key="b1",
        block_size=2,
        entities=[Entity(id=100, name="A"), Entity(id=200, name="B")],
    )
    mock_predictor = MagicMock(side_effect=RuntimeError("permanent"))

    with (
        patch.object(EntityMatcher, "_ensure_lm", return_value=MagicMock()),
        patch.object(EntityMatcher, "predictor", new_callable=PropertyMock) as mock_prop,
        patch("time.sleep"),
    ):
        mock_prop.return_value = mock_predictor
        resolution = matcher.resolve_block(block)

    assert mock_predictor.call_count == 3  # config default er.matching.max_retries
    assert all(e.match_skip_reason == "error_recovery" for e in resolution.resolved_entities)


def test_max_output_tokens_read_from_config() -> None:
    """The LM is constructed with er.matching.max_output_tokens (65536 by
    default), not a small hardcoded value that truncates large blocks."""
    matcher = EntityMatcher()
    with (
        patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
        patch("serf.match.matcher.dspy.LM") as mock_lm,
    ):
        matcher._ensure_lm()
    _, kwargs = mock_lm.call_args
    assert kwargs["max_tokens"] == 65536
