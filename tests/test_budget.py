"""Tests for persistent, hard-capped budget ledgers."""

from unittest.mock import MagicMock, patch

import pytest

from serf.dspy.budget import BudgetExceededError, BudgetLedger, TrackedLM, get_ledger


def test_ledger_starts_at_zero(tmp_path: object) -> None:
    """A freshly created ledger has zero spend and zero calls."""
    ledger = BudgetLedger("test", cap_usd=10.0, ledger_dir=str(tmp_path))
    assert ledger.spend_usd == 0.0
    assert ledger.call_count == 0


def test_ledger_persists_across_instances(tmp_path: object) -> None:
    """Spend survives reattaching to the same ledger file (simulating a
    process restart), which is the entire point of a hard budget guard."""
    ledger = BudgetLedger("test", cap_usd=10.0, ledger_dir=str(tmp_path))
    ledger.record(1.5)
    ledger.record(2.5)

    reattached = BudgetLedger("test", cap_usd=10.0, ledger_dir=str(tmp_path))
    assert reattached.spend_usd == 4.0
    assert reattached.call_count == 2


def test_record_zero_cost_does_not_increase_spend(tmp_path: object) -> None:
    """A cache hit (cost 0.0) is still counted as a call but adds no spend."""
    ledger = BudgetLedger("test", cap_usd=10.0, ledger_dir=str(tmp_path))
    ledger.record(0.0)
    assert ledger.spend_usd == 0.0
    assert ledger.call_count == 1


def test_check_raises_once_cap_reached(tmp_path: object) -> None:
    """check() raises BudgetExceededError once cumulative spend >= cap."""
    ledger = BudgetLedger("test", cap_usd=5.0, ledger_dir=str(tmp_path))
    ledger.check()  # no spend yet, should not raise
    ledger.record(5.0)
    with pytest.raises(BudgetExceededError):
        ledger.check()


def test_get_ledger_reads_cap_from_config() -> None:
    """get_ledger builds a BudgetLedger using budget.<name>.cap_usd from config.yml."""
    ledger = get_ledger("gemini")
    assert ledger.cap_usd == 100.0
    assert ledger.name == "gemini"


def test_get_ledger_unknown_name_raises() -> None:
    """An unconfigured ledger name is a programming error, not silently ignored."""
    with pytest.raises(KeyError):
        get_ledger("does_not_exist_in_config")


def test_tracked_lm_records_real_cost_after_call(tmp_path: object) -> None:
    """TrackedLM.update_history records the litellm-computed cost into the ledger."""
    ledger = BudgetLedger("test", cap_usd=10.0, ledger_dir=str(tmp_path))
    lm = TrackedLM("gemini/gemini-3.5-flash-lite", ledger=ledger, api_key="fake")
    lm.update_history({"cost": 0.0042})
    assert ledger.spend_usd == pytest.approx(0.0042)


def test_tracked_lm_treats_missing_cost_as_zero(tmp_path: object) -> None:
    """A cache hit has cost=None in its history entry; that must not crash
    or be treated as an unknown/non-zero charge."""
    ledger = BudgetLedger("test", cap_usd=10.0, ledger_dir=str(tmp_path))
    lm = TrackedLM("gemini/gemini-3.5-flash-lite", ledger=ledger, api_key="fake")
    lm.update_history({"cost": None})
    assert ledger.spend_usd == 0.0
    assert ledger.call_count == 1


def test_tracked_lm_call_rejected_once_ledger_exhausted(tmp_path: object) -> None:
    """__call__ checks the ledger and raises before ever reaching the network
    once the cap is already exhausted."""
    ledger = BudgetLedger("test", cap_usd=1.0, ledger_dir=str(tmp_path))
    ledger.record(1.0)
    lm = TrackedLM("gemini/gemini-3.5-flash-lite", ledger=ledger, api_key="fake")

    with (
        patch("dspy.LM.__call__", return_value=["should not be reached"]) as mock_call,
        pytest.raises(BudgetExceededError),
    ):
        lm("some prompt")
    mock_call.assert_not_called()


def test_tracked_lm_call_allowed_under_cap(tmp_path: object) -> None:
    """__call__ passes through to dspy.LM.__call__ when under the cap."""
    ledger = BudgetLedger("test", cap_usd=10.0, ledger_dir=str(tmp_path))
    lm = TrackedLM("gemini/gemini-3.5-flash-lite", ledger=ledger, api_key="fake")

    with patch("dspy.LM.__call__", return_value=["ok"]) as mock_call:
        result = lm("some prompt")
    mock_call.assert_called_once()
    assert result == ["ok"]


def test_matcher_uses_gemini_ledger_for_gemini_model() -> None:
    """EntityMatcher routes Gemini models to the 'gemini' ledger."""
    from serf.match.matcher import EntityMatcher

    matcher = EntityMatcher(model="gemini/gemini-3.5-flash-lite")
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        lm = matcher._ensure_lm()
    assert isinstance(lm, TrackedLM)
    assert lm.ledger.name == "gemini"


def test_matcher_uses_gpt_oss_ledger_for_gpt_oss_model() -> None:
    """EntityMatcher routes gpt-oss models to their own separate ledger,
    never the Gemini cap (docs/SERF_LONG_SHOT_PLAN.md Section 9.6, rule 6)."""
    from serf.match.matcher import EntityMatcher

    matcher = EntityMatcher(model="openai/gpt-oss-120b-maas")
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        lm = matcher._ensure_lm()
    assert isinstance(lm, TrackedLM)
    assert lm.ledger.name == "gpt_oss_120b_maas"


def test_tracked_lm_is_a_magicmock_safe_subclass() -> None:
    """Sanity check that TrackedLM can be constructed without a real API key
    reaching the network (no call made at construction time)."""
    with patch("dspy.LM.__init__", return_value=None) as mock_init:
        TrackedLM("gemini/gemini-3.5-flash-lite", ledger=MagicMock(), api_key="fake")
    mock_init.assert_called_once()
