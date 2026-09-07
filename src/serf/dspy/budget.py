"""Persistent, hard-capped budget ledgers for LLM API spend.

`TrackedLM` wraps `dspy.LM` so that every call's real cost -- computed by
litellm from the model's list price and the call's actual token usage, and
zero on a cache hit -- is recorded into a `BudgetLedger`. The ledger is a
small JSON file, not an in-memory counter, so restarting the process does
not reset the cap: the whole point of a hard budget guard is that it
survives crashes and restarts (docs/SERF_LONG_SHOT_PLAN.md Section 9.6).
"""

import json
import threading
from pathlib import Path
from typing import Any

import dspy

from serf.logs import get_logger

logger = get_logger(__name__)

DEFAULT_LEDGER_DIR = "data/budget"


class BudgetExceededError(Exception):
    """Raised when a ledger's cap has already been reached."""


class BudgetLedger:
    """Tracks one named budget's cumulative USD spend, persisted to disk."""

    def __init__(self, name: str, cap_usd: float, ledger_dir: str = DEFAULT_LEDGER_DIR) -> None:
        """Initialize (or reattach to) a named, capped, file-backed ledger.

        Parameters
        ----------
        name : str
            Ledger name (e.g. "gemini", "gpt_oss_120b_maas"). One file per name.
        cap_usd : float
            Hard spend cap in USD. `check()` raises once cumulative spend >= this.
        ledger_dir : str
            Directory holding one JSON file per ledger name.
        """
        self.name = name
        self.cap_usd = cap_usd
        self._path = Path(ledger_dir) / f"{name}.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self._path.exists():
            self._write(0.0, 0)

    def _read(self) -> tuple[float, int]:
        data = json.loads(self._path.read_text())
        return data["spend_usd"], data["call_count"]

    def _write(self, spend_usd: float, call_count: int) -> None:
        self._path.write_text(
            json.dumps({"spend_usd": spend_usd, "call_count": call_count}, indent=2)
        )

    @property
    def spend_usd(self) -> float:
        """Current cumulative spend in USD."""
        return self._read()[0]

    @property
    def call_count(self) -> int:
        """Current cumulative non-cached call count."""
        return self._read()[1]

    def check(self) -> None:
        """Raise BudgetExceededError if this ledger's cap is already reached."""
        spend, _ = self._read()
        if spend >= self.cap_usd:
            raise BudgetExceededError(
                f"{self.name!r} budget exhausted: ${spend:.4f} spent >= ${self.cap_usd:.2f} cap"
            )

    def record(self, cost_usd: float) -> float:
        """Record an actual spend (0.0 for cache hits) and return the new total.

        Parameters
        ----------
        cost_usd : float
            Actual USD cost of the call that just completed.

        Returns
        -------
        float
            New cumulative spend after recording.
        """
        with self._lock:
            spend, calls = self._read()
            spend += cost_usd
            calls += 1
            self._write(spend, calls)
        if spend > self.cap_usd:
            logger.warning(f"{self.name!r} budget EXCEEDED: ${spend:.4f} > ${self.cap_usd:.2f} cap")
        return spend


def get_ledger(name: str) -> BudgetLedger:
    """Build the named BudgetLedger using its cap from config.yml `budget.<name>.cap_usd`.

    Parameters
    ----------
    name : str
        Ledger name matching a `budget.<name>` section in config.yml

    Returns
    -------
    BudgetLedger
        Ledger enforcing that section's cap_usd
    """
    from serf.config import config

    cap = config.get(f"budget.{name}.cap_usd")
    return BudgetLedger(name=name, cap_usd=float(cap))


class TrackedLM(dspy.LM):
    """A dspy.LM that records every call's real cost into a BudgetLedger.

    Rejects a call before it happens if the ledger is already at or over
    its cap; otherwise behaves exactly like dspy.LM. Works for both the
    sync and async call paths, since both funnel through `update_history`.
    """

    def __init__(self, *args: Any, ledger: BudgetLedger, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ledger = ledger

    def update_history(self, entry: dict[str, Any]) -> None:
        super().update_history(entry)
        self.ledger.record(entry.get("cost") or 0.0)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.ledger.check()
        return super().__call__(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        self.ledger.check()
        return await super().acall(*args, **kwargs)
