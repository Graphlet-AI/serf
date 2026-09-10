"""Tests that litellm is fully imported before matcher threads use it.

DSPy imports litellm on first use. When several matcher threads make their first
LLM call at the same time, they can observe a half-executed litellm module and
lose the block to ``partially initialized module 'litellm' has no attribute
'completion'``. Importing litellm when ``serf.dspy.lm`` loads removes the race.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any

import serf.dspy.lm  # noqa: F401  imported for its litellm side effect


def test_litellm_is_materialized_when_the_lm_factory_is_imported() -> None:
    """Importing the LM factory leaves a fully executed litellm in sys.modules."""
    module = sys.modules.get("litellm")

    assert isinstance(module, ModuleType)
    assert callable(module.completion)


def test_concurrent_first_use_sees_the_completion_attribute() -> None:
    """Threads reaching litellm at the same time all see a usable module."""

    def read_completion(_index: int) -> Any:
        return sys.modules["litellm"].completion

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(read_completion, range(8)))

    assert all(callable(result) for result in results)


def test_dspy_resolves_the_same_litellm_module() -> None:
    """DSPy's lazy loader hands back the module serf already imported."""
    from dspy.clients._litellm import get_litellm

    assert get_litellm(feature="test") is sys.modules["litellm"]
