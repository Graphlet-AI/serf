"""Tests that the Deno runtime `dspy.Flex` needs is installed and usable.

`dspy.Flex` runs optimizer-authored code inside `dspy.PythonInterpreter`, which
is Deno plus Pyodide. The runtime arrives through the `deno` dependency, which
vendors the binary into the virtualenv instead of requiring a system package.
These tests fail loudly if that dependency is ever dropped, because the symptom
without them is a `Flex` that raises only when someone finally tries to use it.
"""

import dspy
from dspy.primitives.python_interpreter import (
    MAX_DENO_VERSION,
    MIN_DENO_VERSION,
    PythonInterpreter,
    _find_deno_executable,
    _get_deno_version,
)
from dspy.utils.dummies import DummyLM

from serf.dspy.dataset_signatures import get_dataset_spec


def test_deno_version_is_one_dspy_supports() -> None:
    """The vendored binary resolves and falls inside dspy's supported range."""
    version = _get_deno_version(_find_deno_executable())
    assert version is not None
    assert MIN_DENO_VERSION <= version < MAX_DENO_VERSION


def test_interpreter_runs_deterministic_python() -> None:
    """The sandbox executes plain Python, which is what Flex trades LM calls for."""
    with PythonInterpreter() as interpreter:
        assert interpreter("sum(range(10))") == 45


def test_flex_forward_bridges_sandbox_to_host_lm() -> None:
    """A Flex call runs guest code and routes the predictor call back to the host."""
    flex = dspy.Flex("question -> answer")
    # Pin the adapter as well as the LM: another test configures an adapter
    # globally, and DummyLM only round-trips through the one it formats for.
    with dspy.context(lm=DummyLM([{"answer": "four"}]), adapter=dspy.ChatAdapter()):
        assert flex(question="how many legs does a dog have?").answer == "four"


def test_flex_baseline_keeps_a_dataset_signature_intact() -> None:
    """The per-dataset instructions survive into the source GEPA starts from."""
    signature = get_dataset_spec("dblp-acm").signature
    module_src = dspy.Flex(signature).module_src
    assert module_src is not None
    assert "class DblpAcmBlockMatchModule(dspy.Module):" in module_src
    assert "Find the DBLP-ACM publication duplicates inside one block." in module_src
