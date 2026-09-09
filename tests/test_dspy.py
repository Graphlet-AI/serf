"""Tests for DSPy integration with the XMLAdapter."""

import os
from collections.abc import Generator

import dspy
import pytest

from serf.dspy.lm import create_lm

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)


@pytest.fixture
def lm() -> Generator[dspy.LM, None, None]:
    """Get the XMLAdapter style language model using the teacher LM."""
    lm = create_lm(role="teacher")
    dspy.configure(lm=lm, adapter=dspy.XMLAdapter())

    yield lm


def test_dspy_simple_math(lm: dspy.LM) -> None:
    """Test the integration of dspy with the XMLAdapter."""
    math = dspy.ChainOfThought("question -> answer: float")
    math(question="Two dice are tossed. What is the probability that the sum equals two?")
