"""Tests for DSPy integration with the BAMLAdapter."""

import os
from collections.abc import Generator

import dspy
import pytest

from serf.dspy.baml_adapter import BAMLAdapter


@pytest.fixture
def lm() -> Generator[dspy.LM, None, None]:
    """Get the BAMLAdapter style language model."""
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    lm = dspy.LM("gemini/gemini-2.0-flash", api_key=GEMINI_API_KEY)
    dspy.configure(lm=lm, adapter=BAMLAdapter())

    yield lm


def test_dspy_simple_math(lm: dspy.LM) -> None:
    """Test the integration of dspy with the BAMLAdapter."""
    math = dspy.ChainOfThought("question -> answer: float")
    math(question="Two dice are tossed. What is the probability that the sum equals two?")
