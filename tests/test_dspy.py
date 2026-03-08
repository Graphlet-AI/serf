import os

import dspy  # type: ignore
import pytest

from serf.dspy.baml_adapter import BAMLAdapter  # type: ignore


@pytest.fixture
def lm() -> dspy.LM:
    """Get the BAMLAdapter style language model"""
    # Get Gemini API key from environment variable
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    lm = dspy.LM("gemini/gemini-2.0-flash", api_key=GEMINI_API_KEY)
    dspy.configure(lm=lm, adapter=BAMLAdapter())

    yield lm


def test_dspy_simple_math(lm: dspy.LM) -> None:
    # Test the integration of dspy with the BAML adapter
    math = dspy.ChainOfThought("question -> answer: float")
    math(question="Two dice are tossed. What is the probability that the sum equals two?")
