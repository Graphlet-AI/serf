"""Tests for few_shot module."""

import json

from serf.match.few_shot import format_few_shot_examples, get_default_few_shot_examples


def test_get_default_few_shot_examples_returns_valid_json() -> None:
    """get_default_few_shot_examples returns parseable JSON."""
    result = get_default_few_shot_examples()
    parsed = json.loads(result)
    assert "input" in parsed
    assert "output" in parsed


def test_get_default_few_shot_examples_has_correct_merge_example() -> None:
    """Default example shows id 1 and 22 merging with source_ids accumulation."""
    result = get_default_few_shot_examples()
    parsed = json.loads(result)
    assert parsed["input"][0]["id"] == 1
    assert parsed["input"][0]["source_ids"] == [3, 7]
    assert parsed["input"][1]["id"] == 22
    assert parsed["input"][1]["source_ids"] == [2, 4]
    out = parsed["output"]["resolved_entities"][0]
    assert out["id"] == 1
    assert out["source_ids"] == [22, 3, 7, 2, 4]


def test_format_few_shot_examples() -> None:
    """format_few_shot_examples formats custom examples as JSON."""
    examples = [{"input": [{"id": 1}], "output": {"resolved": [{"id": 1}]}}]
    result = format_few_shot_examples(examples)
    parsed = json.loads(result)
    assert parsed == examples
