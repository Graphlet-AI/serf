"""Few-shot example generation for block matching.

Provides default and custom examples showing correct ID tracking and
source_ids accumulation for the BlockMatch signature.
"""

import json
from typing import Any


def get_default_few_shot_examples() -> str:
    """Return a JSON string with a default merge example.

    The example shows correct ID tracking and source_ids accumulation:
    entities with ids 1 (source_ids=[3,7]) and 22 (source_ids=[2,4])
    merging into master id=1, source_ids=[22,3,7,2,4].

    Returns
    -------
    str
        JSON string with the default merge example
    """
    example = {
        "input": [
            {"id": 1, "name": "Acme Corp", "source_ids": [3, 7]},
            {"id": 22, "name": "ACME Corporation", "source_ids": [2, 4]},
        ],
        "output": {
            "resolved_entities": [
                {
                    "id": 1,
                    "name": "Acme Corp",
                    "source_ids": [22, 3, 7, 2, 4],
                },
            ],
            "matches": [
                {
                    "entity_a_id": 1,
                    "entity_b_id": 22,
                    "is_match": True,
                    "confidence": 0.95,
                    "reasoning": "Same company, different naming",
                },
            ],
        },
    }
    return json.dumps(example, indent=2)


def format_few_shot_examples(examples: list[dict[str, Any]]) -> str:
    """Format custom examples as a JSON string.

    Parameters
    ----------
    examples : list[dict[str, Any]]
        List of example dicts, each with input/output structure

    Returns
    -------
    str
        JSON string with formatted examples
    """
    return json.dumps(examples, indent=2)
