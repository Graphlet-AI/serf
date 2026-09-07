"""Driver script for a GEPA optimization run against a benchmark dataset.

Not part of the package: a one-off script for this build session's
optimization runs, invoked directly with
`uv run python scripts/run_gepa.py <dataset> <sample_size> <auto>`.
"""

# isort: off
# numpy must load before dspy in a fresh process -- see the matching comment
# in src/serf/dspy/optimize.py.
import numpy  # noqa: F401
import dspy  # noqa: F401

# isort: on
import json
import os
import sys
import time

from serf.dspy.optimize import run_gepa_optimization
from serf.logs import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "dblp-acm"
    sample_arg = sys.argv[2] if len(sys.argv) > 2 else "300"
    sample_size = None if sample_arg == "full" else int(sample_arg)
    auto_or_calls = sys.argv[3] if len(sys.argv) > 3 else "light"
    auto = auto_or_calls if auto_or_calls in ("light", "medium", "heavy") else None
    max_metric_calls = None if auto else int(auto_or_calls)
    require_true_pair = "--all-blocks" not in sys.argv

    start = time.time()
    result = run_gepa_optimization(
        dataset_name=dataset_name,
        task_model="gemini/gemini-3.5-flash-lite",
        reflection_model="gemini/gemini-3.7-flash",
        sample_size=sample_size,
        target_block_size=30,
        auto=auto,
        max_metric_calls=max_metric_calls,
        require_true_pair=require_true_pair,
        seed=0,
    )
    elapsed = time.time() - start

    optimized = result.pop("optimized_program")
    print(json.dumps(result, indent=2))
    print(f"Elapsed: {elapsed:.1f}s")

    dataset_slug = dataset_name.replace("-", "_")
    out_path = f"data/gepa/{dataset_slug}_optimized_{sample_size}_{auto_or_calls}.json"
    os.makedirs("data/gepa", exist_ok=True)
    optimized.save(out_path)
    print(f"Saved optimized program to {out_path}")


if __name__ == "__main__":
    main()
