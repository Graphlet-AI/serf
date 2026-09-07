"""Driver script for a GEPA optimization run against DBLP-ACM.

Not part of the package: a one-off script for this build session's
optimization run, invoked directly with `uv run python scripts/run_gepa_dblp_acm.py`.
"""

# isort: off
# numpy must load before dspy in a fresh process -- see the matching comment
# in src/serf/dspy/optimize.py.
import numpy  # noqa: F401
import dspy  # noqa: F401

# isort: on
import json
import sys
import time

from serf.dspy.optimize import run_gepa_optimization
from serf.logs import get_logger

logger = get_logger(__name__)


def main() -> None:
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    auto = sys.argv[2] if len(sys.argv) > 2 else "light"

    start = time.time()
    result = run_gepa_optimization(
        dataset_name="dblp-acm",
        task_model="gemini/gemini-3.5-flash-lite",
        reflection_model="gemini/gemini-3.7-flash",
        sample_size=sample_size,
        target_block_size=30,
        auto=auto,
        seed=0,
    )
    elapsed = time.time() - start

    optimized = result.pop("optimized_program")
    print(json.dumps(result, indent=2))
    print(f"Elapsed: {elapsed:.1f}s")

    out_path = f"data/gepa/dblp_acm_optimized_{sample_size}_{auto}.json"
    import os

    os.makedirs("data/gepa", exist_ok=True)
    optimized.save(out_path)
    print(f"Saved optimized program to {out_path}")


if __name__ == "__main__":
    main()
