"""DSPy types, signatures, LM factory, and GEPA optimization.

Only lightweight members are re-exported here. ``serf.dspy.optimize`` imports
``serf.block.pipeline``, which imports ``serf.dspy.types``, so re-exporting it
from this package would create an import cycle. Import it directly instead:
``from serf.dspy.optimize import optimize_module``.
"""

from serf.dspy.lm import create_lm
from serf.eval.splits import get_all_split_sizes, get_split_sizes

__all__ = [
    "create_lm",
    "get_all_split_sizes",
    "get_split_sizes",
]
