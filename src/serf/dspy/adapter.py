"""XML adapter that repairs the metacharacters a model leaves unescaped.

``dspy.XMLAdapter`` parses the completion with ``xml.etree.ElementTree``, which
rejects the whole document over a single bare ``&``. Entity resolution echoes
source text back through the output fields, and real product catalogs are full
of ``Home & Student``, ``AT&T`` and ``< 30 lbs``, so the strict parse fails on
most blocks. DSPy answers a parse failure by re-running the block through
``JSONAdapter``, which doubles the LLM calls and introduces its own failures, so
the cost of a bare ampersand is a second inference and sometimes a lost block.

Escaping stray metacharacters before the retry recovers those blocks without
touching well-formed markup, since a repaired document is only ever parsed after
the unrepaired one has already failed.
"""

import re
from typing import Any

import dspy
from dspy.signatures.signature import Signature

from serf.logs import get_logger

logger = get_logger(__name__)

# An ampersand that does not open a named, decimal or hex character reference.
BARE_AMPERSAND = re.compile(r"&(?!(?:#[0-9]+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);)")

# A less-than that does not open a tag, a closing tag, a comment or a PI.
BARE_LESS_THAN = re.compile(r"<(?![/?!]?[A-Za-z_])")


def escape_stray_metacharacters(completion: str) -> str:
    """Escape ``&`` and ``<`` that are text rather than markup.

    Parameters
    ----------
    completion : str
        Raw XML completion from the LM

    Returns
    -------
    str
        The completion with stray metacharacters replaced by entities
    """
    return BARE_LESS_THAN.sub("&lt;", BARE_AMPERSAND.sub("&amp;", completion))


class RepairingXMLAdapter(dspy.XMLAdapter):
    """``dspy.XMLAdapter`` that retries a failed parse with escaped text."""

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse a completion, escaping stray metacharacters on failure.

        Parameters
        ----------
        signature : type[Signature]
            DSPy signature whose output fields are being read
        completion : str
            Raw XML completion from the LM

        Returns
        -------
        dict[str, Any]
            Parsed output fields
        """
        try:
            return super().parse(signature, completion)
        except Exception:
            repaired = escape_stray_metacharacters(completion)
            if repaired == completion:
                raise
            logger.debug("Retrying XML parse with stray metacharacters escaped")
            return super().parse(signature, repaired)
