"""A hardened dspy.XMLAdapter for real-world text containing XML special characters.

`dspy.XMLAdapter.parse()` uses `xml.etree.ElementTree`, a strict XML parser.
When an entity's `name`/`description` contains a raw `&` (e.g. "Black &
White", "Effective & Efficient..." -- roughly 10% of DBLP-ACM's records), the
model tends to echo it back unescaped, and the response fails to parse as
XML at all. This is a real, measured cause of prediction failures during
GEPA optimization, not a hypothetical edge case.
"""

import re

import dspy

# A bare `&` not already followed by a known XML entity name/numeric
# reference. Escaping only bare ampersands (not attempting to fix stray
# `<`/`>`, which are far rarer in this domain and much harder to
# distinguish from real tags) fixes the overwhelming majority of cases.
_BARE_AMPERSAND = re.compile(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)")


def escape_bare_ampersands(text: str) -> str:
    """Escape `&` characters that are not already part of a valid XML entity.

    Parameters
    ----------
    text : str
        Raw text, possibly containing unescaped ampersands

    Returns
    -------
    str
        Text with every bare `&` replaced by `&amp;`
    """
    return _BARE_AMPERSAND.sub("&amp;", text)


class RobustXMLAdapter(dspy.XMLAdapter):
    """dspy.XMLAdapter that escapes bare ampersands before parsing.

    Behaves identically to dspy.XMLAdapter in every other respect; this is
    the minimal, targeted fix for a real failure mode, not a rewrite.
    """

    def parse(self, signature: type[dspy.Signature], completion: str) -> dict[str, object]:
        return super().parse(signature, escape_bare_ampersands(completion))
