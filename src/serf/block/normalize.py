"""Entity name normalization for blocking.

Provides name cleaning, corporate suffix removal, acronym generation,
and multilingual stop word filtering for creating blocking keys.
"""

import re
import unicodedata

from cleanco import basename

from serf.logs import get_logger

logger = get_logger(__name__)

# Common multilingual stop words for acronym filtering
_STOP_WORDS: set[str] | None = None


def get_multilingual_stop_words() -> set[str]:
    """Get a set of common multilingual stop words.

    Returns
    -------
    set[str]
        Set of lowercase stop words
    """
    global _STOP_WORDS
    if _STOP_WORDS is not None:
        return _STOP_WORDS

    _STOP_WORDS = {
        # English
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "to",
        "for",
        "is",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "into",
        "its",
        "it",
        "not",
        "but",
        "be",
        "are",
        "was",
        "were",
        "been",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "shall",
        "may",
        "can",
        "could",
        "would",
        "should",
        "this",
        "that",
        "these",
        "those",
        # German
        "der",
        "die",
        "das",
        "und",
        "oder",
        "von",
        "mit",
        "fur",
        "für",
        # French
        "le",
        "la",
        "les",
        "de",
        "du",
        "des",
        "et",
        "ou",
        "en",
        # Spanish
        "el",
        "los",
        "las",
        "del",
        "por",
        "con",
        "una",
        "uno",
        "y",
        "o",
    }
    return _STOP_WORDS


def normalize_name(name: str) -> str:
    """Normalize an entity name for comparison.

    Lowercases, strips whitespace, removes punctuation, normalizes
    unicode characters, and collapses multiple spaces.

    Parameters
    ----------
    name : str
        The entity name to normalize

    Returns
    -------
    str
        Normalized name
    """
    # Normalize unicode and strip combining characters (accents)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    # Lowercase
    name = name.lower().strip()
    # Remove punctuation (keep alphanumeric and spaces)
    name = re.sub(r"[^\w\s]", " ", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_basename(name: str) -> str:
    """Remove corporate suffixes from a company name.

    Uses the cleanco library to strip Inc., LLC, Ltd., Corp., etc.

    Parameters
    ----------
    name : str
        Company name with potential corporate suffix

    Returns
    -------
    str
        Company name without corporate suffix
    """
    result: str = basename(name)
    return result


def get_corporate_ending(name: str) -> str:
    """Extract the corporate ending from a company name.

    Parameters
    ----------
    name : str
        Company name

    Returns
    -------
    str
        The corporate suffix, or empty string if none found
    """
    base = get_basename(name)
    ending = name[len(base) :].strip()
    return ending


def get_acronyms(name: str) -> list[str]:
    """Generate acronyms from an entity name.

    Removes corporate suffixes, filters stop words, then creates
    acronyms from the initial letters of remaining words.

    Parameters
    ----------
    name : str
        Entity name

    Returns
    -------
    list[str]
        List of generated acronyms (may be empty)
    """
    stop_words = get_multilingual_stop_words()
    # Remove corporate suffix
    clean = get_basename(name)
    # Normalize
    clean = normalize_name(clean)
    # Split into words and filter stop words
    words = [w for w in clean.split() if w not in stop_words and len(w) > 1]

    if len(words) < 2:
        return []

    # Generate acronym from first letters
    acronym = "".join(w[0] for w in words).upper()
    return [acronym]


# Common TLD suffixes for domain-based blocking
DOMAIN_SUFFIXES = {
    ".com",
    ".org",
    ".net",
    ".edu",
    ".gov",
    ".io",
    ".co",
    ".ai",
    ".us",
    ".uk",
    ".de",
    ".fr",
    ".jp",
    ".cn",
    ".in",
    ".br",
    ".au",
    ".ca",
    ".ru",
    ".it",
    ".es",
    ".nl",
    ".se",
    ".no",
    ".fi",
    ".dk",
    ".ch",
    ".at",
    ".be",
    ".pt",
    ".pl",
    ".cz",
    ".ie",
    ".nz",
    ".sg",
    ".hk",
    ".kr",
    ".tw",
    ".mx",
    ".ar",
    ".cl",
    ".co.uk",
    ".co.jp",
    ".com.au",
    ".com.br",
    ".co.in",
    ".com.mx",
}


def remove_domain_suffix(name: str) -> str:
    """Remove domain suffixes from a name for blocking.

    Parameters
    ----------
    name : str
        Name that may contain a domain suffix

    Returns
    -------
    str
        Name without domain suffix
    """
    lower = name.lower()
    for suffix in sorted(DOMAIN_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name
