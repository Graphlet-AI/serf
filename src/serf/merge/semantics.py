"""Type-aware rules for combining the values of one field across merged records.

When several records turn out to denote one entity, each of their fields holds
several values, and most of those values are spellings of one another rather
than genuinely different facts. "Russell H Jurney" and "Russ Journey" are one
person's name written twice; "CA" and "WA" are two states. Collapsing the first
pair and keeping the second apart needs to know what kind of value the field
holds, which is what ``serf.analyze.field_detection`` already infers.

Freeform prose is the exception that proves the rule: two descriptions of one
company are two facts, not two spellings, so nothing here ever merges them.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from serf.config import config
from serf.logs import get_logger

logger = get_logger(__name__)

DEDUPE_FUZZY = "fuzzy"
DEDUPE_EXACT = "exact"
DEDUPE_NONE = "none"
DEDUPE_STRATEGIES = (DEDUPE_FUZZY, DEDUPE_EXACT, DEDUPE_NONE)

FIELD_TYPE_NAME = "name"
FIELD_TYPE_ADDRESS = "address"
FIELD_TYPE_URL = "url"
FIELD_TYPE_EMAIL = "email"
FIELD_TYPE_PHONE = "phone"
FIELD_TYPE_IDENTIFIER = "identifier"
FIELD_TYPE_DATE = "date"
FIELD_TYPE_NUMERIC = "numeric"
FIELD_TYPE_TEXT = "text"

_DEFAULT_THRESHOLD = 0.8
_PUNCTUATION = ".,;:!?'\"()[]{}<>@#*_\\/|`~^"

# Street-type abbreviations collapse two spellings of one address. Kept short
# and one-directional on purpose: a longer table starts merging addresses that
# only look alike.
_ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "rd": "road",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "pkwy": "parkway",
    "hwy": "highway",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

_URL_PREFIXES = ("https://", "http://", "//")


@dataclass(frozen=True)
class MergePolicy:
    """How the values of one field are combined when records merge.

    Parameters
    ----------
    dedupe : str
        ``fuzzy`` to collapse values whose similarity clears ``threshold``,
        ``exact`` to collapse only values that normalise identically, ``none``
        to keep every distinct value
    threshold : float
        Similarity above which two values are treated as one, used by ``fuzzy``
    """

    dedupe: str = DEDUPE_EXACT
    threshold: float = _DEFAULT_THRESHOLD


def policy_for(field_type: str) -> MergePolicy:
    """Return the configured merge policy for an inferred field type.

    Parameters
    ----------
    field_type : str
        Type from ``serf.analyze.field_detection.detect_field_type``

    Returns
    -------
    MergePolicy
        Policy from ``merge.semantics``, falling back to exact deduplication
        for a type the config does not mention
    """
    settings: dict[str, Any] = config.get(f"merge.semantics.{field_type}", {}) or {}
    dedupe = str(settings.get("dedupe", DEDUPE_EXACT))
    if dedupe not in DEDUPE_STRATEGIES:
        logger.warning(
            f"Unknown dedupe strategy '{dedupe}' for field type '{field_type}'; "
            f"using '{DEDUPE_EXACT}'. Valid strategies: {DEDUPE_STRATEGIES}"
        )
        dedupe = DEDUPE_EXACT
    default_threshold = float(config.get("matcher.similarity.min_threshold", _DEFAULT_THRESHOLD))
    return MergePolicy(dedupe=dedupe, threshold=float(settings.get("threshold", default_threshold)))


def normalize(value: Any, field_type: str = FIELD_TYPE_TEXT) -> str:
    """Reduce a value to the form two spellings of it would share.

    Parameters
    ----------
    value : Any
        Raw field value
    field_type : str
        Inferred type, which decides how aggressively the value is reduced

    Returns
    -------
    str
        Normalised text, empty when the value carries nothing
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    if field_type == FIELD_TYPE_PHONE:
        return "".join(char for char in text if char.isdigit())

    if field_type == FIELD_TYPE_URL:
        lowered = text.lower().rstrip("/")
        for prefix in _URL_PREFIXES:
            if lowered.startswith(prefix):
                lowered = lowered[len(prefix) :]
                break
        return lowered[4:] if lowered.startswith("www.") else lowered

    if field_type == FIELD_TYPE_EMAIL:
        return text.lower()

    if field_type == FIELD_TYPE_IDENTIFIER:
        return "".join(char for char in text.lower() if char.isalnum())

    stripped = "".join(" " if char in _PUNCTUATION else char for char in text.lower())
    tokens = stripped.split()

    if field_type == FIELD_TYPE_ADDRESS:
        tokens = [_ADDRESS_ABBREVIATIONS.get(token, token) for token in tokens]

    return " ".join(tokens)


def similarity(left: Any, right: Any, field_type: str = FIELD_TYPE_TEXT) -> float:
    """Score how alike two values are once normalised.

    Parameters
    ----------
    left : Any
        First value
    right : Any
        Second value
    field_type : str
        Inferred type, which decides the normalisation applied first

    Returns
    -------
    float
        Ratio between 0.0 and 1.0; 1.0 when the two normalise identically
    """
    a, b = normalize(left, field_type), normalize(right, field_type)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def completeness(value: Any, field_type: str = FIELD_TYPE_TEXT) -> tuple[int, int]:
    """Rank how much a value says, for picking the survivor of a collapse.

    More tokens beats fewer, then longer beats shorter. A middle initial makes
    "Russell H Jurney" the fullest way the group writes that name, and that is
    the value the merged record should carry.

    Tokens are counted on the normalised text and length on the raw text, on
    purpose. Normalising is what lets two spellings be compared at all, but it
    also expands "St" into "street", and ranking length after that would call
    the abbreviation as complete as the word it stands for.

    Parameters
    ----------
    value : Any
        Raw field value
    field_type : str
        Inferred type, which decides the normalisation applied first

    Returns
    -------
    tuple[int, int]
        Normalised token count, then raw character count
    """
    return (len(normalize(value, field_type).split()), len(str(value).strip()))


def merge_values(values: list[Any], field_type: str = FIELD_TYPE_TEXT) -> list[Any]:
    """Combine every value one field held across a group of merged records.

    Values that are spellings of one another collapse to the most complete of
    them; values that genuinely differ are all kept. The result is ordered most
    complete first, so a consumer that wants a single value can take the head
    and get the one this field's type says is the fullest.

    Parameters
    ----------
    values : list[Any]
        Every value the field held, in record order, blanks included
    field_type : str
        Type from ``serf.analyze.field_detection.detect_field_type``

    Returns
    -------
    list[Any]
        Distinct values, most complete first, ties broken by text so the output
        does not depend on the order the records arrived in
    """
    populated = [value for value in values if normalize(value, field_type)]
    if not populated:
        return []

    policy = policy_for(field_type)

    if policy.dedupe == DEDUPE_NONE:
        seen: dict[str, Any] = {}
        for value in populated:
            seen.setdefault(str(value).strip(), value)
        return list(seen.values())

    clusters: list[list[Any]] = []
    keys: list[str] = []
    for value in populated:
        key = normalize(value, field_type)
        if key in keys:
            clusters[keys.index(key)].append(value)
            continue
        keys.append(key)
        clusters.append([value])

    if policy.dedupe == DEDUPE_FUZZY:
        clusters = _collapse_similar(clusters, field_type, policy.threshold)

    def rank(item: Any) -> tuple[int, int]:
        return completeness(item, field_type)

    survivors = [max(cluster, key=rank) for cluster in clusters]
    return sorted(survivors, key=lambda item: (-rank(item)[0], -rank(item)[1], str(item)))


def _collapse_similar(
    clusters: list[list[Any]], field_type: str, threshold: float
) -> list[list[Any]]:
    """Join clusters whose values are similar enough to be one value.

    Linkage is single, not complete: a value joins a cluster when it is close
    enough to any member. "Russ Journey" reaches "Russell H Jurney" only
    through "Russell Jurney", and all three are one name.

    Parameters
    ----------
    clusters : list[list[Any]]
        Clusters of exactly-equal values, in first-seen order
    field_type : str
        Inferred type, which decides the normalisation applied first
    threshold : float
        Similarity at or above which two values are treated as one

    Returns
    -------
    list[list[Any]]
        Clusters after joining, in first-seen order
    """
    parent = list(range(len(clusters)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if find(i) == find(j):
                continue
            if any(
                similarity(left, right, field_type) >= threshold
                for left in clusters[i]
                for right in clusters[j]
            ):
                parent[max(find(i), find(j))] = min(find(i), find(j))

    joined: dict[int, list[Any]] = {}
    for index, cluster in enumerate(clusters):
        joined.setdefault(find(index), []).extend(cluster)
    return [joined[key] for key in sorted(joined)]
