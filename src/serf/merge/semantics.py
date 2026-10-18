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

from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, cast

import dspy

from serf.config import config
from serf.dspy.adapter import RepairingXMLAdapter
from serf.dspy.lm import create_lm
from serf.dspy.schemas.base import (
    annotated_merge_field,
    build_merge_signature,
    field_merge_instruction,
)
from serf.dspy.signatures import DeduplicateListValues, SelectBestValue
from serf.logs import get_logger

logger = get_logger(__name__)

DEDUPE_FUZZY = "fuzzy"
DEDUPE_EXACT = "exact"
DEDUPE_NONE = "none"
DEDUPE_STRATEGIES = (DEDUPE_FUZZY, DEDUPE_EXACT, DEDUPE_NONE)

CARDINALITY_LIST = "list"
CARDINALITY_SINGLE = "single"
CARDINALITIES = (CARDINALITY_LIST, CARDINALITY_SINGLE)

FIELD_TYPE_NAME = "name"
FIELD_TYPE_ADDRESS = "address"
FIELD_TYPE_URL = "url"
FIELD_TYPE_EMAIL = "email"
FIELD_TYPE_PHONE = "phone"
FIELD_TYPE_IDENTIFIER = "identifier"
FIELD_TYPE_DATE = "date"
FIELD_TYPE_NUMERIC = "numeric"
FIELD_TYPE_TEXT = "text"
FIELD_TYPES = (
    FIELD_TYPE_NAME,
    FIELD_TYPE_ADDRESS,
    FIELD_TYPE_URL,
    FIELD_TYPE_EMAIL,
    FIELD_TYPE_PHONE,
    FIELD_TYPE_IDENTIFIER,
    FIELD_TYPE_DATE,
    FIELD_TYPE_NUMERIC,
    FIELD_TYPE_TEXT,
)

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
    cardinality : str
        ``list`` to deduplicate values into a list of individual values, or
        ``single`` to select one canonical value for a singular field
    """

    dedupe: str = DEDUPE_EXACT
    threshold: float = _DEFAULT_THRESHOLD
    cardinality: str = CARDINALITY_LIST


def normalize_cardinality(value: str | None) -> str:
    """Normalise a cardinality name, accepting ``singular`` as ``single``.

    Parameters
    ----------
    value : str | None
        Raw cardinality string

    Returns
    -------
    str
        ``list`` or ``single``
    """
    if value is None:
        return CARDINALITY_LIST
    cleaned = str(value).strip().lower()
    if cleaned == "singular":
        return CARDINALITY_SINGLE
    return cleaned


def policy_for(field_type: str, cardinality: str | None = None) -> MergePolicy:
    """Return the configured merge policy for an inferred field type.

    Parameters
    ----------
    field_type : str
        Type from ``serf.analyze.field_detection.detect_field_type``
    cardinality : str | None
        Optional override for ``list`` or ``single`` cardinality

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
    resolved_cardinality = normalize_cardinality(
        cardinality or settings.get("cardinality", CARDINALITY_LIST)
    )
    if resolved_cardinality not in CARDINALITIES:
        logger.warning(
            f"Unknown cardinality '{resolved_cardinality}' for field type '{field_type}'; "
            f"using '{CARDINALITY_LIST}'. Valid cardinalities: {CARDINALITIES}"
        )
        resolved_cardinality = CARDINALITY_LIST
    default_threshold = float(config.get("matcher.similarity.min_threshold", _DEFAULT_THRESHOLD))
    return MergePolicy(
        dedupe=dedupe,
        threshold=float(settings.get("threshold", default_threshold)),
        cardinality=resolved_cardinality,
    )


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


def flatten_values(values: Iterable[Any]) -> list[Any]:
    """Flatten scalar and nested list/tuple/set values into individual items.

    Parameters
    ----------
    values : Iterable[Any]
        Raw field values, some of which may already be lists, tuples, or sets

    Returns
    -------
    list[Any]
        Individual scalar items in encounter order
    """
    flat: list[Any] = []
    for item in values:
        if isinstance(item, (list, tuple, set)):
            flat.extend(flatten_values(item))
        else:
            flat.append(item)
    return flat


def merge_values(
    values: Iterable[Any],
    field_type: str = FIELD_TYPE_TEXT,
    policy: MergePolicy | None = None,
    *,
    field_name: str = "",
    field_description: str = "",
    use_llm: bool = False,
    lm: Any = None,
) -> list[Any]:
    """Combine every value one field held across a group of merged records.

    When ``use_llm`` is True (or ``lm`` is supplied) and multiple distinct
    values exist, the LLM compares and deduplicates the values into a list of
    individual values via the ``DeduplicateListValues`` DSPy signature using
    the field's DSPy merge annotation. Otherwise, deterministic normalisation
    and clustering order the distinct values most complete first.

    Parameters
    ----------
    values : Iterable[Any]
        Every value the field held, in record order, blanks and nested lists included
    field_type : str
        Type from ``serf.analyze.field_detection.detect_field_type``
    policy : MergePolicy | None
        Policy to apply, overriding the one configured for ``field_type``.
        A schema passes its per-field override here.
    field_name : str
        Name of the field, passed to the LLM prompt when ``use_llm`` is True
    field_description : str
        Description of the field, passed to the LLM prompt when ``use_llm`` is True
    use_llm : bool
        Deduplicate candidate values into a list of individual values via the LLM
    lm : Any
        Optional DSPy LM instance overriding ``dspy.settings.lm`` / ``create_lm()``

    Returns
    -------
    list[Any]
        Distinct values, most complete first, ties broken by text so the output
        does not depend on the order the records arrived in
    """
    populated = [value for value in flatten_values(values) if normalize(value, field_type)]
    if not populated:
        return []

    policy = policy or policy_for(field_type)

    if use_llm or lm is not None:
        distinct_by_text: dict[str, Any] = {}
        for value in populated:
            distinct_by_text.setdefault(str(value).strip(), value)
        if len(distinct_by_text) == 1:
            return [next(iter(distinct_by_text.values()))]
        try:
            active_lm = lm or getattr(dspy.settings, "lm", None) or create_lm(role="student")
            instruction = field_merge_instruction(
                field_type,
                cardinality=CARDINALITY_LIST,
                dedupe=policy.dedupe,
                description=field_description,
            )
            predictor = cast(dspy.Predict, dspy.Predict(DeduplicateListValues))
            with dspy.context(lm=active_lm, adapter=RepairingXMLAdapter()):
                prediction = predictor(
                    field_name=field_name or field_type,
                    field_type=field_type,
                    field_description=instruction,
                    candidate_values=list(distinct_by_text.keys()),
                )
            raw_items = getattr(prediction, "deduplicated_values", None)
            if isinstance(raw_items, list):
                resolved_list: list[Any] = []
                for item in raw_items:
                    text = str(item).strip()
                    if not text:
                        continue
                    resolved_list.append(distinct_by_text.get(text, item))
                if resolved_list:
                    return resolved_list
        except Exception as e:
            logger.warning(
                f"LLM list deduplication failed for field '{field_name or field_type}', "
                f"falling back to deterministic merge: {e}"
            )

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


def select_best_value(
    values: Iterable[Any],
    field_type: str = FIELD_TYPE_TEXT,
    policy: MergePolicy | None = None,
    *,
    field_name: str = "",
    field_description: str = "",
    use_llm: bool = False,
    lm: Any = None,
) -> Any:
    """Select the single best canonical value for a singular field.

    When ``use_llm`` is True (or ``lm`` is supplied) and more than one distinct
    value is present, the choice goes through the ``SelectBestValue`` DSPy
    signature so the language model picks the most complete and accurate form.
    When all values are identical, or when ``use_llm`` is False, the most
    complete deduplicated value is returned directly.

    Parameters
    ----------
    values : Iterable[Any]
        Candidate values from the records being merged
    field_type : str
        Semantic field type
    policy : MergePolicy | None
        Optional per-field merge policy
    field_name : str
        Name of the field, passed to the LLM prompt when ``use_llm`` is True
    field_description : str
        Description of the field, passed to the LLM prompt when ``use_llm`` is True
    use_llm : bool
        Use the LLM via DSPy ``SelectBestValue`` when multiple distinct values exist
    lm : Any
        Optional DSPy LM instance overriding ``dspy.settings.lm`` / ``create_lm()``

    Returns
    -------
    Any
        The selected canonical scalar value, or ``None`` if no non-blank value exists
    """
    populated = [value for value in flatten_values(values) if normalize(value, field_type)]
    if not populated:
        return None

    distinct_by_text: dict[str, Any] = {}
    for value in populated:
        distinct_by_text.setdefault(str(value).strip(), value)

    if len(distinct_by_text) == 1:
        return next(iter(distinct_by_text.values()))

    merged = merge_values(populated, field_type, policy)
    candidates_for_fallback = merged if merged else list(distinct_by_text.values())
    fallback = min(
        candidates_for_fallback,
        key=lambda item: (
            -completeness(item, field_type)[0],
            -completeness(item, field_type)[1],
            str(item),
        ),
    )

    if not use_llm and lm is None:
        return fallback

    try:
        active_lm = lm or getattr(dspy.settings, "lm", None) or create_lm(role="student")
        instruction = field_merge_instruction(
            field_type,
            cardinality=CARDINALITY_SINGLE,
            dedupe=policy.dedupe if policy is not None else None,
            description=field_description,
        )
        predictor = cast(dspy.Predict, dspy.Predict(SelectBestValue))
        with dspy.context(lm=active_lm, adapter=RepairingXMLAdapter()):
            prediction = predictor(
                field_name=field_name or field_type,
                field_type=field_type,
                field_description=instruction,
                candidate_values=list(distinct_by_text.keys()),
            )
        chosen_text = str(getattr(prediction, "best_value", "") or "").strip()
        if chosen_text in distinct_by_text:
            return distinct_by_text[chosen_text]
        chosen_norm = normalize(chosen_text, field_type)
        if chosen_norm:
            for candidate_text, original_value in distinct_by_text.items():
                if normalize(candidate_text, field_type) == chosen_norm:
                    return original_value
            if isinstance(fallback, str):
                return chosen_text
    except Exception as e:
        logger.warning(
            f"LLM value selection failed for singular field '{field_name or field_type}', "
            f"falling back to completeness ranking: {e}"
        )
    return fallback


def _extract_prediction_field(
    prediction: Any,
    key: str,
    *,
    is_singular: bool,
) -> Any:
    """Read one field from a DSPy prediction without triggering mock auto-attributes."""
    if isinstance(prediction, dict):
        return prediction.get(key)
    store = getattr(prediction, "_store", None)
    if isinstance(store, dict) and key in store:
        return store[key]
    obj_vars: dict[str, Any] = vars(prediction) if hasattr(prediction, "__dict__") else {}
    if key in obj_vars:
        return obj_vars[key]
    if is_singular and "best_value" in obj_vars:
        return obj_vars["best_value"]
    if not is_singular and "deduplicated_values" in obj_vars:
        return obj_vars["deduplicated_values"]
    return None


def merge_records_with_dspy(
    records: list[dict[str, Any]],
    ordered_fields: list[str],
    field_types: dict[str, str],
    policies: dict[str, MergePolicy],
    singular_fields: set[str],
    field_descriptions: dict[str, str],
    *,
    signature: type[dspy.Signature] | None = None,
    lm: Any = None,
) -> dict[str, Any]:
    """Merge field values across matched records using a DSPy signature annotated per field.

    Builds a DSPy signature whose output fields carry ``Annotated[..., dspy.OutputField(...)]``
    merge semantics for each singular and list field, and calls the LLM once to
    resolve all fields of the record group.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Input records denoting the same entity (reserved identity fields stripped)
    ordered_fields : list[str]
        Field names in first-seen order
    field_types : dict[str, str]
        Mapping of field name to semantic type
    policies : dict[str, MergePolicy]
        Mapping of field name to merge policy
    singular_fields : set[str]
        Field names that resolve to a single best value
    field_descriptions : dict[str, str]
        Mapping of field name to description
    signature : type[dspy.Signature] | None
        Optional pre-built DSPy signature from ``serf.schema.generate.merge_signature``
    lm : Any
        Optional DSPy LM instance

    Returns
    -------
    dict[str, Any]
        Field values resolved by the LLM
    """
    if not ordered_fields or len(records) <= 1:
        return {}

    sig = signature
    if sig is None:
        annotations: dict[str, Any] = {}
        for key in ordered_fields:
            ftype = field_types.get(key, FIELD_TYPE_TEXT)
            policy = policies.get(key) or policy_for(ftype)
            cardinality = CARDINALITY_SINGLE if key in singular_fields else CARDINALITY_LIST
            annotations[key] = annotated_merge_field(
                str,
                ftype,
                cardinality=cardinality,
                dedupe=policy.dedupe,
                description=field_descriptions.get(key, ""),
                required=False,
            )
        sig = build_merge_signature("MergeEntityFields", annotations)

    active_lm = lm or getattr(dspy.settings, "lm", None) or create_lm(role="student")
    predictor = cast(dspy.Predict, dspy.Predict(sig))
    with dspy.context(lm=active_lm, adapter=RepairingXMLAdapter()):
        prediction = predictor(records=records)

    resolved: dict[str, Any] = {}
    for key in ordered_fields:
        is_singular = key in singular_fields
        raw_val = _extract_prediction_field(prediction, key, is_singular=is_singular)
        if raw_val is None:
            continue
        if is_singular:
            text = str(raw_val).strip()
            if text:
                resolved[key] = raw_val if not isinstance(raw_val, str) else text
        elif isinstance(raw_val, list):
            cleaned = [
                item.strip() if isinstance(item, str) else item
                for item in flatten_values(raw_val)
                if normalize(item, field_types.get(key, FIELD_TYPE_TEXT))
            ]
            if cleaned:
                resolved[key] = cleaned
    return resolved


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
