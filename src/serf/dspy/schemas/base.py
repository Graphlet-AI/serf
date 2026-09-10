"""Shared Pydantic base types for per-dataset entity matching schemas.

Every benchmark match task joins two distinct sources whose records mean
different things and whose attributes are populated differently. These base
types hold what all of them share: the identity of one record on one side of
the join, and the outer candidate pair the LLM emits for a matched record pair.

Concrete subclasses live in one module per dataset and carry the source-specific
fields plus the field descriptions that give the LLM the domain knowledge the
entity resolution literature documents for that task.
"""

from typing import Annotated, Any, Self

from pydantic import BaseModel, BeforeValidator, Field

from serf.dspy.types import Entity
from serf.logs import get_logger

logger = get_logger(__name__)

LEFT_ATTRIBUTE_PREFIX = "l_"
RIGHT_ATTRIBUTE_PREFIX = "r_"

SIDE_LEFT = "left"
SIDE_RIGHT = "right"
SIDE_UNKNOWN = "unknown"

_SOURCE_ID_COLUMN = "id"
_IDENTITY_FIELDS = ("record_id", "source_id", "source_name")
_NUMERIC_NOISE_CHARACTERS = "$€£,"


def strip_side_prefix(attributes: dict[str, Any]) -> dict[str, Any]:
    """Drop the ``l_``/``r_`` prefix the benchmark loader adds to source columns.

    ``BenchmarkDataset.to_entities`` prefixes every source column so left and
    right attributes never collide inside one ``Entity``. Typed side models are
    named after the raw source columns, so the prefix has to come back off.

    Parameters
    ----------
    attributes : dict[str, Any]
        Entity attributes, possibly prefixed

    Returns
    -------
    dict[str, Any]
        Attributes keyed by their original source column names
    """
    stripped: dict[str, Any] = {}
    for key, value in attributes.items():
        if key.startswith((LEFT_ATTRIBUTE_PREFIX, RIGHT_ATTRIBUTE_PREFIX)):
            stripped[key[len(LEFT_ATTRIBUTE_PREFIX) :]] = value
        else:
            stripped[key] = value
    return stripped


def entity_side(entity: Entity) -> str:
    """Return which side of the join an entity came from.

    Parameters
    ----------
    entity : Entity
        Entity produced by ``BenchmarkDataset.to_entities``

    Returns
    -------
    str
        ``left``, ``right``, or ``unknown`` when no prefixed attribute is present
    """
    for key in entity.attributes:
        if key.startswith(LEFT_ATTRIBUTE_PREFIX):
            return SIDE_LEFT
        if key.startswith(RIGHT_ATTRIBUTE_PREFIX):
            return SIDE_RIGHT
    return SIDE_UNKNOWN


def _clean_text(value: Any) -> str:
    """Coerce a source value into a stripped string.

    Parameters
    ----------
    value : Any
        Raw CSV or model-emitted value

    Returns
    -------
    str
        Stripped text, empty when the value is missing
    """
    if value is None:
        return ""
    return str(value).strip()


def _parse_optional_int(value: Any) -> int | None:
    """Parse a possibly dirty source value as an integer.

    Parameters
    ----------
    value : Any
        Raw CSV or model-emitted value

    Returns
    -------
    int | None
        Parsed integer, or None when the value is blank or unparseable
    """
    number = _parse_optional_float(value)
    return None if number is None else int(number)


def _parse_optional_float(value: Any) -> float | None:
    """Parse a possibly dirty source value as a float.

    Blank strings, currency symbols, and thousands separators are common in the
    benchmark CSVs, and an unparseable value is treated as missing rather than
    as a validation error, because these sources are documented as dirty.

    Parameters
    ----------
    value : Any
        Raw CSV or model-emitted value

    Returns
    -------
    float | None
        Parsed float, or None when the value is blank or unparseable
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = "".join(char for char in str(value) if char not in _NUMERIC_NOISE_CHARACTERS).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


SourceText = Annotated[str, BeforeValidator(_clean_text)]
"""Text column from a source CSV; missing values become the empty string."""

SourceYear = Annotated[int | None, BeforeValidator(_parse_optional_int)]
"""Year column; blank or unparseable values become None."""

SourcePrice = Annotated[float | None, BeforeValidator(_parse_optional_float)]
"""Price column; currency symbols are stripped and dirty values become None."""


def field_guide(model: type[BaseModel]) -> str:
    """Render a model's field descriptions as instruction text.

    DSPy 3.3.1 deliberately emits no type description for input fields, and
    ``XMLAdapter`` renders nested output models as a bare tag skeleton, so
    ``Field(description=...)`` never reaches the prompt on its own. Rendering the
    descriptions into the signature docstring keeps the typed models as the single
    source of truth while still telling the LM what each column means.

    Parameters
    ----------
    model : type[BaseModel]
        Model whose fields should be described

    Returns
    -------
    str
        One indented bullet per field, ready to embed in a signature docstring
    """
    lines = []
    for name, field in model.model_fields.items():
        description = " ".join((field.description or "").split())
        lines.append(f"    - {name}: {description}")
    return "\n".join(lines)


class EntitySide(BaseModel):
    """One record on one side of a two-source entity matching task.

    Parameters
    ----------
    record_id : int
        Integer id of this record within the block being matched
    source_id : str
        Primary key of this record in its own source table
    source_name : str
        Name of the source this record came from, used as a discriminator
    """

    record_id: int = Field(
        description=(
            "Integer id of this record inside the block. Copy it verbatim into "
            "the output; it is the only way to identify the record."
        )
    )
    source_id: SourceText = Field(
        default="",
        description="Primary key of this record in its own source table",
    )
    source_name: SourceText = Field(
        default="",
        description="Name of the source system this record came from",
    )

    @classmethod
    def source_columns(cls) -> list[str]:
        """Return the source column names this side model expects.

        Returns
        -------
        list[str]
            Field names that map one-to-one onto source CSV columns
        """
        return [name for name in cls.model_fields if name not in _IDENTITY_FIELDS]

    @classmethod
    def from_entity(cls, entity: Entity) -> Self:
        """Build a typed side record from a pipeline entity.

        Parameters
        ----------
        entity : Entity
            Entity whose attributes carry the prefixed source columns

        Returns
        -------
        Self
            Typed record for this side of the join
        """
        columns = strip_side_prefix(entity.attributes)
        values: dict[str, Any] = {
            "record_id": entity.id,
            "source_id": str(columns.get(_SOURCE_ID_COLUMN, "")),
        }
        for name in cls.source_columns():
            values[name] = columns.get(name, "")
        return cls.model_validate(values)


class EntityMatchCandidate(BaseModel):
    """A pair of records, one from each source, judged by the LLM.

    Parameters
    ----------
    left : EntitySide
        Record from the left source of the match task
    right : EntitySide
        Record from the right source of the match task
    is_match : bool
        Whether the two records describe the same real-world entity
    confidence : float
        Confidence in the decision, between 0 and 1
    justification : str
        Short explanation naming the evidence that drove the decision
    """

    left: EntitySide = Field(description="Record from the left source, copied from the input")
    right: EntitySide = Field(description="Record from the right source, copied from the input")
    is_match: bool = Field(description="True when both records describe the same real-world entity")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the decision between 0.0 and 1.0",
    )
    justification: str = Field(
        default="",
        description=(
            "One or two sentences naming the specific field values that decided "
            "the match, so the decision explains itself"
        ),
    )
