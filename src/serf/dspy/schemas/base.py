"""Shared Pydantic base types for per-dataset entity matching schemas.

Every benchmark match task joins two distinct sources whose records mean
different things and whose attributes are populated differently. These base
types hold what all of them share: the identity of one record on one side of
the join, and the outer candidate pair the LLM emits for a matched record pair.

Concrete subclasses live in one module per dataset and carry the source-specific
fields plus the field descriptions that give the LLM the domain knowledge the
entity resolution literature documents for that task.
"""

from typing import Annotated, Any, Self, cast

import dspy
from pydantic import BaseModel, BeforeValidator, Field, create_model

from serf.config import config
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

_DEFAULT_SINGLE_INSTRUCTION = (
    "Compare values across the matched records and select the single best, "
    "most complete, and accurate value."
)
_DEFAULT_LIST_INSTRUCTION = (
    "Compare values across the matched records and deduplicate into a list "
    "of individual values, collapsing formatting variants while keeping "
    "distinct values separate."
)


def field_merge_instruction(
    field_type: str = "text",
    *,
    cardinality: str = "list",
    dedupe: str | None = None,
    description: str = "",
) -> str:
    """Build the field-level LLM merge instruction for a semantic type and cardinality.

    Parameters
    ----------
    field_type : str
        Semantic type of the field (``name``, ``address``, ``url``, ``email``,
        ``phone``, ``identifier``, ``date``, ``numeric``, ``text``)
    cardinality : str
        ``list`` to deduplicate into a list of individual values, or ``single``
        (or ``singular``) to select the single best value
    dedupe : str | None
        Optional deduplication strategy override (``fuzzy``, ``exact``, ``none``)
    description : str
        Domain description of the field from the entity schema

    Returns
    -------
    str
        Instruction text for the DSPy field annotation
    """
    normalized_cardinality = (
        "single" if str(cardinality).strip().lower() in {"single", "singular"} else "list"
    )
    settings: dict[str, Any] = config.get(f"merge.semantics.{field_type}", {}) or {}
    if normalized_cardinality == "single":
        instruction = str(settings.get("single_instruction") or _DEFAULT_SINGLE_INSTRUCTION)
    elif dedupe == "none":
        instruction = str(
            config.get("merge.semantics.text.list_instruction", _DEFAULT_LIST_INSTRUCTION)
        )
    elif dedupe == "exact" and settings.get("dedupe") == "fuzzy":
        instruction = str(config.get("merge.exact_list_instruction", _DEFAULT_LIST_INSTRUCTION))
    else:
        instruction = str(settings.get("list_instruction") or _DEFAULT_LIST_INSTRUCTION)

    clean_desc = " ".join(description.split()).rstrip(". ")
    if clean_desc:
        return f"{clean_desc}. {instruction}"
    return instruction


def merge_output_field(
    field_type: str = "text",
    *,
    cardinality: str = "list",
    dedupe: str | None = None,
    description: str = "",
    required: bool = False,
) -> Any:
    """Create a DSPy ``OutputField`` carrying field-level merge semantics.

    Parameters
    ----------
    field_type : str
        Semantic type of the field
    cardinality : str
        ``list`` or ``single``
    dedupe : str | None
        Optional deduplication strategy override
    description : str
        Domain description of the field
    required : bool
        Whether a singular field is required

    Returns
    -------
    Any
        Configured ``dspy.OutputField``
    """
    full_description = field_merge_instruction(
        field_type,
        cardinality=cardinality,
        dedupe=dedupe,
        description=description,
    )
    normalized_cardinality = (
        "single" if str(cardinality).strip().lower() in {"single", "singular"} else "list"
    )
    if normalized_cardinality == "single":
        if required:
            return dspy.OutputField(desc=full_description, description=full_description)
        return dspy.OutputField(
            default=None,
            desc=full_description,
            description=full_description,
        )
    return dspy.OutputField(
        default_factory=list,
        desc=full_description,
        description=full_description,
    )


def annotated_merge_field(
    scalar_type: type = str,
    field_type: str = "text",
    *,
    cardinality: str = "list",
    dedupe: str | None = None,
    description: str = "",
    required: bool = False,
) -> Any:
    """Build a ``typing.Annotated`` DSPy output field type with merge semantics.

    Parameters
    ----------
    scalar_type : type
        Scalar Python type of a single field value (``str``, ``int``, ``float``, ``bool``)
    field_type : str
        Semantic field type (``name``, ``address``, ``email``, etc.)
    cardinality : str
        ``list`` for deduplicated lists of individual values, ``single`` for a single best value
    dedupe : str | None
        Optional deduplication strategy override
    description : str
        Domain description of the field
    required : bool
        Whether a singular field is required

    Returns
    -------
    Any
        ``Annotated[T, dspy.OutputField(...)]`` ready for Pydantic models and DSPy signatures
    """
    field_info = merge_output_field(
        field_type,
        cardinality=cardinality,
        dedupe=dedupe,
        description=description,
        required=required,
    )
    normalized_cardinality = (
        "single" if str(cardinality).strip().lower() in {"single", "singular"} else "list"
    )
    if normalized_cardinality == "single":
        target_type: Any = scalar_type if required else (scalar_type | None)
        return Annotated[target_type, field_info]
    return Annotated[list[scalar_type], field_info]  # type: ignore[valid-type]


def build_merge_signature(
    name: str,
    field_annotations: dict[str, Any],
    *,
    docstring: str = "",
) -> type[dspy.Signature]:
    """Build a DSPy ``Signature`` whose output fields carry per-field merge annotations.

    Parameters
    ----------
    name : str
        Signature class name
    field_annotations : dict[str, Any]
        Mapping of field name to ``Annotated[..., dspy.OutputField(...)]`` type
    docstring : str
        Instructions for the signature

    Returns
    -------
    type[dspy.Signature]
        Generated DSPy ``Signature`` class
    """
    definitions: dict[str, Any] = {
        "records": (
            list[dict[str, Any]],
            dspy.InputField(
                desc=(
                    "Records that denote the same real-world entity. Compare field values "
                    "across all records and merge each field according to its output field "
                    "annotation."
                )
            ),
        ),
        **field_annotations,
    }
    instructions = docstring or (
        "Merge matched records that denote the same real-world entity into one canonical record. "
        "Rely on the LLM and the field-level output annotations to compare values across records, "
        "selecting the single best value for singular fields and deduplicating list fields into "
        "lists of individual values."
    )
    return cast(
        type[dspy.Signature],
        create_model(
            name,
            __base__=dspy.Signature,
            __doc__=instructions,
            **definitions,
        ),
    )


SingularName = Annotated[str, merge_output_field("name", cardinality="single", required=True)]
SingularAddress = Annotated[str, merge_output_field("address", cardinality="single", required=True)]
SingularUrl = Annotated[str, merge_output_field("url", cardinality="single", required=True)]
SingularEmail = Annotated[str, merge_output_field("email", cardinality="single", required=True)]
SingularPhone = Annotated[str, merge_output_field("phone", cardinality="single", required=True)]
SingularIdentifier = Annotated[
    str, merge_output_field("identifier", cardinality="single", required=True)
]
SingularDate = Annotated[str, merge_output_field("date", cardinality="single", required=True)]
SingularNumeric = Annotated[
    float | None, merge_output_field("numeric", cardinality="single", required=False)
]
SingularText = Annotated[str, merge_output_field("text", cardinality="single", required=True)]

NameList = Annotated[list[str], merge_output_field("name", cardinality="list")]
AddressList = Annotated[list[str], merge_output_field("address", cardinality="list")]
UrlList = Annotated[list[str], merge_output_field("url", cardinality="list")]
EmailList = Annotated[list[str], merge_output_field("email", cardinality="list")]
PhoneList = Annotated[list[str], merge_output_field("phone", cardinality="list")]
IdentifierList = Annotated[list[str], merge_output_field("identifier", cardinality="list")]
DateList = Annotated[list[str], merge_output_field("date", cardinality="list")]
NumericList = Annotated[list[float], merge_output_field("numeric", cardinality="list")]
TextList = Annotated[list[str], merge_output_field("text", cardinality="list")]


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
        extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        raw_desc = field.description or str(extra.get("desc", ""))
        description = " ".join(raw_desc.split())
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


class ResolvedEntity(BaseModel):
    """One real-world entity, named by the records in a block that denote it.

    The matcher returns a partition of the block rather than a list of pairwise
    verdicts. A pair list can contradict itself - a matched to b, b matched to
    c, a not matched to c - and something downstream then has to decide what
    the model meant. A partition cannot: the grouping is the answer. It also
    makes a dropped record visible, because a record missing from every group
    is a hole in the cover rather than an absent pair nobody was counting.

    Only ids cross the wire. Which value of a field the merged record should
    carry is decided by ``serf.merge.canonical`` from the field's type, not by
    the model, so there is nothing here for it to copy wrongly.

    Parameters
    ----------
    record_ids : list[int]
        Every record_id in this block that denotes this one entity. A record
        matching nothing forms a group of one.
    justification : str
        Short explanation naming the evidence that grouped these records
    """

    record_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Every record_id from the block that denotes this one real-world "
            "entity. A record that matches nothing gets a group of its own. "
            "Copy the ids exactly; never invent or renumber one."
        ),
    )
    justification: str = Field(
        default="",
        description=(
            "One or two sentences naming the specific field values that put "
            "these records together, so the decision explains itself. Leave "
            "empty for a group of one."
        ),
    )


class EntityMatchCandidate(BaseModel):
    """A pair of records, one from each source, judged by the LLM.

    Superseded as the matcher's output contract by ``ResolvedEntity``; kept
    because the per-dataset subclasses document each task's record shape.

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
