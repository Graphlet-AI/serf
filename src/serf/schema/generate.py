"""Turn an entity schema into the Pydantic classes a DSPy signature needs.

Two classes come out of one schema. The source model is what the LLM reads: one
record as it arrives, every field a scalar. The canonical model is what a merge
produces: the same fields, every one of them a list, plus the lineage.

Both carry the schema's field descriptions, because that is how the domain
knowledge reaches the prompt. DSPy 3.3.1 emits no type description for input
fields and ``XMLAdapter`` renders a nested output model as a bare tag skeleton,
so ``serf.dspy.schemas.base.field_guide`` renders them into the signature
docstring instead.
"""

from typing import Annotated, Any, cast

import dspy
from pydantic import BaseModel, Field, create_model

from serf.dspy.schemas.base import (
    EntitySide,
    build_merge_signature,
    field_guide,
)
from serf.logs import get_logger
from serf.schema.spec import EntitySchema, FieldSpec

logger = get_logger(__name__)

_source_cache: dict[str, type[EntitySide]] = {}
_canonical_cache: dict[str, type[BaseModel]] = {}
_merge_signature_cache: dict[str, type[dspy.Signature]] = {}
_match_and_merge_signature_cache: dict[str, type[dspy.Signature]] = {}


def merge_field_annotation(field: FieldSpec) -> Any:
    """Return the DSPy ``Annotated[..., dspy.OutputField(...)]`` annotation for a field.

    Parameters
    ----------
    field : FieldSpec
        Schema field specification

    Returns
    -------
    Any
        ``Annotated`` type combining the field's scalar/list Python type with
        a ``dspy.OutputField`` describing its field-level merge semantics
    """
    return field.dspy_annotation()


class CanonicalEntity(BaseModel):
    """Base for a generated canonical record: a uuid and the lineage behind it.

    Parameters
    ----------
    uuid : str
        Identity of this entity, minted when the record is a merge
    source_uuids : list[str]
        Every input uuid this record absorbed, transitively, sorted
    """

    uuid: str = Field(
        description=(
            "Identity of this resolved entity. A merge of several records gets "
            "a new uuid that none of them used."
        )
    )
    source_uuids: list[str] = Field(
        default_factory=list,
        description=(
            "Every input record uuid this entity stands for, including uuids "
            "its inputs had already absorbed. Never drop one."
        ),
    )


def source_model(schema: EntitySchema) -> type[EntitySide]:
    """Build the Pydantic class for one source record of this entity type.

    Parameters
    ----------
    schema : EntitySchema
        Schema to generate from

    Returns
    -------
    type[EntitySide]
        Subclass of ``EntitySide``, so it keeps ``record_id`` and the
        ``from_entity`` adapter the matcher uses
    """
    key = schema.model_dump_json()
    cached = _source_cache.get(key)
    if cached is not None:
        return cached

    definitions: dict[str, Any] = {}
    for field in schema.fields:
        scalar = field.scalar_type()
        if field.required:
            definitions[field.name] = (scalar, Field(description=field.description))
        else:
            definitions[field.name] = (
                scalar | None,
                Field(default=None, description=field.description),
            )

    model = create_model(schema.entity, __base__=EntitySide, **definitions)
    model.__doc__ = schema.description or f"One {schema.entity} record from a source."
    _source_cache[key] = model
    return model


def canonical_model(schema: EntitySchema) -> type[BaseModel]:
    """Build the Pydantic class for a resolved record of this entity type.

    Each field is defined with its DSPy ``Annotated[..., dspy.OutputField(...)]``
    merge annotation: list fields hold ``list[scalar]`` and instruct the LLM to
    deduplicate values into a list of individual items; singular fields hold a
    single ``scalar`` (or ``scalar | None`` when optional) and instruct the LLM
    to select the single best value across the merged records.

    Parameters
    ----------
    schema : EntitySchema
        Schema to generate from

    Returns
    -------
    type[BaseModel]
        Subclass of ``CanonicalEntity`` with one DSPy-annotated field per schema field
    """
    key = schema.model_dump_json()
    cached = _canonical_cache.get(key)
    if cached is not None:
        return cached

    definitions: dict[str, Any] = {
        field.name: merge_field_annotation(field) for field in schema.fields
    }

    model = create_model(
        f"Canonical{schema.entity}",
        __base__=CanonicalEntity,
        **definitions,
    )
    model.__doc__ = (
        f"{schema.description} " if schema.description else ""
    ) + "Every field carries its DSPy merge annotation for singular or list resolution."
    _canonical_cache[key] = model
    return model


def merge_signature(schema: EntitySchema) -> type[dspy.Signature]:
    """Build the DSPy ``Signature`` that merges a group of matched records for this schema.

    Every field in ``schema.fields`` becomes a top-level DSPy output field
    annotated with ``merge_field_annotation(field)``, so the LLM compares field
    values across the matched records and resolves both singular fields (single
    best value) and list fields (deduplicated lists of individual values such
    as addresses or names) according to their annotations.

    Parameters
    ----------
    schema : EntitySchema
        Schema to generate from

    Returns
    -------
    type[dspy.Signature]
        Subclass of ``dspy.Signature`` for field-level LLM merging
    """
    key = schema.model_dump_json()
    cached = _merge_signature_cache.get(key)
    if cached is not None:
        return cached

    canonical = canonical_model(schema)
    guide = field_guide(canonical)
    header = f"{schema.description.strip()}\n\n" if schema.description else ""
    docstring = (
        f"{header}Merge matched {schema.entity} records that denote the same real-world entity "
        "into one canonical record. Rely on the LLM to compare values across records at the "
        "field level and follow each output field's merge annotation:\n"
        f"{guide}"
    )
    sig = build_merge_signature(
        f"Merge{schema.entity}",
        schema.dspy_field_annotations(),
        docstring=docstring,
    )
    _merge_signature_cache[key] = sig
    return sig


def match_and_merge_signature(schema: EntitySchema) -> type[dspy.Signature]:
    """Build the DSPy ``Signature`` that matches a block and merges each group's fields.

    Parameters
    ----------
    schema : EntitySchema
        Schema to generate from

    Returns
    -------
    type[dspy.Signature]
        Subclass of ``dspy.Signature`` taking ``list[source_model(schema)]`` and
        producing ``list[canonical_model(schema)]``
    """
    key = schema.model_dump_json()
    cached = _match_and_merge_signature_cache.get(key)
    if cached is not None:
        return cached

    source = source_model(schema)
    canonical = canonical_model(schema)
    header = f"{schema.description.strip()}\n\n" if schema.description else ""
    docstring = (
        f"{header}Examine all {schema.entity} records in the block, identify which records "
        "denote the same real-world entity, and merge each matched group into one canonical "
        "record. Rely on the LLM to make field-level comparisons at the time of match and "
        "merge.\n\n"
        f"Input {schema.entity} fields:\n{field_guide(source)}\n\n"
        f"Output Canonical{schema.entity} fields and merge semantics:\n{field_guide(canonical)}"
    )
    definitions: dict[str, Any] = {
        "records": Annotated[
            list[source],  # type: ignore[valid-type]
            dspy.InputField(
                desc=f"Source {schema.entity} records in this block to compare, match, and merge"
            ),
        ],
        "resolved": Annotated[
            list[canonical],  # type: ignore[valid-type]
            dspy.OutputField(
                desc=(
                    f"Resolved canonical {schema.entity} records with fields merged according "
                    "to their field-level annotations and source_uuids lineage preserved"
                )
            ),
        ],
    }
    sig = cast(
        type[dspy.Signature],
        create_model(
            f"MatchAndMerge{schema.entity}",
            __base__=dspy.Signature,
            __doc__=docstring,
            **definitions,
        ),
    )
    _match_and_merge_signature_cache[key] = sig
    return sig


def clear_cache() -> None:
    """Drop the generated-class cache.

    Generated classes are cached so a signature built twice from one schema
    gets the same type, which DSPy compares by identity. Tests that rebuild a
    schema under one name need the cache cleared between cases.
    """
    _source_cache.clear()
    _canonical_cache.clear()
    _merge_signature_cache.clear()
    _match_and_merge_signature_cache.clear()
