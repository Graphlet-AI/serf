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

from typing import Any

from pydantic import BaseModel, Field, create_model

from serf.dspy.schemas.base import EntitySide
from serf.logs import get_logger
from serf.schema.spec import EntitySchema

logger = get_logger(__name__)

_source_cache: dict[str, type[EntitySide]] = {}
_canonical_cache: dict[str, type[BaseModel]] = {}


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

    Every field is a list, because a merged record can legitimately hold two
    states, and the list is ordered most complete first.

    Parameters
    ----------
    schema : EntitySchema
        Schema to generate from

    Returns
    -------
    type[BaseModel]
        Subclass of ``CanonicalEntity`` with one list-valued field per schema
        field
    """
    key = schema.model_dump_json()
    cached = _canonical_cache.get(key)
    if cached is not None:
        return cached

    definitions: dict[str, Any] = {}
    for field in schema.fields:
        scalar = field.scalar_type()
        description = field.description
        if description:
            description = f"{description} Most complete value first."
        definitions[field.name] = (
            list[scalar],  # type: ignore[valid-type]
            Field(default_factory=list, description=description),
        )

    model = create_model(
        f"Canonical{schema.entity}",
        __base__=CanonicalEntity,
        **definitions,
    )
    model.__doc__ = (
        f"{schema.description} " if schema.description else ""
    ) + "Every field holds a list, most complete value first."
    _canonical_cache[key] = model
    return model


def clear_cache() -> None:
    """Drop the generated-class cache.

    Generated classes are cached so a signature built twice from one schema
    gets the same type, which DSPy compares by identity. Tests that rebuild a
    schema under one name need the cache cleared between cases.
    """
    _source_cache.clear()
    _canonical_cache.clear()
