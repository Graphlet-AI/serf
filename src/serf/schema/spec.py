"""Declarative entity schemas: the YAML that drives typing and merging.

A schema says what fields an entity has, what kind of value each one holds, and
how values of that field combine when records merge. Field type detection can
guess most of this from the data, but only a schema can state the things the
values do not show: that a column of two-letter codes is a jurisdiction rather
than a name, that two part numbers differing in one character are different
products, that a column of prose must never be collapsed.

The schema is the single source of truth for three consumers: the Pydantic
classes that go into DSPy signatures, the field descriptions that reach the
prompt, and the merge policy canonicalization applies.
"""

import builtins
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from serf.logs import get_logger
from serf.merge.semantics import (
    DEDUPE_STRATEGIES,
    FIELD_TYPE_TEXT,
    MergePolicy,
    policy_for,
)

logger = get_logger(__name__)

FIELD_TYPES = (
    "name",
    "address",
    "url",
    "email",
    "phone",
    "identifier",
    "date",
    "numeric",
    "text",
)

PYTHON_TYPES: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}

# What a field of each semantic type usually holds, used when the schema does
# not say. Everything is text unless the type implies otherwise.
_DEFAULT_PYTHON_TYPE = {
    "numeric": "float",
}


class FieldMergeSpec(BaseModel):
    """Per-field override of the merge policy its type would otherwise get.

    Parameters
    ----------
    dedupe : str | None
        ``fuzzy``, ``exact`` or ``none``. Inherited from the field type when
        omitted.
    threshold : float | None
        Similarity above which two values are one value, for ``fuzzy``
    """

    dedupe: str | None = None
    threshold: float | None = None

    @field_validator("dedupe")
    @classmethod
    def _known_strategy(cls, value: str | None) -> str | None:
        """Reject a strategy name that would silently fall back at merge time.

        Parameters
        ----------
        value : str | None
            Configured strategy

        Returns
        -------
        str | None
            The strategy, unchanged

        Raises
        ------
        ValueError
            If the strategy is not one SERF implements
        """
        if value is not None and value not in DEDUPE_STRATEGIES:
            raise ValueError(f"dedupe must be one of {DEDUPE_STRATEGIES}, got {value!r}")
        return value


class FieldSpec(BaseModel):
    """One field of an entity.

    Parameters
    ----------
    name : str
        Field name, as it appears in the source and in the canonical record
    type : str
        Semantic type, which decides how values of this field are compared and
        combined. One of ``FIELD_TYPES``.
    python_type : str | None
        Scalar type of a single value: ``str``, ``int``, ``float`` or ``bool``.
        Derived from ``type`` when omitted.
    description : str
        What the field means, in the words the LLM should read. This reaches
        the prompt, so it is worth writing properly.
    required : bool
        Whether a source record must carry the field
    merge : FieldMergeSpec | None
        Override of the merge policy this field's type would get
    """

    name: str
    type: str = FIELD_TYPE_TEXT
    python_type: str | None = None
    description: str = ""
    required: bool = False
    merge: FieldMergeSpec | None = None

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        """Reject a semantic type with no merge semantics behind it.

        Parameters
        ----------
        value : str
            Configured type

        Returns
        -------
        str
            The type, unchanged

        Raises
        ------
        ValueError
            If the type is not one SERF knows how to compare
        """
        if value not in FIELD_TYPES:
            raise ValueError(f"type must be one of {FIELD_TYPES}, got {value!r}")
        return value

    @field_validator("python_type")
    @classmethod
    def _known_python_type(cls, value: str | None) -> str | None:
        """Reject a scalar type that cannot be turned into an annotation.

        Parameters
        ----------
        value : str | None
            Configured scalar type

        Returns
        -------
        str | None
            The type, unchanged

        Raises
        ------
        ValueError
            If the type has no Python equivalent here
        """
        if value is not None and value not in PYTHON_TYPES:
            raise ValueError(f"python_type must be one of {sorted(PYTHON_TYPES)}, got {value!r}")
        return value

    def scalar_type(self) -> builtins.type:
        """Return the Python type of one value of this field.

        Returns
        -------
        type
            Declared scalar type, or the one this field's semantic type implies
        """
        declared = self.python_type or _DEFAULT_PYTHON_TYPE.get(self.type, "str")
        return PYTHON_TYPES[declared]

    def merge_policy(self) -> MergePolicy:
        """Return the merge policy for this field, overrides applied.

        Returns
        -------
        MergePolicy
            The field type's configured policy, with any schema override on top
        """
        base = policy_for(self.type)
        if self.merge is None:
            return base
        return MergePolicy(
            dedupe=self.merge.dedupe or base.dedupe,
            threshold=base.threshold if self.merge.threshold is None else self.merge.threshold,
        )


class EntitySchema(BaseModel):
    """The fields of one kind of entity, and how they merge.

    Parameters
    ----------
    entity : str
        Name of the entity type, used as the generated class name
    description : str
        What this kind of entity is, which becomes the class docstring and
        reaches the prompt
    fields : list[FieldSpec]
        The entity's fields, in the order they should be presented
    """

    entity: str
    description: str = ""
    fields: list[FieldSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_field_names(self) -> "EntitySchema":
        """Reject duplicate field names, which would silently shadow each other.

        Returns
        -------
        EntitySchema
            The schema, unchanged

        Raises
        ------
        ValueError
            If two fields share a name
        """
        names = [field.name for field in self.fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate field names in schema {self.entity!r}: {duplicates}")
        return self

    def field(self, name: str) -> FieldSpec | None:
        """Return one field by name.

        Parameters
        ----------
        name : str
            Field name

        Returns
        -------
        FieldSpec | None
            The field, or None when the schema does not declare it
        """
        return next((field for field in self.fields if field.name == name), None)

    def field_types(self) -> dict[str, str]:
        """Return each field's semantic type, for canonicalization.

        Returns
        -------
        dict[str, str]
            Field name to semantic type
        """
        return {field.name: field.type for field in self.fields}

    def merge_policies(self) -> dict[str, MergePolicy]:
        """Return each field's merge policy, overrides applied.

        Returns
        -------
        dict[str, MergePolicy]
            Field name to policy
        """
        return {field.name: field.merge_policy() for field in self.fields}


def parse_schema(data: dict[str, Any]) -> EntitySchema:
    """Build a schema from an already-loaded mapping.

    Parameters
    ----------
    data : dict[str, Any]
        Schema as read from YAML

    Returns
    -------
    EntitySchema
        Validated schema
    """
    return EntitySchema.model_validate(data)


def load_schema(path: str | Path) -> EntitySchema:
    """Load and validate one schema YAML file.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file

    Returns
    -------
    EntitySchema
        Validated schema

    Raises
    ------
    ValueError
        If the file does not hold a mapping
    """
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Schema file {path} must contain a mapping, got {type(data).__name__}")
    schema = parse_schema(data)
    logger.info(f"Loaded schema {schema.entity!r} with {len(schema.fields)} fields from {path}")
    return schema


def load_schemas(directory: str | Path) -> dict[str, EntitySchema]:
    """Load every schema YAML file in a directory.

    Parameters
    ----------
    directory : str | Path
        Directory holding ``.yml`` or ``.yaml`` schema files

    Returns
    -------
    dict[str, EntitySchema]
        Entity name to schema
    """
    root = Path(directory)
    schemas: dict[str, EntitySchema] = {}
    for path in sorted(root.glob("*.y*ml")):
        schema = load_schema(path)
        schemas[schema.entity] = schema
    return schemas
