"""Declarative entity schemas and the Pydantic classes generated from them."""

from serf.schema.generate import (
    CanonicalEntity,
    canonical_model,
    clear_cache,
    match_and_merge_signature,
    merge_field_annotation,
    merge_signature,
    source_model,
)
from serf.schema.spec import (
    CARDINALITIES,
    CARDINALITY_LIST,
    CARDINALITY_SINGLE,
    FIELD_TYPES,
    PYTHON_TYPES,
    EntitySchema,
    FieldMergeSpec,
    FieldSpec,
    load_schema,
    load_schemas,
    parse_schema,
)

__all__ = [
    "CARDINALITIES",
    "CARDINALITY_LIST",
    "CARDINALITY_SINGLE",
    "FIELD_TYPES",
    "PYTHON_TYPES",
    "CanonicalEntity",
    "EntitySchema",
    "FieldMergeSpec",
    "FieldSpec",
    "canonical_model",
    "clear_cache",
    "load_schema",
    "load_schemas",
    "match_and_merge_signature",
    "merge_field_annotation",
    "merge_signature",
    "parse_schema",
    "source_model",
]
