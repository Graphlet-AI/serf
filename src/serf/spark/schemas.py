"""Pydantic to Spark schema bridge for SERF entity resolution."""

from __future__ import annotations

from types import UnionType
from typing import TYPE_CHECKING, Any, get_args, get_origin

from pydantic import BaseModel
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from serf.dspy.types import Entity
from serf.logs import get_logger

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

logger = get_logger(__name__)

BLOCK_FIELDS = ["block_key", "block_key_type", "entities", "block_size"]


def convert_ints_to_longs(schema: StructType) -> StructType:
    """Recursively convert IntegerType to LongType in a Spark schema.

    Parameters
    ----------
    schema : StructType
        The Spark schema to convert

    Returns
    -------
    StructType
        New schema with IntegerType replaced by LongType
    """
    new_fields = []
    for field in schema.fields:
        if isinstance(field.dataType, IntegerType):
            new_fields.append(StructField(field.name, LongType(), field.nullable))
        elif isinstance(field.dataType, StructType):
            new_fields.append(
                StructField(
                    field.name,
                    convert_ints_to_longs(field.dataType),
                    field.nullable,
                )
            )
        elif isinstance(field.dataType, ArrayType):
            elem = field.dataType.elementType
            if isinstance(elem, IntegerType):
                new_fields.append(StructField(field.name, ArrayType(LongType()), field.nullable))
            elif isinstance(elem, StructType):
                new_fields.append(
                    StructField(
                        field.name,
                        ArrayType(convert_ints_to_longs(elem)),
                        field.nullable,
                    )
                )
            else:
                new_fields.append(field)
        else:
            new_fields.append(field)
    return StructType(new_fields)


def _pydantic_field_to_spark(name: str, annotation: Any, required: bool) -> StructField:
    """Convert a single Pydantic field annotation to a Spark StructField.

    Map: str->StringType, int->LongType, float->DoubleType, bool->BooleanType,
    Optional[X]->nullable X, list[X]->ArrayType(X), dict->StringType (JSON).
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is not None:
        is_union = origin is UnionType or (hasattr(origin, "__name__") and "Union" in str(origin))
        if is_union and type(None) in args:
            inner = next(a for a in args if a is not type(None))
            field = _pydantic_field_to_spark(name, inner, False)
            return StructField(name, field.dataType, nullable=True)
        if origin is list:
            elem_type = args[0] if args else str
            if elem_type is int:
                return StructField(name, ArrayType(LongType()), nullable=not required)
            if elem_type is str:
                return StructField(name, ArrayType(StringType()), nullable=not required)
            return StructField(name, ArrayType(StringType()), nullable=not required)
        if origin is dict:
            return StructField(name, StringType(), nullable=not required)

    if annotation is int:
        return StructField(name, LongType(), nullable=not required)
    if annotation is float:
        return StructField(name, DoubleType(), nullable=not required)
    if annotation is bool:
        return StructField(name, BooleanType(), nullable=not required)
    return StructField(name, StringType(), nullable=not required)


def get_entity_spark_schema(entity_class: type[BaseModel] = Entity) -> StructType:
    """Generate a Spark StructType from a Pydantic model class.

    Map: str->StringType, int->LongType, float->DoubleType, bool->BooleanType,
    Optional[X]->nullable X, list[X]->ArrayType(X), dict->StringType (JSON).

    Parameters
    ----------
    entity_class : type[BaseModel]
        Pydantic model class (default: Entity)

    Returns
    -------
    StructType
        Spark schema for the entity
    """
    fields = []
    for name, field_info in entity_class.model_fields.items():
        annotation = field_info.annotation
        required = field_info.is_required()
        spark_field = _pydantic_field_to_spark(name, annotation, required)
        fields.append(spark_field)
    return StructType(fields)


def normalize_entity_dataframe(df: DataFrame, entity_class: type[BaseModel] = Entity) -> DataFrame:
    """Ensure consistent field order/types. Add missing fields as null with correct type.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with entity data
    entity_class : type[BaseModel]
        Pydantic model class (default: Entity)

    Returns
    -------
    DataFrame
        DataFrame with normalized schema
    """
    from pyspark.sql import functions as F

    target_schema = get_entity_spark_schema(entity_class)
    existing = set(df.columns)

    exprs = []
    for field in target_schema.fields:
        if field.name in existing:
            exprs.append(F.col(field.name).cast(field.dataType).alias(field.name))
        else:
            exprs.append(F.lit(None).cast(field.dataType).alias(field.name))

    return df.select(exprs)


def validate_block_schema(df: DataFrame) -> None:
    """Validate that a DataFrame has required block fields. Raises ValueError if not.

    Parameters
    ----------
    df : DataFrame
        DataFrame to validate

    Raises
    ------
    ValueError
        If any required block field is missing
    """
    missing = [f for f in BLOCK_FIELDS if f not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required block fields: {missing}")


def get_matches_schema() -> StructType:
    """Return schema for reading matches JSONL files.

    Returns
    -------
    StructType
        Schema for MatchDecision records: entity_a_id, entity_b_id, is_match,
        confidence, reasoning
    """
    return StructType(
        [
            StructField("entity_a_id", LongType(), False),
            StructField("entity_b_id", LongType(), False),
            StructField("is_match", BooleanType(), False),
            StructField("confidence", DoubleType(), False),
            StructField("reasoning", StringType(), False),
        ]
    )
