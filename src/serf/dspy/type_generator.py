"""Auto-generate Pydantic Entity subclasses from PySpark DataFrame schemas.

When a user provides a DataFrame without a custom entity type, SERF
auto-generates a Pydantic class from the DataFrame schema.
"""

from typing import Any

from pydantic import Field, create_model
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from serf.dspy.types import DatasetProfile, Entity

# Spark type to Python type mapping
SPARK_TYPE_MAP: dict[type, type] = {
    StringType: str,
    LongType: int,
    IntegerType: int,
    DoubleType: float,
    FloatType: float,
    BooleanType: bool,
}

# ER metadata fields that are added automatically
ER_METADATA_FIELDS = {
    "id",
    "uuid",
    "name",
    "description",
    "entity_type",
    "attributes",
    "source_ids",
    "source_uuids",
    "match_skip",
    "match_skip_reason",
    "match_skip_history",
}


def spark_type_to_python(spark_type: Any) -> type:
    """Convert a Spark DataType to a Python type.

    Parameters
    ----------
    spark_type : Any
        A PySpark DataType instance

    Returns
    -------
    type
        The corresponding Python type
    """
    for spark_cls, py_type in SPARK_TYPE_MAP.items():
        if isinstance(spark_type, spark_cls):
            return py_type

    if isinstance(spark_type, ArrayType):
        element_type = spark_type_to_python(spark_type.elementType)
        return list[element_type]  # type: ignore[valid-type]

    # Default to str for unknown types
    return str


def entity_type_from_spark_schema(
    schema: StructType,
    profile: DatasetProfile | None = None,
    entity_type_name: str = "AutoEntity",
) -> type[Entity]:
    """Generate a Pydantic Entity subclass from a Spark StructType schema.

    Uses the DatasetProfile to enrich fields with descriptions
    (e.g., marking a field as "name", "identifier", "date").

    Parameters
    ----------
    schema : StructType
        The Spark schema to convert
    profile : Optional[DatasetProfile]
        Profiling results identifying field types and roles
    entity_type_name : str
        Name for the generated class

    Returns
    -------
    type[Entity]
        A dynamically created Pydantic subclass of Entity
    """
    field_definitions: dict[str, Any] = {}
    profile_map: dict[str, str] = {}

    if profile:
        for fp in profile.field_profiles:
            profile_map[fp.name] = fp.inferred_type

    for field in schema.fields:
        field: StructField
        # Skip ER metadata fields — they come from Entity base class
        if field.name in ER_METADATA_FIELDS:
            continue

        py_type = spark_type_to_python(field.dataType)
        description = profile_map.get(field.name, "")

        if field.nullable:
            optional_type = py_type | None  # type: ignore[valid-type]
            field_definitions[field.name] = (
                optional_type,
                Field(default=None, description=description),
            )
        else:
            field_definitions[field.name] = (
                py_type,
                Field(description=description),
            )

    model = create_model(
        entity_type_name,
        __base__=Entity,
        **field_definitions,
    )
    return model
