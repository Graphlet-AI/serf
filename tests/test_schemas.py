"""Tests for Spark schema utilities."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from serf.spark.schemas import (
    convert_ints_to_longs,
    get_entity_spark_schema,
    get_matches_schema,
    validate_block_schema,
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Create a local SparkSession for tests."""
    return SparkSession.builder.master("local[*]").appName("test_schemas").getOrCreate()


def test_get_entity_spark_schema_returns_valid_struct_type() -> None:
    """Test get_entity_spark_schema returns valid StructType."""
    schema = get_entity_spark_schema()
    assert isinstance(schema, StructType)
    assert len(schema.fields) > 0
    field_names = {f.name for f in schema.fields}
    assert "id" in field_names
    assert "name" in field_names
    assert "entity_type" in field_names
    assert "attributes" in field_names
    id_field = next(f for f in schema.fields if f.name == "id")
    assert isinstance(id_field.dataType, LongType)
    name_field = next(f for f in schema.fields if f.name == "name")
    assert isinstance(name_field.dataType, StringType)


def test_convert_ints_to_longs() -> None:
    """Test convert_ints_to_longs replaces IntegerType with LongType."""
    schema = StructType(
        [
            StructField("a", IntegerType(), False),
            StructField("b", StringType(), False),
            StructField("c", IntegerType(), True),
        ]
    )
    result = convert_ints_to_longs(schema)
    assert isinstance(result.fields[0].dataType, LongType)
    assert isinstance(result.fields[1].dataType, StringType)
    assert isinstance(result.fields[2].dataType, LongType)


def test_validate_block_schema_valid(spark: SparkSession) -> None:
    """Test validate_block_schema passes with valid DataFrame."""
    from pyspark.sql import Row

    df = spark.createDataFrame(
        [Row(block_key="k1", block_key_type="semantic", entities=[], block_size=0)],
        schema=StructType(
            [
                StructField("block_key", StringType(), False),
                StructField("block_key_type", StringType(), False),
                StructField("entities", ArrayType(StringType()), False),
                StructField("block_size", LongType(), False),
            ]
        ),
    )
    validate_block_schema(df)


def test_validate_block_schema_invalid(spark: SparkSession) -> None:
    """Test validate_block_schema raises ValueError when fields missing."""
    from pyspark.sql import Row

    df = spark.createDataFrame([Row(a=1, b=2)], schema="a int, b int")
    with pytest.raises(ValueError, match="missing required block fields"):
        validate_block_schema(df)


def test_get_matches_schema() -> None:
    """Test get_matches_schema returns expected schema."""
    schema = get_matches_schema()
    assert isinstance(schema, StructType)
    names = [f.name for f in schema.fields]
    assert "entity_a_id" in names
    assert "entity_b_id" in names
    assert "is_match" in names
    assert "confidence" in names
    assert "reasoning" in names
