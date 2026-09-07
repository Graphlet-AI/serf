"""Tests for auto-generating Pydantic Entity subclasses from Spark schemas."""

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

from serf.dspy.type_generator import entity_type_from_spark_schema, spark_type_to_python
from serf.dspy.types import DatasetProfile, Entity, FieldProfile


def test_spark_type_to_python_string() -> None:
    """Test StringType maps to str."""
    assert spark_type_to_python(StringType()) is str


def test_spark_type_to_python_long() -> None:
    """Test LongType maps to int."""
    assert spark_type_to_python(LongType()) is int


def test_spark_type_to_python_integer() -> None:
    """Test IntegerType maps to int."""
    assert spark_type_to_python(IntegerType()) is int


def test_spark_type_to_python_double() -> None:
    """Test DoubleType maps to float."""
    assert spark_type_to_python(DoubleType()) is float


def test_spark_type_to_python_boolean() -> None:
    """Test BooleanType maps to bool."""
    assert spark_type_to_python(BooleanType()) is bool


def test_spark_type_to_python_array() -> None:
    """Test ArrayType(StringType) maps to list[str]."""
    result = spark_type_to_python(ArrayType(StringType()))
    assert result == list[str]


def test_entity_type_from_simple_schema() -> None:
    """Test generating Entity subclass from a simple schema."""
    schema = StructType(
        [
            StructField("title", StringType(), nullable=True),
            StructField("price", DoubleType(), nullable=True),
            StructField("category", StringType(), nullable=True),
        ]
    )

    EntityClass = entity_type_from_spark_schema(schema, entity_type_name="ProductEntity")
    assert issubclass(EntityClass, Entity)
    assert EntityClass.__name__ == "ProductEntity"

    # Should be able to create an instance with the new fields
    instance = EntityClass(
        id=1,
        name="Test Product",
        title="Test Product",  # type: ignore[call-arg]
        price=29.99,  # type: ignore[call-arg]
        category="Electronics",  # type: ignore[call-arg]
    )
    assert instance.title == "Test Product"  # type: ignore[attr-defined]
    assert instance.price == 29.99  # type: ignore[attr-defined]
    assert instance.category == "Electronics"  # type: ignore[attr-defined]
    # Base Entity fields still work
    assert instance.id == 1
    assert instance.name == "Test Product"


def test_entity_type_skips_er_metadata() -> None:
    """Test that ER metadata fields from the schema are skipped."""
    schema = StructType(
        [
            StructField("id", IntegerType(), nullable=False),
            StructField("uuid", StringType(), nullable=True),
            StructField("name", StringType(), nullable=False),
            StructField("title", StringType(), nullable=True),
            StructField("source_ids", ArrayType(IntegerType()), nullable=True),
        ]
    )

    EntityClass = entity_type_from_spark_schema(schema)
    # title should be present as a new field, but id/uuid/name/source_ids
    # should come from the base Entity class
    instance = EntityClass(id=1, name="Test", title="My Title")  # type: ignore[call-arg]
    assert instance.title == "My Title"  # type: ignore[attr-defined]


def test_entity_type_with_profile() -> None:
    """Test generating Entity subclass with profile descriptions."""
    schema = StructType(
        [
            StructField("product_name", StringType(), nullable=True),
            StructField("price_usd", DoubleType(), nullable=True),
        ]
    )
    profile = DatasetProfile(
        record_count=100,
        field_profiles=[
            FieldProfile(name="product_name", inferred_type="name"),
            FieldProfile(name="price_usd", inferred_type="numeric"),
        ],
    )

    EntityClass = entity_type_from_spark_schema(
        schema, profile=profile, entity_type_name="EnrichedEntity"
    )
    assert issubclass(EntityClass, Entity)

    instance = EntityClass(
        id=1,
        name="Widget",
        product_name="Widget Pro",  # type: ignore[call-arg]
        price_usd=9.99,  # type: ignore[call-arg]
    )
    assert instance.product_name == "Widget Pro"  # type: ignore[attr-defined]


def test_entity_type_non_nullable_field() -> None:
    """Test that non-nullable fields are required."""
    schema = StructType(
        [
            StructField("required_field", StringType(), nullable=False),
        ]
    )

    EntityClass = entity_type_from_spark_schema(schema)
    instance = EntityClass(id=1, name="Test", required_field="value")  # type: ignore[call-arg]
    assert instance.required_field == "value"  # type: ignore[attr-defined]


def test_entity_type_nullable_field_defaults_none() -> None:
    """Test that nullable fields default to None."""
    schema = StructType(
        [
            StructField("optional_field", StringType(), nullable=True),
        ]
    )

    EntityClass = entity_type_from_spark_schema(schema)
    instance = EntityClass(id=1, name="Test")
    assert instance.optional_field is None  # type: ignore[attr-defined]
