"""Tests for PySpark-native entity resolution pipeline."""

import pytest
from pyspark.sql import SparkSession

from serf.spark.pipeline import salt_blocks


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Create a local SparkSession for testing."""
    return SparkSession.builder.master("local[*]").appName("serf-test-spark-pipeline").getOrCreate()


def test_salt_blocks_small_blocks_get_zero(spark: SparkSession) -> None:
    """Test that blocks smaller than max_block_size get salt_id=0."""
    data = [
        (1, "block_a", 3),
        (2, "block_a", 3),
        (3, "block_a", 3),
    ]
    df = spark.createDataFrame(data, ["er_id", "block_key", "block_size"])
    salted = salt_blocks(df, max_block_size=10)
    salt_values = [row["salt_id"] for row in salted.collect()]
    assert all(s == 0 for s in salt_values)


def test_salt_blocks_large_blocks_get_salted(spark: SparkSession) -> None:
    """Test that oversized blocks get distributed salt_ids."""
    data = [(i, "block_big", 20) for i in range(20)]
    df = spark.createDataFrame(data, ["er_id", "block_key", "block_size"])
    salted = salt_blocks(df, max_block_size=5)
    salt_values = sorted(set(row["salt_id"] for row in salted.collect()))
    # 20 items / 5 per salt = 4 salt groups (0, 1, 2, 3)
    assert len(salt_values) == 4
    assert salt_values == [0, 1, 2, 3]


def test_salt_blocks_mixed(spark: SparkSession) -> None:
    """Test salting with both small and large blocks."""
    data = [
        (1, "small", 2),
        (2, "small", 2),
        (3, "big", 10),
        (4, "big", 10),
        (5, "big", 10),
        (6, "big", 10),
        (7, "big", 10),
        (8, "big", 10),
        (9, "big", 10),
        (10, "big", 10),
        (11, "big", 10),
        (12, "big", 10),
    ]
    df = spark.createDataFrame(data, ["er_id", "block_key", "block_size"])
    salted = salt_blocks(df, max_block_size=5)

    small_salts = [row["salt_id"] for row in salted.filter("block_key = 'small'").collect()]
    big_salts = sorted(set(row["salt_id"] for row in salted.filter("block_key = 'big'").collect()))

    assert all(s == 0 for s in small_salts)
    assert len(big_salts) == 2  # 10 items / 5 = 2 salt groups
