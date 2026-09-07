"""Tests for Spark graph utilities."""

import pytest
from pyspark.sql import SparkSession

from serf.spark.graph import connected_components


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Create a local SparkSession for tests."""
    return SparkSession.builder.master("local[*]").appName("test_graph").getOrCreate()


def test_connected_components_simple_graph(spark: SparkSession) -> None:
    """Test connected_components: vertices {1,2,3,4,5}, edges {(1,2),(2,3),(4,5)}.

    Expected: {1,2,3} in one component (min=1), {4,5} in another (min=4).
    """
    vertices = spark.createDataFrame(
        [(1,), (2,), (3,), (4,), (5,)],
        schema="id long",
    )
    edges = spark.createDataFrame(
        [(1, 2), (2, 3), (4, 5)],
        schema="src long, dst long",
    )
    result = connected_components(vertices, edges, spark)
    rows = {row.id: row.component for row in result.collect()}
    assert rows[1] == 1
    assert rows[2] == 1
    assert rows[3] == 1
    assert rows[4] == 4
    assert rows[5] == 4


def test_connected_components_disconnected_vertices(spark: SparkSession) -> None:
    """Test with disconnected vertices (no edges)."""
    vertices = spark.createDataFrame(
        [(1,), (2,), (3,)],
        schema="id long",
    )
    edges = spark.createDataFrame([], schema="src long, dst long")
    result = connected_components(vertices, edges, spark)
    rows = {row.id: row.component for row in result.collect()}
    assert rows[1] == 1
    assert rows[2] == 2
    assert rows[3] == 3


def test_connected_components_single_vertex(spark: SparkSession) -> None:
    """Test with single vertex."""
    vertices = spark.createDataFrame([(1,)], schema="id long")
    edges = spark.createDataFrame([], schema="src long, dst long")
    result = connected_components(vertices, edges, spark)
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0].id == 1
    assert rows[0].component == 1
