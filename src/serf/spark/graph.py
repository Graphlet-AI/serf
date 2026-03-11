"""Graph algorithms for SERF entity resolution.

NOTE: Connected components is not needed in SERF's current architecture.
SERF matches entire blocks at once via LLM (not pairwise), so transitive
closure is handled within the block-level matching prompt. This module
is retained for potential future use with pairwise matching strategies.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def connected_components(
    vertices_df: DataFrame, edges_df: DataFrame, spark: SparkSession
) -> DataFrame:
    """Find connected components using iterative min-label propagation.

    vertices_df: must have 'id' column (long)
    edges_df: must have 'src' and 'dst' columns (long)

    Returns DataFrame with columns 'id' and 'component' where component
    is the minimum ID in the connected component.

    Implementation: iterative self-join until convergence.
    Each iteration, propagate the minimum label to all neighbors.

    Parameters
    ----------
    vertices_df : DataFrame
        Vertices with 'id' column
    edges_df : DataFrame
        Edges with 'src' and 'dst' columns
    spark : SparkSession
        SparkSession for execution

    Returns
    -------
    DataFrame
        DataFrame with id, component columns
    """
    comp = vertices_df.select(F.col("id"), F.col("id").alias("component"))

    edges_src = edges_df.select(F.col("src").alias("id"), F.col("dst").alias("neighbor"))
    edges_dst = edges_df.select(F.col("dst").alias("id"), F.col("src").alias("neighbor"))
    edges_both = edges_src.unionByName(edges_dst).distinct()

    max_iter = 1000
    for _ in range(max_iter):
        e = edges_both.alias("e")
        c = comp.alias("c")
        neighbor_comp = e.join(c, F.col("e.neighbor") == F.col("c.id")).select(
            F.col("e.id"), F.col("c.component")
        )
        min_per_vertex = neighbor_comp.groupBy("id").agg(
            F.min("component").alias("min_neighbor_comp")
        )
        comp_next = comp.join(min_per_vertex, on="id", how="left").select(
            F.col("id"),
            F.least(
                F.col("component"),
                F.coalesce(F.col("min_neighbor_comp"), F.col("component")),
            ).alias("component"),
        )
        comp_next.cache()

        comp_old = comp.alias("old")
        comp_new = comp_next.alias("new")
        diff_count = (
            comp_old.join(comp_new, F.col("old.id") == F.col("new.id"))
            .filter(F.col("old.component") != F.col("new.component"))
            .count()
        )
        comp.unpersist()
        comp = comp_next
        if diff_count == 0:
            break

    return comp
