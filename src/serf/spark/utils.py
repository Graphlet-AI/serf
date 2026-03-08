"""Shared Spark utilities for SERF entity resolution."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def split_large_blocks(df: DataFrame, max_block_size: int = 200) -> DataFrame:
    """Split oversized blocks in a DataFrame into sub-blocks.

    The DataFrame should have columns: block_key, block_key_type, block_size,
    entities (array). Uses pyspark.sql.functions to explode, row_number,
    and re-aggregate.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with block_key, block_key_type, block_size, entities
    max_block_size : int
        Maximum entities per block before splitting (default: 200)

    Returns
    -------
    DataFrame
        DataFrame with blocks split, new block_key = original_block_key + _sub_N
    """
    small = df.filter(F.col("block_size") <= max_block_size)
    large = df.filter(F.col("block_size") > max_block_size)

    if large.isEmpty():
        return df

    exploded = large.withColumn("entity", F.explode("entities")).withColumn(
        "idx", F.monotonically_increasing_id()
    )
    w = Window.partitionBy("block_key").orderBy("idx")
    with_sub = exploded.withColumn(
        "sub_block",
        F.floor((F.row_number().over(w) - 1) / max_block_size),
    )
    sub_blocks = (
        with_sub.groupBy("block_key", "block_key_type", "sub_block")
        .agg(
            F.collect_list("entity").alias("entities"),
            F.count("entity").alias("block_size"),
        )
        .withColumn(
            "block_key",
            F.concat(
                F.col("block_key"),
                F.lit("_sub_"),
                F.col("sub_block").cast("string"),
            ),
        )
        .drop("sub_block")
    )
    return small.unionByName(sub_blocks, allowMissingColumns=True)


def select_most_common_property(
    df: DataFrame, group_col: str, value_col: str, result_col: str
) -> DataFrame:
    """Select the most common value per group using window functions.

    Tiebreaker: longest string value.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    group_col : str
        Column to group by
    value_col : str
        Column containing values to pick from
    result_col : str
        Output column name for the selected value

    Returns
    -------
    DataFrame
        DataFrame with one row per group and result_col = most common value
    """
    counted = df.groupBy(group_col, value_col).agg(F.count("*").alias("_cnt"))
    w = Window.partitionBy(group_col).orderBy(
        F.col("_cnt").desc(), F.length(F.col(value_col)).desc()
    )
    ranked = counted.withColumn("_rn", F.row_number().over(w))
    best = ranked.filter(F.col("_rn") == 1).select(
        F.col(group_col),
        F.col(value_col).alias(result_col),
    )
    base = df.drop(result_col) if result_col in df.columns else df
    return base.join(best, on=group_col, how="left")
