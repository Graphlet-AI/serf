"""Shared Spark utilities for SERF entity resolution."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


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
        F.col("_cnt").desc(),
        F.length(F.col(value_col)).desc(),
    )
    ranked = counted.withColumn("_rn", F.row_number().over(w))
    best = ranked.filter(F.col("_rn") == 1).select(
        F.col(group_col),
        F.col(value_col).alias(result_col),
    )
    base = df.drop(result_col) if result_col in df.columns else df
    return base.join(best, on=group_col, how="left")
