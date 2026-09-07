"""Iceberg integration for SERF entity resolution.

Best-effort, functional stubs that work if Iceberg JARs are available.
"""

from pyspark.sql import DataFrame, SparkSession


def create_iceberg_session(warehouse_path: str = "data/iceberg") -> SparkSession:
    """Create a SparkSession configured for local Iceberg catalog.

    Parameters
    ----------
    warehouse_path : str
        Path for Iceberg warehouse (default: data/iceberg)

    Returns
    -------
    SparkSession
        SparkSession with Iceberg support
    """
    return (
        SparkSession.builder.master("local[*]")
        .appName("serf-iceberg")
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
        .config("spark.sql.catalog.spark_catalog.type", "hadoop")
        .config("spark.sql.catalog.spark_catalog.warehouse", warehouse_path)
        .getOrCreate()
    )


def write_to_iceberg(df: DataFrame, table_name: str, spark: SparkSession) -> None:
    """Write DataFrame to an Iceberg table. Creates table if not exists.

    Parameters
    ----------
    df : DataFrame
        DataFrame to write
    table_name : str
        Fully qualified table name (e.g. spark_catalog.db.table)
    spark : SparkSession
        SparkSession with Iceberg catalog
    """
    df.writeTo(table_name).using("iceberg").createOrReplace()


def read_from_iceberg(
    table_name: str,
    spark: SparkSession,
    snapshot_id: int | None = None,
) -> DataFrame:
    """Read from Iceberg table, optionally at a specific snapshot for time travel.

    Parameters
    ----------
    table_name : str
        Fully qualified table name
    spark : SparkSession
        SparkSession with Iceberg catalog
    snapshot_id : int | None
        Snapshot ID for time travel (None = latest)

    Returns
    -------
    DataFrame
        DataFrame with table contents
    """
    if snapshot_id is not None:
        return spark.read.option("snapshot-id", str(snapshot_id)).table(table_name)
    return spark.table(table_name)
