"""PySpark-native entity resolution pipeline.

Runs the full blocking → LLM matching → merging pipeline on PySpark
DataFrames. Uses mapInPandas for LLM matching so that Pydantic/DSPy
logic runs inside Spark workers without ever calling toPandas() on
the full dataset.

Architecture:
    PySpark DataFrame (input)
    → Collect name strings for embedding (small — one column only)
    → FAISS blocking in subprocess (returns block assignments)
    → Join block assignments back to DataFrame
    → Salt oversized blocks (simple column add, no UDTF)
    → Sort by (block_key, salt_id) and mapInPandas(match_fn)
    → Resolved PySpark DataFrame
    → Iterate until convergence
"""

import json
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from serf.block.subprocess_embed import cluster_in_subprocess, embed_in_subprocess
from serf.config import config
from serf.logs import get_logger

logger = get_logger(__name__)


def salt_blocks(df: DataFrame, max_block_size: int = 100) -> DataFrame:
    """Add a salt column to split oversized blocks.

    Blocks larger than max_block_size get a salt_id that distributes
    their rows into sub-blocks of at most max_block_size. Small blocks
    get salt_id=0.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with block_key and block_size columns
    max_block_size : int
        Maximum entities per salted block

    Returns
    -------
    DataFrame
        DataFrame with salt_id column added
    """
    w = Window.partitionBy("block_key").orderBy(F.monotonically_increasing_id())
    return df.withColumn(
        "salt_id",
        F.when(
            F.col("block_size") > max_block_size,
            F.floor((F.row_number().over(w) - 1) / max_block_size),
        )
        .otherwise(F.lit(0))
        .cast(T.IntegerType()),
    )


def _make_match_fn(
    name_col: str,
    text_cols: list[str],
    model: str,
    iteration: int = 1,
    output_columns: list[str] | None = None,
) -> Any:
    """Create the mapInPandas matching function.

    The function processes a stream of pandas DataFrames. The input
    is pre-sorted by (block_key, salt_id), so consecutive rows with
    the same (block_key, salt_id) form a block. The function splits
    the stream into blocks, runs DSPy BlockMatch on each, and yields
    resolved DataFrames.

    Parameters
    ----------
    name_col : str
        Column to use as entity name
    text_cols : list[str]
        Columns for entity description
    model : str
        LLM model name
    iteration : int
        Current ER iteration
    output_columns : list[str] | None
        Expected output column names

    Returns
    -------
    Callable
        Function suitable for mapInPandas
    """

    def match_blocks(pdf_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Match entities within blocks via LLM.

        Runs inside Spark workers via mapInPandas. The input iterator
        yields pandas DataFrames (partitions). We accumulate rows by
        (block_key, salt_id) and process each complete block.
        """
        from serf.dspy.types import Entity, EntityBlock
        from serf.match.matcher import EntityMatcher

        matcher = EntityMatcher(model=model)

        for pdf in pdf_iter:
            if pdf.empty:
                yield pdf
                continue

            # Group rows by (block_key, salt_id) within this partition
            grouped = pdf.groupby(["block_key", "salt_id"], sort=False)

            all_output_rows: list[dict[str, Any]] = []

            for group_keys, group_pdf in grouped:
                block_key = str(group_keys[0])  # type: ignore[index]
                salt_id = int(group_keys[1])  # type: ignore[index]
                entities: list[Entity] = []
                for row_idx, row in group_pdf.iterrows():
                    row_dict = {str(k): v for k, v in row.items() if pd.notna(v)}
                    name_val = str(row_dict.get(name_col, f"entity_{row_idx}"))
                    desc_parts = [str(row_dict.get(c, "")) for c in text_cols if row_dict.get(c)]
                    entities.append(
                        Entity(
                            id=int(row_dict.get("er_id", 0)),
                            uuid=str(uuid4()),
                            name=name_val,
                            description=" ".join(desc_parts),
                            attributes={
                                k: v
                                for k, v in row_dict.items()
                                if k
                                not in (
                                    "block_key",
                                    "block_size",
                                    "salt_id",
                                    "er_id",
                                )
                            },
                        )
                    )

                block = EntityBlock(
                    block_key=f"{block_key}_s{salt_id}",
                    block_key_type="semantic",
                    block_size=len(entities),
                    entities=entities,
                )

                resolution = matcher.resolve_block(block, iteration=iteration)

                for e in resolution.resolved_entities:
                    row_out: dict[str, Any] = {}
                    for k, v in e.attributes.items():
                        row_out[k] = v
                    row_out[name_col] = e.name
                    row_out["block_key"] = str(block_key)
                    row_out["block_size"] = len(resolution.resolved_entities)
                    row_out["salt_id"] = int(salt_id)
                    row_out["er_id"] = e.id
                    row_out["er_uuid"] = e.uuid
                    row_out["er_source_ids"] = json.dumps(e.source_ids) if e.source_ids else None
                    row_out["er_source_uuids"] = (
                        json.dumps(e.source_uuids) if e.source_uuids else None
                    )
                    row_out["er_match_skip"] = e.match_skip
                    row_out["er_match_skip_reason"] = e.match_skip_reason
                    all_output_rows.append(row_out)

            if all_output_rows:
                yield pd.DataFrame(all_output_rows)
            else:
                yield pd.DataFrame()

    return match_blocks


def run_spark_pipeline(
    df: DataFrame,
    spark: SparkSession,
    name_col: str | None = None,
    text_cols: list[str] | None = None,
    model: str | None = None,
    target_block_size: int = 30,
    max_block_size: int = 100,
    max_iterations: int = 5,
    convergence_threshold: float = 0.01,
    limit: int | None = None,
) -> DataFrame:
    """Run entity resolution on a PySpark DataFrame.

    Uses embeddings (subprocess) for blocking and mapInPandas for
    LLM matching. Never calls toPandas() on the full dataset.

    Parameters
    ----------
    df : DataFrame
        Input PySpark DataFrame with entity records
    spark : SparkSession
        Active Spark session
    name_col : str | None
        Column for entity name. Auto-detected if None.
    text_cols : list[str] | None
        Columns for description. Auto-detected if None.
    model : str | None
        LLM model. From config if None.
    target_block_size : int
        Target entities per FAISS block
    max_block_size : int
        Hard cap — oversized blocks get salted
    max_iterations : int
        Max ER iterations (0 for auto)
    convergence_threshold : float
        Stop when per-round reduction < this fraction
    limit : int | None
        Max blocks to process (for testing)

    Returns
    -------
    DataFrame
        Resolved PySpark DataFrame with ER metadata columns
    """
    model = model or config.get("models.llm")
    embedding_model = config.get("models.embedding")

    # Auto-detect name column
    if name_col is None:
        for candidate in ["title", "name", "product_name", "company_name"]:
            if candidate in df.columns:
                name_col = candidate
                break
        if name_col is None:
            name_col = [c for c in df.columns if c.lower() != "id"][0]

    # Auto-detect text columns
    if text_cols is None:
        text_cols = [
            f.name
            for f in df.schema.fields
            if isinstance(f.dataType, T.StringType)
            and f.name != name_col
            and f.name.lower() != "id"
        ]

    logger.info(f"Spark pipeline: name_col={name_col}, text_cols={text_cols}")

    # Add sequential er_id
    w = Window.orderBy(F.monotonically_increasing_id())
    working_df = df.withColumn("er_id", F.row_number().over(w))
    original_count = working_df.count()
    logger.info(f"Input: {original_count} entities")

    max_iters = max_iterations if max_iterations > 0 else 20

    for iteration in range(1, max_iters + 1):
        iter_count = working_df.count()
        logger.info(f"\n=== Iteration {iteration}: {iter_count} entities ===")

        # Phase 1: Collect name strings for embedding (small — one column)
        name_rows = working_df.select("er_id", name_col).collect()
        ids = [str(row["er_id"]) for row in name_rows]
        texts = [str(row[name_col] or "") for row in name_rows]

        # Phase 2: Embed in subprocess
        logger.info("  Embedding in subprocess...")
        embeddings = embed_in_subprocess(texts, model_name=embedding_model)

        # Phase 3: Cluster in subprocess
        effective_target = max(10, target_block_size // iteration)
        logger.info(f"  Clustering (target={effective_target})...")
        block_assignments = cluster_in_subprocess(
            embeddings, ids, target_block_size=effective_target
        )

        # Phase 4: Build block assignment DataFrame and join
        assignment_rows = []
        for block_key, entity_ids in block_assignments.items():
            block_size = len(entity_ids)
            for eid in entity_ids:
                assignment_rows.append((int(eid), block_key, block_size))

        block_schema = T.StructType(
            [
                T.StructField("er_id", T.IntegerType()),
                T.StructField("block_key", T.StringType()),
                T.StructField("block_size", T.IntegerType()),
            ]
        )
        block_df = spark.createDataFrame(assignment_rows, schema=block_schema)
        blocked_df = working_df.join(block_df, on="er_id", how="inner")

        block_count = blocked_df.select("block_key").distinct().count()
        logger.info(f"  {block_count} blocks created")

        # Phase 5: Salt oversized blocks
        salted_df = salt_blocks(blocked_df, max_block_size=max_block_size)

        # Optional: limit blocks for testing
        if limit:
            block_keys = salted_df.select("block_key").distinct().limit(limit).collect()
            block_key_list = [row["block_key"] for row in block_keys]
            salted_df = salted_df.filter(F.col("block_key").isin(block_key_list))
            logger.info(f"  Limited to {limit} blocks")

        # Phase 6: Sort by (block_key, salt_id) then mapInPandas
        sorted_df = salted_df.repartition("block_key", "salt_id").sortWithinPartitions(
            "block_key", "salt_id"
        )

        match_fn = _make_match_fn(name_col, text_cols, model, iteration)
        output_schema = sorted_df.schema

        # Add ER metadata fields to schema
        er_fields = [
            T.StructField("er_uuid", T.StringType(), True),
            T.StructField("er_source_ids", T.StringType(), True),
            T.StructField("er_source_uuids", T.StringType(), True),
            T.StructField("er_match_skip", T.BooleanType(), True),
            T.StructField("er_match_skip_reason", T.StringType(), True),
        ]
        existing_names = {f.name for f in output_schema.fields}
        for f in er_fields:
            if f.name not in existing_names:
                output_schema = output_schema.add(f)

        logger.info(f"  Matching with LLM via mapInPandas (model={model})...")
        resolved_df = sorted_df.mapInPandas(match_fn, schema=output_schema)

        resolved_count = resolved_df.count()

        # Compute reduction
        reduction = iter_count - resolved_count
        reduction_pct = (reduction / iter_count * 100) if iter_count > 0 else 0.0
        overall_pct = (
            (original_count - resolved_count) / original_count * 100 if original_count > 0 else 0.0
        )
        logger.info(
            f"  Iteration {iteration}: {iter_count} → {resolved_count} "
            f"({reduction_pct:.1f}% reduction, {overall_pct:.1f}% overall)"
        )

        # Check convergence
        if reduction_pct < convergence_threshold * 100:
            logger.info(f"  Converged: {reduction_pct:.2f}% < threshold")
            working_df = resolved_df
            break

        # Re-assign sequential IDs for next iteration
        working_df = resolved_df.drop("block_key", "block_size", "salt_id")
        w2 = Window.orderBy(F.monotonically_increasing_id())
        working_df = working_df.withColumn("er_id", F.row_number().over(w2))

    logger.info(f"\nSpark pipeline complete: {original_count} → {working_df.count()} entities")
    return working_df
