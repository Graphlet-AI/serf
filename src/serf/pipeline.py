"""End-to-end entity resolution pipeline.

Takes any tabular data (CSV, Parquet, or Iceberg) and runs the full
blocking → LLM matching → merging pipeline with iterative convergence.

Embeddings are used ONLY for blocking (FAISS clustering).
All matching is done by LLM via DSPy BlockMatch signatures.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from serf.block.embeddings import EntityEmbedder
from serf.block.faiss_blocker import FAISSBlocker
from serf.dspy.types import Entity, EntityBlock, IterationMetrics
from serf.logs import get_logger

logger = get_logger(__name__)

# Candidate columns for auto-detecting the entity name
NAME_CANDIDATES = ["title", "name", "product_name", "company_name", "entity_name"]


class ERConfig:
    """Configuration for an entity resolution run.

    Parameters
    ----------
    name_field : str | None
        Column to use as entity name. Auto-detected if None.
    text_fields : list[str] | None
        Columns to use for embedding text. Auto-detected if None.
    entity_type : str
        Label for the entity type.
    blocking_method : str
        Blocking method: "semantic" or "name".
    target_block_size : int
        Target entities per FAISS block.
    max_block_size : int
        Max entities per block before splitting.
    model : str
        LLM model name for matching.
    max_iterations : int
        Maximum ER iterations.
    convergence_threshold : float
        Stop when per-round reduction falls below this.
    """

    def __init__(
        self,
        name_field: str | None = None,
        text_fields: list[str] | None = None,
        blocking_fields: list[str] | None = None,
        entity_type: str = "entity",
        blocking_method: str = "semantic",
        target_block_size: int = 30,
        max_block_size: int = 100,
        model: str = "gemini/gemini-2.0-flash",
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
    ) -> None:
        self.name_field = name_field
        self.text_fields = text_fields
        self.blocking_fields = blocking_fields
        self.entity_type = entity_type
        self.blocking_method = blocking_method
        self.target_block_size = target_block_size
        self.max_block_size = max_block_size
        self.model = model
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    @classmethod
    def from_yaml(cls, path: str) -> "ERConfig":
        """Load ER config from a YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML config file

        Returns
        -------
        ERConfig
            Loaded configuration
        """
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        blocking = data.get("blocking", {})
        matching = data.get("matching", {})

        return cls(
            name_field=data.get("name_field"),
            text_fields=data.get("text_fields"),
            blocking_fields=data.get("blocking_fields"),
            entity_type=data.get("entity_type", "entity"),
            blocking_method=blocking.get("method", "semantic"),
            target_block_size=blocking.get("target_block_size", 30),
            max_block_size=blocking.get("max_block_size", 100),
            model=matching.get("model", "gemini/gemini-2.0-flash"),
            max_iterations=data.get("max_iterations", 5),
            convergence_threshold=data.get("convergence_threshold", 0.01),
        )


def load_data(input_path: str) -> pd.DataFrame:
    """Load tabular data from CSV, Parquet, or Iceberg.

    Parameters
    ----------
    input_path : str
        File path (.csv, .parquet) or Iceberg URI (iceberg://...)

    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    if input_path.startswith("iceberg://"):
        return _load_iceberg(input_path)
    path = Path(input_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        for encoding in ("utf-8", "latin-1"):
            try:
                df: pd.DataFrame = pd.read_csv(path, sep=sep, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, sep=sep, encoding="latin-1")
    raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv, .parquet, or iceberg://")


def _load_iceberg(uri: str) -> pd.DataFrame:
    """Load data from an Iceberg table.

    Parameters
    ----------
    uri : str
        Iceberg URI like iceberg://catalog.database.table

    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    from pyspark.sql import SparkSession

    table_name = uri.replace("iceberg://", "")
    spark = SparkSession.builder.appName("serf").getOrCreate()
    df = spark.read.format("iceberg").load(table_name).toPandas()
    return df


def _detect_name_field(df: pd.DataFrame) -> str:
    """Auto-detect the best name column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns
    -------
    str
        Name of the detected column
    """
    for candidate in NAME_CANDIDATES:
        for col in df.columns:
            if str(col).lower() == candidate.lower():
                return str(col)
    # Fall back to first non-id string column
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if str(col).lower() != "id" and (
            dtype_str in ("object", "str", "string") or dtype_str.startswith("str")
        ):
            return str(col)
    return str(df.columns[0])


def _detect_text_fields(df: pd.DataFrame, name_field: str) -> list[str]:
    """Auto-detect text columns for embedding.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    name_field : str
        Already-detected name field (excluded from result)

    Returns
    -------
    list[str]
        Text column names
    """
    text_cols = []
    for col in df.columns:
        col_str = str(col)
        if col_str == name_field or col_str.lower() == "id":
            continue
        dtype = str(df[col].dtype)
        if dtype in ("object", "str", "string") or dtype.startswith("str"):
            text_cols.append(col_str)
    return text_cols


def dataframe_to_entities(
    df: pd.DataFrame,
    name_field: str,
    text_fields: list[str],
    entity_type: str = "entity",
) -> list[Entity]:
    """Convert a DataFrame to a list of Entity objects.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    name_field : str
        Column to use as entity name
    text_fields : list[str]
        Columns to concatenate for description
    entity_type : str
        Entity type label

    Returns
    -------
    list[Entity]
        Converted entities
    """
    entities: list[Entity] = []
    for i, (_idx, row) in enumerate(df.iterrows()):
        row_dict = {str(k): v for k, v in row.items() if pd.notna(v)}

        name = str(row_dict.get(name_field, f"entity_{i}"))
        desc_parts = [str(row_dict[col]) for col in text_fields if col in row_dict]
        description = " ".join(desc_parts)

        attrs: dict[str, Any] = {}
        for k, v in row_dict.items():
            attrs[k] = str(v) if not isinstance(v, int | float | bool) else v

        entities.append(
            Entity(
                id=i,
                name=name,
                description=description,
                entity_type=entity_type,
                attributes=attrs,
            )
        )
    return entities


def run_pipeline(
    input_path: str,
    output_path: str,
    er_config: ERConfig | None = None,
) -> dict[str, Any]:
    """Run the full entity resolution pipeline.

    Uses embeddings for blocking (FAISS clustering) and LLM for matching
    (DSPy BlockMatch). Runs multiple iterations until convergence.

    Parameters
    ----------
    input_path : str
        Path to input data (CSV, Parquet, or Iceberg URI)
    output_path : str
        Directory for output files
    er_config : ERConfig | None
        Pipeline configuration. Uses defaults if None.

    Returns
    -------
    dict[str, Any]
        Summary with metrics per iteration and final counts
    """
    cfg = er_config or ERConfig()
    start = time.time()

    logger.info(f"Loading data from {input_path}")
    df = load_data(input_path)
    logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")

    # Detect fields
    name_field = cfg.name_field or _detect_name_field(df)
    text_fields = cfg.text_fields or _detect_text_fields(df, name_field)
    logger.info(f"Name field: {name_field}")
    logger.info(f"Text fields: {text_fields}")

    # Convert to entities
    entities = dataframe_to_entities(df, name_field, text_fields, cfg.entity_type)
    original_count = len(entities)
    logger.info(f"Created {original_count} entities")

    # Initialize embedder for blocking (shared across iterations)
    embedder = EntityEmbedder()

    iteration_metrics: list[IterationMetrics] = []
    prev_reduction_pct = 100.0  # Track previous round's reduction for auto-convergence

    # max_iterations=0 means auto-convergence (up to 20 iterations)
    max_iters = cfg.max_iterations if cfg.max_iterations > 0 else 20

    for iteration in range(1, max_iters + 1):
        iter_start = time.time()
        logger.info(f"\n=== Iteration {iteration} ===")
        logger.info(f"  Entities: {len(entities)}")

        # Phase 1: Embed for blocking (name-only by default, configurable)
        logger.info("  Embedding for blocking...")
        texts = [e.text_for_embedding(cfg.blocking_fields) for e in entities]
        embeddings = embedder.embed(texts)

        # Phase 2: Block with FAISS
        logger.info("  Blocking with FAISS...")
        ids = [str(e.id) for e in entities]
        effective_target = max(10, cfg.target_block_size // iteration)
        blocker = FAISSBlocker(
            target_block_size=effective_target,
            iteration=iteration,
            auto_scale=False,
        )
        block_assignments = blocker.block(embeddings, ids)

        # Build EntityBlocks
        entity_map = {e.id: e for e in entities}
        blocks: list[EntityBlock] = []
        for bk, eids in block_assignments.items():
            block_ents = [entity_map[int(eid)] for eid in eids]
            blocks.append(
                EntityBlock(
                    block_key=bk,
                    block_key_type="semantic",
                    block_size=len(block_ents),
                    entities=block_ents,
                )
            )

        logger.info(f"  Created {len(blocks)} blocks")

        # Phase 3: Match with LLM
        resolved = _llm_match_and_merge(blocks, cfg)

        # Compute iteration metrics
        reduction = len(entities) - len(resolved)
        reduction_pct = (reduction / len(entities) * 100) if entities else 0.0
        overall_pct = (
            (original_count - len(resolved)) / original_count * 100 if original_count > 0 else 0.0
        )

        metrics = IterationMetrics(
            iteration=iteration,
            input_entities=len(entities),
            output_entities=len(resolved),
            reduction_pct=reduction_pct,
            overall_reduction_pct=overall_pct,
            blocks_count=len(blocks),
        )
        iteration_metrics.append(metrics)

        iter_elapsed = time.time() - iter_start
        logger.info(
            f"  Iteration {iteration}: {len(entities)} → {len(resolved)} "
            f"({reduction_pct:.1f}% reduction, {iter_elapsed:.1f}s)"
        )

        # Check convergence: stop when reduction drops below threshold
        converged = False
        if reduction_pct < cfg.convergence_threshold * 100:
            logger.info(
                f"  Converged (below threshold): {reduction_pct:.2f}% < "
                f"{cfg.convergence_threshold * 100:.2f}%"
            )
            converged = True

        # Auto-convergence: stop when reduction plateaus (no improvement)
        is_auto = cfg.max_iterations == 0
        is_plateau = reduction_pct <= 0 or (
            prev_reduction_pct > 0 and reduction_pct / prev_reduction_pct < 0.1
        )
        if is_auto and iteration > 1 and is_plateau:
            logger.info(
                f"  Auto-converged (plateau): reduction dropped from "
                f"{prev_reduction_pct:.2f}% to {reduction_pct:.2f}%"
            )
            converged = True

        prev_reduction_pct = reduction_pct

        if converged:
            entities = resolved
            break

        # Re-assign sequential IDs for next iteration
        for i, e in enumerate(resolved):
            e.id = i
        entities = resolved

    # Write output
    os.makedirs(output_path, exist_ok=True)
    _write_output(entities, output_path)

    elapsed = time.time() - start
    summary: dict[str, Any] = {
        "input_path": input_path,
        "output_path": output_path,
        "original_count": original_count,
        "final_count": len(entities),
        "overall_reduction_pct": (
            (original_count - len(entities)) / original_count * 100 if original_count > 0 else 0.0
        ),
        "iterations": len(iteration_metrics),
        "elapsed_seconds": elapsed,
        "iteration_metrics": [m.model_dump() for m in iteration_metrics],
    }

    summary_path = os.path.join(output_path, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"\nPipeline complete: {original_count} → {len(entities)} entities "
        f"({summary['overall_reduction_pct']:.1f}% reduction) in {elapsed:.1f}s"
    )

    return summary


def _llm_match_and_merge(blocks: list[EntityBlock], cfg: ERConfig) -> list[Entity]:
    """Run LLM-based matching on blocks and return resolved entities.

    Parameters
    ----------
    blocks : list[EntityBlock]
        Blocks to process
    cfg : ERConfig
        Pipeline configuration

    Returns
    -------
    list[Entity]
        Resolved entities after LLM matching and merging
    """
    from serf.match.matcher import EntityMatcher

    logger.info("  Matching with LLM...")
    matcher = EntityMatcher(model=cfg.model)
    resolutions = asyncio.run(matcher.resolve_blocks(blocks))

    resolved: list[Entity] = []
    for r in resolutions:
        resolved.extend(r.resolved_entities)
    return resolved


def _write_output(entities: list[Entity], output_path: str) -> None:
    """Write resolved entities to Parquet and CSV.

    Parameters
    ----------
    entities : list[Entity]
        Resolved entities
    output_path : str
        Output directory
    """
    records = []
    for e in entities:
        row: dict[str, Any] = {
            "id": e.id,
            "name": e.name,
            "description": e.description,
            "entity_type": e.entity_type,
        }
        for k, v in e.attributes.items():
            row[k] = v
        if e.source_ids:
            row["source_ids"] = json.dumps(e.source_ids)
        if e.source_uuids:
            row["source_uuids"] = json.dumps(e.source_uuids)
        records.append(row)

    df = pd.DataFrame(records)
    df.to_parquet(os.path.join(output_path, "resolved.parquet"), index=False)
    df.to_csv(os.path.join(output_path, "resolved.csv"), index=False)
    logger.info(f"  Wrote {len(df)} resolved entities to {output_path}")
