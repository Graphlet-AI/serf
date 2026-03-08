"""End-to-end entity resolution pipeline.

Takes any tabular data (CSV, Parquet, or Iceberg) and runs the full
blocking → matching → merging pipeline with iterative convergence.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from serf.block.embeddings import EntityEmbedder
from serf.block.faiss_blocker import FAISSBlocker
from serf.dspy.types import Entity, EntityBlock, IterationMetrics
from serf.logs import get_logger
from serf.merge.merger import EntityMerger

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
    matching_mode : str
        "embedding" for cosine similarity, "llm" for LLM-based matching.
    similarity_threshold : float
        Cosine similarity threshold for embedding mode.
    model : str
        LLM model name for llm mode.
    max_iterations : int
        Maximum ER iterations.
    convergence_threshold : float
        Stop when per-round reduction falls below this.
    """

    def __init__(
        self,
        name_field: str | None = None,
        text_fields: list[str] | None = None,
        entity_type: str = "entity",
        blocking_method: str = "semantic",
        target_block_size: int = 50,
        max_block_size: int = 200,
        matching_mode: str = "embedding",
        similarity_threshold: float = 0.85,
        model: str = "gemini/gemini-2.0-flash",
        max_iterations: int = 3,
        convergence_threshold: float = 0.01,
    ) -> None:
        self.name_field = name_field
        self.text_fields = text_fields
        self.entity_type = entity_type
        self.blocking_method = blocking_method
        self.target_block_size = target_block_size
        self.max_block_size = max_block_size
        self.matching_mode = matching_mode
        self.similarity_threshold = similarity_threshold
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
            entity_type=data.get("entity_type", "entity"),
            blocking_method=blocking.get("method", "semantic"),
            target_block_size=blocking.get("target_block_size", 50),
            max_block_size=blocking.get("max_block_size", 200),
            matching_mode=matching.get("mode", "embedding"),
            similarity_threshold=matching.get("similarity_threshold", 0.85),
            model=matching.get("model", "gemini/gemini-2.0-flash"),
            max_iterations=data.get("max_iterations", 3),
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
        if str(col).lower() != "id" and df[col].dtype in ("object", "str", "string"):
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

        # Convert all values to strings for attributes
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


def _embedding_match_within_blocks(
    blocks: list[EntityBlock],
    embeddings: np.ndarray,
    entity_id_to_idx: dict[int, int],
    similarity_threshold: float,
) -> list[tuple[int, int]]:
    """Match entities within blocks using embedding cosine similarity.

    Parameters
    ----------
    blocks : list[EntityBlock]
        Blocks to match within
    embeddings : np.ndarray
        All entity embeddings
    entity_id_to_idx : dict[int, int]
        Map from entity ID to embedding index
    similarity_threshold : float
        Minimum cosine similarity to consider a match

    Returns
    -------
    list[tuple[int, int]]
        List of (entity_a_id, entity_b_id) match pairs
    """
    match_pairs: list[tuple[int, int]] = []
    for blk in blocks:
        if blk.block_size < 2:
            continue
        ents = blk.entities
        idxs = [entity_id_to_idx[e.id] for e in ents]
        block_embs = embeddings[idxs]
        sim = np.dot(block_embs, block_embs.T)
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                if sim[i, j] >= similarity_threshold:
                    match_pairs.append((ents[i].id, ents[j].id))
    return match_pairs


def _merge_matched_entities(
    entities: list[Entity],
    match_pairs: list[tuple[int, int]],
) -> list[Entity]:
    """Merge matched entities using union-find for transitive closure.

    Parameters
    ----------
    entities : list[Entity]
        All entities
    match_pairs : list[tuple[int, int]]
        Pairs of matching entity IDs

    Returns
    -------
    list[Entity]
        Merged entities
    """
    if not match_pairs:
        return list(entities)

    # Union-find for transitive closure
    parent: dict[int, int] = {e.id: e.id for e in entities}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for a, b in match_pairs:
        union(a, b)

    # Group entities by their root
    groups: dict[int, list[Entity]] = {}
    for e in entities:
        root = find(e.id)
        if root not in groups:
            groups[root] = []
        groups[root].append(e)

    # Merge each group
    merger = EntityMerger()
    resolved: list[Entity] = []
    for group_entities in groups.values():
        if len(group_entities) == 1:
            resolved.append(group_entities[0])
        else:
            merged = merger.merge_entities(group_entities)
            resolved.append(merged)

    return resolved


def run_pipeline(
    input_path: str,
    output_path: str,
    er_config: ERConfig | None = None,
) -> dict[str, Any]:
    """Run the full entity resolution pipeline.

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

    # Initialize embedder (shared across iterations)
    embedder = EntityEmbedder()

    iteration_metrics: list[IterationMetrics] = []

    for iteration in range(1, cfg.max_iterations + 1):
        iter_start = time.time()
        logger.info(f"\n=== Iteration {iteration} ===")
        logger.info(f"  Entities: {len(entities)}")

        # Embed
        logger.info("  Embedding...")
        texts = [e.text_for_embedding() for e in entities]
        embeddings = embedder.embed(texts)
        entity_id_to_idx = {e.id: i for i, e in enumerate(entities)}

        # Block
        logger.info("  Blocking...")
        ids = [str(e.id) for e in entities]
        effective_target = max(10, cfg.target_block_size // iteration)
        blocker = FAISSBlocker(
            target_block_size=effective_target,
            iteration=iteration,
            auto_scale=False,  # We handle scaling above
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

        # Match
        if cfg.matching_mode == "llm":
            resolved = _llm_match_and_merge(blocks, cfg)
        else:
            match_pairs = _embedding_match_within_blocks(
                blocks, embeddings, entity_id_to_idx, cfg.similarity_threshold
            )
            logger.info(f"  Found {len(match_pairs)} match pairs")
            resolved = _merge_matched_entities(entities, match_pairs)

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

        # Check convergence
        if reduction_pct < cfg.convergence_threshold * 100:
            logger.info(
                f"  Converged: {reduction_pct:.2f}% < "
                f"{cfg.convergence_threshold * 100:.2f}% threshold"
            )
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

    # Save summary
    summary_path = os.path.join(output_path, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"\nPipeline complete: {original_count} → {len(entities)} entities "
        f"({summary['overall_reduction_pct']:.1f}% reduction) in {elapsed:.1f}s"
    )

    return summary


def _llm_match_and_merge(blocks: list[EntityBlock], cfg: ERConfig) -> list[Entity]:
    """Run LLM-based matching and return resolved entities.

    Parameters
    ----------
    blocks : list[EntityBlock]
        Blocks to process
    cfg : ERConfig
        Pipeline configuration

    Returns
    -------
    list[Entity]
        Resolved entities after LLM matching
    """
    import asyncio

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
        # Flatten attributes into columns
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
