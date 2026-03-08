"""Main CLI entry point for SERF."""

import json
import os
import time
from typing import Any

import click

from serf.logs import get_logger, setup_logging

logger = get_logger(__name__)


@click.group(context_settings={"show_default": True})
@click.version_option()
def cli() -> None:
    """SERF: Semantic Entity Resolution Framework CLI."""
    setup_logging()


# ---------------------------------------------------------------------------
# run  (main entry point for end-to-end ER on any data)
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input data file (CSV, Parquet) or Iceberg URI",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=True,
    help="Output directory for resolved entities",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    required=False,
    help="ER config YAML file with field mappings and parameters",
)
@click.option("--name-field", type=str, required=False, help="Column to use as entity name")
@click.option(
    "--text-fields",
    type=str,
    required=False,
    help="Comma-separated columns for embedding text",
)
@click.option("--entity-type", type=str, default="entity", help="Entity type label")
@click.option(
    "--model",
    type=str,
    default="gemini/gemini-2.0-flash",
    help="LLM model for matching",
)
@click.option("--max-iterations", type=int, default=3, help="Maximum ER iterations")
@click.option(
    "--convergence-threshold",
    type=float,
    default=0.01,
    help="Stop when per-round reduction fraction is below this",
)
@click.option(
    "--target-block-size",
    type=int,
    default=50,
    help="Target entities per FAISS block",
)
def run(
    input_path: str,
    output_path: str,
    config_path: str | None,
    name_field: str | None,
    text_fields: str | None,
    entity_type: str,
    model: str,
    max_iterations: int,
    convergence_threshold: float,
    target_block_size: int,
) -> None:
    """Run entity resolution on any CSV, Parquet, or Iceberg table.

    Uses embeddings for blocking (FAISS clustering) and LLM for matching
    (DSPy BlockMatch). Runs iterative rounds until convergence.
    Writes resolved entities as Parquet and CSV.

    Requires GEMINI_API_KEY environment variable (or appropriate key for the model).
    """
    from serf.pipeline import ERConfig, run_pipeline

    # Build config: YAML file first, then CLI overrides
    er_config = ERConfig.from_yaml(config_path) if config_path else ERConfig()

    # CLI flags override config file values
    if name_field:
        er_config.name_field = name_field
    if text_fields:
        er_config.text_fields = [f.strip() for f in text_fields.split(",")]
    er_config.entity_type = entity_type
    er_config.model = model
    er_config.max_iterations = max_iterations
    er_config.convergence_threshold = convergence_threshold
    er_config.target_block_size = target_block_size

    click.echo("SERF Entity Resolution")
    click.echo(f"  Input:  {input_path}")
    click.echo(f"  Output: {output_path}")
    click.echo(f"  Model:  {model}")
    if config_path:
        click.echo(f"  Config: {config_path}")

    summary = run_pipeline(input_path, output_path, er_config)

    click.echo(f"\n{'=' * 50}")
    click.echo(f"  Original entities:  {summary['original_count']}")
    click.echo(f"  Resolved entities:  {summary['final_count']}")
    click.echo(f"  Overall reduction:  {summary['overall_reduction_pct']:.1f}%")
    click.echo(f"  Iterations:         {summary['iterations']}")
    click.echo(f"  Elapsed:            {summary['elapsed_seconds']:.1f}s")
    click.echo(f"{'=' * 50}")
    click.echo(f"\nResults written to {output_path}/")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input data file (CSV or Parquet)",
)
def analyze(input_path: str) -> None:
    """Profile a dataset and recommend ER strategy."""
    import pandas as pd

    from serf.analyze.profiler import DatasetProfiler

    logger.info(f"Analyzing dataset: {input_path}")
    start = time.time()

    df = pd.read_parquet(input_path) if input_path.endswith(".parquet") else pd.read_csv(input_path)

    records = df.to_dict("records")
    profiler = DatasetProfiler()
    profile = profiler.profile(records)

    elapsed = time.time() - start
    click.echo(f"\nDataset Profile ({elapsed:.1f}s)")
    click.echo(f"  Records: {profile.record_count}")
    click.echo(f"  Fields: {len(profile.field_profiles)}")
    click.echo(f"  Estimated duplicate rate: {profile.estimated_duplicate_rate:.1%}")
    click.echo(f"\n  Recommended blocking fields: {profile.recommended_blocking_fields}")
    click.echo(f"  Recommended matching fields: {profile.recommended_matching_fields}")
    click.echo("\n  Field details:")
    for fp in profile.field_profiles:
        click.echo(
            f"    {fp.name}: type={fp.inferred_type}, "
            f"completeness={fp.completeness:.1%}, "
            f"uniqueness={fp.uniqueness:.1%}"
        )


# ---------------------------------------------------------------------------
# block
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input data file or directory",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=True,
    help="Output directory for results",
)
@click.option("--iteration", type=int, default=1, help="ER iteration number")
@click.option(
    "--method",
    type=click.Choice(["semantic", "name"]),
    default="semantic",
    help="Blocking method to use",
)
@click.option("--target-block-size", type=int, default=50, help="Target entities per block")
@click.option("--max-block-size", type=int, default=200, help="Maximum entities per block")
def block(
    input_path: str,
    output_path: str,
    iteration: int,
    method: str,
    target_block_size: int,
    max_block_size: int,
) -> None:
    """Perform semantic blocking on input data."""
    import pandas as pd

    from serf.block.pipeline import SemanticBlockingPipeline

    logger.info(f"Starting blocking: input={input_path}, method={method}")
    start = time.time()

    df = pd.read_parquet(input_path) if input_path.endswith(".parquet") else pd.read_csv(input_path)

    entities = _dataframe_to_entities(df)

    pipeline = SemanticBlockingPipeline(
        target_block_size=target_block_size,
        max_block_size=max_block_size,
        iteration=iteration,
    )
    blocks, metrics = pipeline.run(entities)

    os.makedirs(output_path, exist_ok=True)
    blocks_file = os.path.join(output_path, "blocks.jsonl")
    with open(blocks_file, "w") as f:
        for b in blocks:
            f.write(b.model_dump_json() + "\n")

    elapsed = time.time() - start
    click.echo(f"\nBlocking complete ({elapsed:.1f}s)")
    click.echo(f"  Input entities: {len(entities)}")
    click.echo(f"  Blocks created: {metrics.total_blocks}")
    click.echo(f"  Avg block size: {metrics.avg_block_size:.1f}")
    click.echo(f"  Max block size: {metrics.max_block_size}")
    click.echo(f"  Reduction ratio: {metrics.reduction_ratio:.4f}")


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------


@cli.command(name="match")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input directory with blocked data",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=True,
    help="Output directory for matched results",
)
@click.option("--iteration", type=int, default=1, help="ER iteration number")
@click.option(
    "--batch-size",
    type=int,
    default=10,
    help="Number of blocks to process concurrently",
)
def match(input_path: str, output_path: str, iteration: int, batch_size: int) -> None:
    """Match entities within blocks using LLM."""
    import asyncio

    from serf.dspy.types import EntityBlock
    from serf.match.matcher import EntityMatcher

    logger.info(f"Starting matching: input={input_path}, iteration={iteration}")
    start = time.time()

    blocks_file = os.path.join(input_path, "blocks.jsonl")
    blocks = []
    with open(blocks_file) as f:
        for line in f:
            blocks.append(EntityBlock.model_validate_json(line.strip()))

    matcher = EntityMatcher(batch_size=batch_size)
    resolutions = asyncio.run(matcher.resolve_blocks(blocks))

    os.makedirs(output_path, exist_ok=True)
    matches_file = os.path.join(output_path, "matches.jsonl")
    with open(matches_file, "w") as f:
        for r in resolutions:
            f.write(r.model_dump_json() + "\n")

    total_input = sum(r.original_count for r in resolutions)
    total_output = sum(r.resolved_count for r in resolutions)
    elapsed = time.time() - start
    click.echo(f"\nMatching complete ({elapsed:.1f}s)")
    click.echo(f"  Blocks processed: {len(resolutions)}")
    click.echo(f"  Entities in: {total_input}")
    click.echo(f"  Entities out: {total_output}")
    if total_input > 0:
        click.echo(f"  Reduction: {(1 - total_output / total_input) * 100:.1f}%")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@cli.command(name="eval")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input directory with match results",
)
@click.option(
    "--ground-truth",
    "-g",
    type=click.Path(exists=True),
    required=False,
    help="Ground truth file with labeled pairs (CSV)",
)
def evaluate(input_path: str, ground_truth: str | None) -> None:
    """Evaluate entity resolution results."""
    from serf.dspy.types import BlockResolution
    from serf.eval.metrics import evaluate_resolution

    logger.info(f"Evaluating: input={input_path}")

    matches_file = os.path.join(input_path, "matches.jsonl")
    resolutions = []
    with open(matches_file) as f:
        for line in f:
            resolutions.append(BlockResolution.model_validate_json(line.strip()))

    total_input = sum(r.original_count for r in resolutions)
    total_output = sum(r.resolved_count for r in resolutions)
    resolved_count = sum(1 for r in resolutions if r.was_resolved)

    click.echo("\nEvaluation Summary")
    click.echo(f"  Total blocks: {len(resolutions)}")
    click.echo(f"  Blocks with merges: {resolved_count}")
    click.echo(f"  Entities in: {total_input}")
    click.echo(f"  Entities out: {total_output}")
    if total_input > 0:
        click.echo(f"  Reduction: {(1 - total_output / total_input) * 100:.1f}%")

    if ground_truth:
        import pandas as pd

        gt_df = pd.read_csv(ground_truth)
        true_pairs: set[tuple[int, int]] = set()
        for _, row in gt_df.iterrows():
            if row.get("label", 0) == 1:
                a, b = int(row["ltable_id"]), int(row["rtable_id"])
                true_pairs.add((min(a, b), max(a, b)))

        predicted_pairs: set[tuple[int, int]] = set()
        for r in resolutions:
            for m in r.matches:
                if m.is_match:
                    a, b = m.entity_a_id, m.entity_b_id
                    predicted_pairs.add((min(a, b), max(a, b)))

        metrics = evaluate_resolution(predicted_pairs, true_pairs)
        click.echo(f"\n  Precision: {metrics['precision']:.4f}")
        click.echo(f"  Recall: {metrics['recall']:.4f}")
        click.echo(f"  F1 Score: {metrics['f1_score']:.4f}")


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input directory with resolved entities",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=True,
    help="Output directory for resolved edges",
)
def edges(input_path: str, output_path: str) -> None:
    """Resolve edges after node merging."""
    logger.info(f"Starting edge resolution: input={input_path}")
    click.echo(f"Resolving edges from {input_path} to {output_path}")
    click.echo("Edge resolution requires edges data. Use the Python API for full edge resolution.")


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input data file or directory",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=True,
    help="Output directory for results",
)
@click.option("--iteration", type=int, default=1, help="ER iteration number")
@click.option(
    "--method",
    type=click.Choice(["semantic", "name"]),
    default="semantic",
    help="Blocking method",
)
@click.option("--target-block-size", type=int, default=50, help="Target entities per block")
@click.option(
    "--batch-size",
    type=int,
    default=10,
    help="Concurrent block processing batch size",
)
def resolve(
    input_path: str,
    output_path: str,
    iteration: int,
    method: str,
    target_block_size: int,
    batch_size: int,
) -> None:
    """Run the full ER pipeline: block -> match -> evaluate."""
    click.echo(f"Running full ER pipeline (iteration {iteration})")
    click.echo("  Step 1: Blocking...")
    ctx = click.get_current_context()
    ctx.invoke(
        block,
        input_path=input_path,
        output_path=f"{output_path}/blocks",
        iteration=iteration,
        method=method,
        target_block_size=target_block_size,
        max_block_size=200,
    )
    click.echo("\n  Step 2: Matching...")
    ctx.invoke(
        match,
        input_path=f"{output_path}/blocks",
        output_path=f"{output_path}/matches",
        iteration=iteration,
        batch_size=batch_size,
    )
    click.echo("\n  Step 3: Evaluating...")
    ctx.invoke(
        evaluate,
        input_path=f"{output_path}/matches",
        ground_truth=None,
    )


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--dataset",
    "-d",
    type=str,
    required=True,
    help="Benchmark dataset name to download",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=False,
    help="Output directory for downloaded data",
)
def download(dataset: str, output_path: str | None) -> None:
    """Download a benchmark dataset."""
    from serf.eval.benchmarks import BenchmarkDataset

    available = BenchmarkDataset.available_datasets()
    if dataset not in available:
        click.echo(f"Unknown dataset: {dataset}")
        click.echo(f"Available: {', '.join(available)}")
        return

    click.echo(f"Downloading {dataset}...")
    benchmark_data = BenchmarkDataset.download(dataset, output_path)
    click.echo(f"  Left table: {len(benchmark_data.table_a)} records")
    click.echo(f"  Right table: {len(benchmark_data.table_b)} records")
    click.echo(f"  Ground truth pairs: {len(benchmark_data.ground_truth)}")
    click.echo("Done.")


# ---------------------------------------------------------------------------
# benchmark  (single dataset, LLM matching)
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--dataset",
    "-d",
    type=str,
    required=True,
    help="Benchmark dataset name",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=False,
    help="Output directory for results",
)
@click.option(
    "--target-block-size",
    type=int,
    default=50,
    help="Target entities per block",
)
@click.option(
    "--model",
    type=str,
    default="gemini/gemini-2.0-flash",
    help="LLM model for matching",
)
@click.option(
    "--max-right-entities",
    type=int,
    default=None,
    help="Limit right table size for large datasets",
)
def benchmark(
    dataset: str,
    output_path: str | None,
    target_block_size: int,
    model: str,
    max_right_entities: int | None,
) -> None:
    """Run ER pipeline against a benchmark dataset and evaluate.

    Uses embeddings for blocking and LLM for matching.
    Requires GEMINI_API_KEY environment variable (or appropriate key for the model).
    """
    from serf.eval.benchmarks import BenchmarkDataset

    available = BenchmarkDataset.available_datasets()
    if dataset not in available:
        click.echo(f"Unknown dataset: {dataset}")
        click.echo(f"Available: {', '.join(available)}")
        return

    click.echo(f"Running benchmark: {dataset}")
    click.echo(f"  Model: {model}")
    start = time.time()

    benchmark_data = BenchmarkDataset.download(dataset, output_path)
    left_entities, right_entities = benchmark_data.to_entities()

    # Optionally limit right table size
    if max_right_entities and len(right_entities) > max_right_entities:
        import random

        gt_right_ids = {b for _, b in benchmark_data.ground_truth}
        matched = [e for e in right_entities if e.id in gt_right_ids]
        unmatched = [e for e in right_entities if e.id not in gt_right_ids]
        sample_size = max(0, max_right_entities - len(matched))
        random.seed(42)
        sampled = random.sample(unmatched, min(sample_size, len(unmatched)))
        right_entities = matched + sampled
        click.echo(f"  Sampled right table to {len(right_entities)} entities")

    all_entities = left_entities + right_entities
    click.echo(f"  Left table: {len(left_entities)} entities")
    click.echo(f"  Right table: {len(right_entities)} entities")
    click.echo(f"  Ground truth pairs: {len(benchmark_data.ground_truth)}")
    click.echo(f"  Total entities: {len(all_entities)}")

    predicted_pairs = _benchmark_llm_matching(all_entities, target_block_size, model)

    metrics = benchmark_data.evaluate(predicted_pairs)
    elapsed = time.time() - start

    click.echo(f"\n  Benchmark Results ({elapsed:.1f}s):")
    click.echo(f"    Precision: {metrics['precision']:.4f}")
    click.echo(f"    Recall:    {metrics['recall']:.4f}")
    click.echo(f"    F1 Score:  {metrics['f1_score']:.4f}")
    click.echo(f"    Predicted pairs: {len(predicted_pairs)}")

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        results_file = os.path.join(output_path, f"{dataset}_results.json")
        with open(results_file, "w") as f:
            json.dump(
                {
                    "dataset": dataset,
                    "model": model,
                    "elapsed_seconds": elapsed,
                    "predicted_pairs": len(predicted_pairs),
                    "true_pairs": len(benchmark_data.ground_truth),
                    **metrics,
                },
                f,
                indent=2,
            )
        click.echo(f"\n  Results saved to {results_file}")


# ---------------------------------------------------------------------------
# benchmark-all  (run all datasets)
# ---------------------------------------------------------------------------


@cli.command(name="benchmark-all")
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    default="data/benchmarks",
    help="Output directory for results",
)
@click.option(
    "--model",
    type=str,
    default="gemini/gemini-2.0-flash",
    help="LLM model for matching",
)
@click.option(
    "--max-right-entities",
    type=int,
    default=5000,
    help="Limit right table size for large datasets",
)
def benchmark_all(
    output_path: str,
    model: str,
    max_right_entities: int,
) -> None:
    """Run LLM-based benchmarks on all available datasets.

    Requires GEMINI_API_KEY environment variable (or appropriate key for the model).
    """
    from serf.eval.benchmarks import BenchmarkDataset

    datasets = BenchmarkDataset.available_datasets()
    click.echo(f"Running benchmarks on {len(datasets)} datasets...")
    click.echo(f"  Model: {model}")
    click.echo(f"  Max right entities: {max_right_entities}")

    results: dict[str, dict[str, float]] = {}
    for name in datasets:
        click.echo(f"\n{'=' * 60}")
        ctx = click.get_current_context()
        ctx.invoke(
            benchmark,
            dataset=name,
            output_path=output_path,
            target_block_size=15,
            model=model,
            max_right_entities=max_right_entities,
        )

        # Load saved results
        results_file = os.path.join(output_path, f"{name}_results.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                results[name] = json.load(f)

    # Print summary table
    click.echo(f"\n{'=' * 70}")
    click.echo(f"{'Dataset':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>10}")
    click.echo("-" * 70)
    for name, m in results.items():
        if "error" in m:
            click.echo(f"{name:<20} {'ERROR':>10}")
        else:
            click.echo(
                f"{name:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1_score']:>10.4f} {m['elapsed_seconds']:>9.1f}s"
            )
    click.echo("=" * 70)

    # Save combined results
    os.makedirs(output_path, exist_ok=True)
    combined_file = os.path.join(output_path, "all_results.json")
    with open(combined_file, "w") as f:
        json.dump(results, f, indent=2)
    click.echo(f"\nCombined results saved to {combined_file}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_name_column(columns: list[str]) -> str:
    """Detect the primary name column from a list of column names.

    Parameters
    ----------
    columns : list[str]
        Column names to search

    Returns
    -------
    str
        The detected name column
    """
    name_candidates = [
        "title",
        "name",
        "product_name",
        "company_name",
        "entity_name",
    ]
    for candidate in name_candidates:
        if candidate in columns:
            return candidate
    for col in columns:
        if col != "id":
            return col
    return columns[0]


def _dataframe_to_entities(df: Any) -> list[Any]:
    """Convert a pandas DataFrame to a list of Entity objects.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with entity records

    Returns
    -------
    list[Entity]
        List of Entity objects
    """

    from serf.dspy.types import Entity

    entities = []
    name_col = _detect_name_column(df.columns.tolist())
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        name = str(row_dict.get(name_col, f"entity_{idx}"))
        desc_parts = [
            str(v) for k, v in row_dict.items() if k != name_col and isinstance(v, str) and v
        ]
        entities.append(
            Entity(
                id=int(row_dict.get("id", idx)),  # type: ignore[arg-type]
                name=name,
                description=" ".join(desc_parts),
                attributes=row_dict,
            )
        )
    return entities


def _benchmark_llm_matching(
    all_entities: list[Any],
    target_block_size: int,
    model: str = "gemini/gemini-2.0-flash",
) -> set[tuple[int, int]]:
    """Run LLM-based matching for benchmarks.

    Embeddings are used for blocking only. Matching is done by LLM.

    Parameters
    ----------
    all_entities : list[Entity]
        All entities to match
    target_block_size : int
        Target block size for FAISS blocking
    model : str
        LLM model name

    Returns
    -------
    set[tuple[int, int]]
        Predicted match pairs
    """
    import asyncio

    from serf.block.pipeline import SemanticBlockingPipeline
    from serf.match.matcher import EntityMatcher

    click.echo("\n  Blocking (embeddings + FAISS)...")
    pipeline = SemanticBlockingPipeline(target_block_size=target_block_size, max_block_size=200)
    blocks, blocking_metrics = pipeline.run(all_entities)
    click.echo(f"    {blocking_metrics.total_blocks} blocks created")

    click.echo(f"  Matching with LLM ({model})...")
    matcher = EntityMatcher(model=model)
    resolutions = asyncio.run(matcher.resolve_blocks(blocks))

    predicted_pairs: set[tuple[int, int]] = set()
    for r in resolutions:
        for m in r.matches:
            if m.is_match:
                a, b = m.entity_a_id, m.entity_b_id
                predicted_pairs.add((min(a, b), max(a, b)))

    click.echo(f"    Predicted {len(predicted_pairs)} match pairs")
    return predicted_pairs


if __name__ == "__main__":
    cli()
