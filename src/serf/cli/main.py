"""Main CLI entry point for SERF."""

import json
import os
import subprocess
import sys
import time
from typing import Any, cast

import click
import pandas as pd

from serf.config import config as serf_config
from serf.dspy.dataset_signatures import SIGNATURE_MODE_GENERIC, SIGNATURE_MODES
from serf.eval.benchmarks import DATASET_REGISTRY
from serf.logs import get_logger, setup_logging
from serf.match.run import entity_members, expand_pairs, merge_matched_entities
from serf.tracking import setup_mlflow

logger = get_logger(__name__)

# Available benchmark dataset names for CLI help
BENCHMARK_DATASETS = list(DATASET_REGISTRY.keys())


@click.group(context_settings={"show_default": True})
@click.version_option()
def cli() -> None:
    """SERF: Semantic Entity Resolution Framework CLI."""
    setup_logging()


# ---------------------------------------------------------------------------
# mlflow  (start local MLflow server)
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--host",
    type=str,
    default=None,
    help="Host to bind the MLflow server to (from config.yml mlflow.host)",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to run the MLflow server on (from config.yml mlflow.port)",
)
@click.option(
    "--backend-store-uri",
    type=str,
    default=None,
    help="Backend store URI for MLflow (from config.yml mlflow.backend_store_uri)",
)
def mlflow(host: str | None, port: int | None, backend_store_uri: str | None) -> None:
    """Start a local MLflow tracking server.

    Runs `mlflow server` with a SQLite backend for tracing DSPy operations.
    The UI will be available at http://<host>:<port>.
    """
    host = host or serf_config.get("mlflow.host", "127.0.0.1")
    port = port or serf_config.get("mlflow.port", 5000)
    backend_store_uri = backend_store_uri or serf_config.get(
        "mlflow.backend_store_uri", "sqlite:///mlflow.db"
    )

    click.echo(f"Starting MLflow server at http://{host}:{port}")
    click.echo(f"  Backend store: {backend_store_uri}")
    click.echo("  Press Ctrl+C to stop\n")

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        str(backend_store_uri),
        "--host",
        str(host),
        "--port",
        str(port),
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        click.echo("\nMLflow server stopped.")
    except subprocess.CalledProcessError as e:
        click.echo(f"MLflow server exited with code {e.returncode}", err=True)
        raise SystemExit(e.returncode) from e


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
    default=None,
    help="LLM model for matching (from config.yml models.llm)",
)
@click.option(
    "--max-iterations",
    type=int,
    default=5,
    help="Maximum ER iterations (0 for auto-convergence)",
)
@click.option(
    "--convergence-threshold",
    type=float,
    default=0.01,
    help="Stop when per-round reduction fraction is below this",
)
@click.option(
    "--target-block-size",
    type=int,
    default=30,
    help="Target entities per FAISS block",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Max blocks to send through LLM matching (for testing)",
)
@click.option(
    "--concurrency",
    type=int,
    default=20,
    help="Number of concurrent LLM requests",
)
def run(
    input_path: str,
    output_path: str,
    config_path: str | None,
    name_field: str | None,
    text_fields: str | None,
    entity_type: str,
    model: str | None,
    max_iterations: int,
    convergence_threshold: float,
    target_block_size: int,
    limit: int | None,
    concurrency: int,
) -> None:
    """Run entity resolution on any CSV, Parquet, or Iceberg table.

    Uses embeddings for blocking (FAISS clustering) and LLM for matching
    (DSPy BlockMatch). Runs iterative rounds until convergence.
    Writes resolved entities as Parquet and CSV.

    Requires VERTEX_AI_TOKEN for GPT OSS 120b (student) and GEMINI_API_KEY
    for Gemini 3.5 Flash-Lite (teacher/analyze).
    """
    from serf.pipeline import ERConfig, run_pipeline

    setup_mlflow()

    # Build config: YAML file first, then CLI overrides
    er_config = ERConfig.from_yaml(config_path) if config_path else ERConfig()

    # CLI flags override config file values
    if name_field:
        er_config.name_field = name_field
    if text_fields:
        er_config.text_fields = [f.strip() for f in text_fields.split(",")]
    er_config.entity_type = entity_type
    if model:
        er_config.model = model
    er_config.max_iterations = max_iterations
    er_config.convergence_threshold = convergence_threshold
    er_config.target_block_size = target_block_size
    er_config.max_concurrent = concurrency
    er_config.limit = limit

    # Auto-scale block size for limited test runs
    if limit and limit <= 20 and er_config.target_block_size >= 20:
        er_config.target_block_size = 5

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
    help="Input data file (CSV, Parquet, or Iceberg URI)",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=False,
    help="Write LLM-generated ER config YAML to this path",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model for config generation (from config.yml models.analyze_llm)",
)
def analyze(input_path: str, output_path: str | None, model: str | None) -> None:
    """Profile a dataset and generate an ER configuration.

    Runs statistical profiling on the input data, then optionally uses an LLM
    to generate a ready-to-use ER config YAML. The generated config can be
    passed directly to `serf run --config`.

    Without --output, prints the statistical profile only.
    With --output, also calls the LLM to generate an ER config YAML file.
    """
    from serf.analyze.profiler import DatasetProfiler, generate_er_config
    from serf.pipeline import load_data

    logger.info(f"Analyzing dataset: {input_path}")
    start = time.time()

    df = load_data(input_path)
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

    if output_path:
        click.echo(f"\n  Generating ER config with LLM ({model})...")
        sample_records = records[:10]
        config_yaml = generate_er_config(profile, sample_records, model=model)

        with open(output_path, "w") as f:
            f.write(config_yaml + "\n")

        click.echo(f"\n  ER config written to {output_path}")
        click.echo(
            f"  Run: serf run --input {input_path} --output data/resolved/ --config {output_path}"
        )
        click.echo("\n  Generated config:\n")
        for line in config_yaml.split("\n"):
            click.echo(f"    {line}")


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
@click.option("--target-block-size", type=int, default=30, help="Target entities per block")
@click.option("--max-block-size", type=int, default=100, help="Maximum entities per block")
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

    setup_mlflow()

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
    """Evaluate entity resolution results with Abzu-level rigor.

    Performs comprehensive validation: entity deduplication, source_uuid
    validation, match_skip analysis, and PASS/FAIL checks.
    """
    from serf.dspy.types import BlockResolution
    from serf.eval.evaluator import (
        evaluate_er_results,
        format_evaluation_report,
        save_evaluation,
    )
    from serf.eval.metrics import evaluate_resolution

    logger.info(f"Evaluating: input={input_path}")

    matches_file = os.path.join(input_path, "matches.jsonl")
    resolutions: list[BlockResolution] = []
    with open(matches_file) as f:
        for line in f:
            resolutions.append(BlockResolution.model_validate_json(line.strip()))

    # Compute original entity count from block resolutions
    total_input = sum(r.original_count for r in resolutions)

    # Collect all UUIDs for validation
    historical_uuids: set[str] = set()
    for r in resolutions:
        for e in r.resolved_entities:
            if e.uuid:
                historical_uuids.add(e.uuid)
            for su in e.source_uuids or []:
                historical_uuids.add(su)

    # Run comprehensive evaluation
    metrics = evaluate_er_results(
        resolutions=resolutions,
        original_entity_count=total_input,
        iteration=1,
        historical_uuids=historical_uuids,
    )

    # Print formatted report
    report = format_evaluation_report(metrics)
    click.echo(report)

    # Save evaluation metrics
    eval_file = os.path.join(input_path, "evaluation.json")
    save_evaluation(metrics, eval_file)

    # Optional ground truth comparison
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
            for e in r.resolved_entities:
                if e.source_ids:
                    for sid in e.source_ids:
                        predicted_pairs.add((min(e.id, sid), max(e.id, sid)))

        gt_metrics = evaluate_resolution(predicted_pairs, true_pairs)
        click.echo("\n  Ground Truth Comparison:")
        results_df = pd.DataFrame(
            [
                {
                    "Metric": "Precision",
                    "Value": f"{gt_metrics['precision']:.4f}",
                },
                {
                    "Metric": "Recall",
                    "Value": f"{gt_metrics['recall']:.4f}",
                },
                {
                    "Metric": "F1 Score",
                    "Value": f"{gt_metrics['f1_score']:.4f}",
                },
                {
                    "Metric": "Predicted Pairs",
                    "Value": str(len(predicted_pairs)),
                },
                {
                    "Metric": "Correct (TP)",
                    "Value": str(gt_metrics["true_positives"]),
                },
                {
                    "Metric": "Wrong (FP)",
                    "Value": str(gt_metrics["false_positives"]),
                },
            ]
        )
        click.echo(results_df.to_string(index=False))


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
@click.option("--target-block-size", type=int, default=30, help="Target entities per block")
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
        max_block_size=100,
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
    type=click.Choice(BENCHMARK_DATASETS, case_sensitive=False),
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

    click.echo(f"Downloading {dataset}...")
    benchmark_data = BenchmarkDataset.download(dataset, output_path)
    click.echo(f"  Left table: {len(benchmark_data.table_a)} records")
    click.echo(f"  Right table: {len(benchmark_data.table_b)} records")
    click.echo(f"  Ground truth pairs: {len(benchmark_data.ground_truth)}")
    click.echo("Done.")


# ---------------------------------------------------------------------------
# profile-benchmark  (exploratory analysis of a benchmark dataset)
# ---------------------------------------------------------------------------


@cli.command(name="profile-benchmark", context_settings={"show_default": True})
@click.option(
    "--dataset",
    "-d",
    "datasets",
    type=click.Choice(BENCHMARK_DATASETS, case_sensitive=False),
    multiple=True,
    help="Benchmark to profile. Repeatable. Defaults to every benchmark",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=False,
    help="Write the Markdown report here instead of printing it",
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(),
    required=False,
    help="Also write the structured profile as JSON here",
)
@click.option(
    "--examples",
    type=int,
    default=6,
    help="How many match and mismatch examples to report per dataset",
)
@click.option(
    "--top-values",
    type=int,
    default=6,
    help="How many frequent values to report per column",
)
def profile_benchmark_command(
    datasets: tuple[str, ...],
    output_path: str | None,
    json_path: str | None,
    examples: int,
    top_values: int,
) -> None:
    """Run exploratory analysis over the benchmark datasets.

    Reports per-column completeness and Magellan discriminativeness, the most
    common values, gold pair cardinality, how often each attribute agrees on a
    true match against the hardest non-matches a token blocker produces, and
    worked examples of both a match and a non-match that string similarity
    gets wrong. Everything is computed in Spark SQL.
    """
    from serf.analyze.benchmarks import format_profile, profile_benchmark

    selected = list(datasets) if datasets else BENCHMARK_DATASETS
    reports: list[str] = []
    profiles: dict[str, Any] = {}
    for name in selected:
        click.echo(f"Profiling {name}...", err=True)
        profile = profile_benchmark(name, examples=examples, top_values=top_values)
        profiles[name] = profile
        reports.append(format_profile(profile))

    report = "\n".join(reports)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        click.echo(f"Wrote {output_path}", err=True)
    else:
        click.echo(report)

    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2, default=str)
        click.echo(f"Wrote {json_path}", err=True)


# ---------------------------------------------------------------------------
# blocking-sweep  (compare embedding models on blocking recall)
# ---------------------------------------------------------------------------


@cli.command(name="blocking-sweep")
@click.option(
    "--dataset",
    "-d",
    "datasets",
    type=click.Choice(BENCHMARK_DATASETS, case_sensitive=False),
    multiple=True,
    help="Benchmark to sweep. Repeatable. Defaults to every benchmark",
)
@click.option(
    "--model",
    "-m",
    "models",
    type=str,
    multiple=True,
    help="Embedding model to test. Repeatable. Defaults to the config candidates",
)
@click.option(
    "--prompt",
    type=str,
    required=False,
    help="Instruction prefix applied to every --model given on the command line",
)
@click.option(
    "--candidate-set",
    type=click.Choice(["small", "large", "all"], case_sensitive=False),
    default="small",
    help="Which configured candidate list to sweep, ignored when --model is given",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    help="Run the model repository's own modelling code, which some architectures require",
)
@click.option(
    "--blocking-strategy",
    type=click.Choice(["name", "json", "both"], case_sensitive=False),
    default="name",
    help="Embed the name alone, every field as JSON, or score both",
)
@click.option(
    "--sample",
    type=int,
    default=0,
    help="Record budget per dataset, or 0 for the whole dataset",
)
@click.option("--seed", type=int, default=42, help="Sampling seed")
@click.option("--target-block-size", type=int, required=False, help="Target entities per block")
@click.option("--max-block-size", type=int, required=False, help="Maximum entities per block")
@click.option(
    "--rounds",
    type=int,
    default=1,
    help="ER rounds to simulate, merging co-blocked gold pairs between them",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=False,
    help="Path to write the sweep results as JSON",
)
def blocking_sweep(
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    prompt: str | None,
    candidate_set: str,
    blocking_strategy: str,
    trust_remote_code: bool,
    sample: int,
    seed: int,
    target_block_size: int | None,
    max_block_size: int | None,
    rounds: int,
    output_path: str | None,
) -> None:
    """Compare embedding models on name-only blocking recall.

    Blocking recall is the share of gold pairs whose two records land in the same
    block, which caps the recall any matcher can reach. No LLM calls are made.

    With --rounds above one, every gold pair that shares a block is merged before
    the next round, so the cumulative recall is the ceiling a multi-round run
    could reach if the matcher never made a mistake.
    """
    from serf.eval.blocking_sweep import EmbeddingCandidate, sweep_dataset

    if models:
        candidates = [EmbeddingCandidate(name, prompt or "", trust_remote_code) for name in models]
    else:
        keys = {
            "small": ["benchmarks.embedding_candidates"],
            "large": ["benchmarks.embedding_candidates_large"],
            "all": ["benchmarks.embedding_candidates", "benchmarks.embedding_candidates_large"],
        }[candidate_set.lower()]
        configured: list[dict[str, object]] = []
        for key in keys:
            configured.extend(cast(list[dict[str, object]], serf_config.get(key, [])))
        candidates = [
            EmbeddingCandidate(
                model=str(entry["model"]),
                prompt=str(entry.get("prompt", "")),
                trust_remote_code=bool(entry.get("trust_remote_code", False)),
            )
            for entry in configured
        ]

    targets = list(datasets) if datasets else BENCHMARK_DATASETS
    strategies = (
        ["name", "json"] if blocking_strategy.lower() == "both" else [blocking_strategy.lower()]
    )

    results = []
    for name in targets:
        click.echo(f"\n{name}")
        for strategy in strategies:
            for result in sweep_dataset(
                dataset=name,
                candidates=candidates,
                sample=sample,
                seed=seed,
                target_block_size=target_block_size,
                max_block_size=max_block_size,
                rounds=rounds,
                strategy=strategy,
            ):
                results.append(result.as_dict())
                label = result.model.split("/")[-1] + (" +prefix" if result.prompt else "")
                round_note = f" round {result.round_number}" if rounds > 1 else ""
                cumulative = (
                    f"  cumulative {result.cumulative_recall:.4f}"
                    f" ({result.cumulative_co_blocked}/{result.gold_pairs})"
                    if rounds > 1
                    else ""
                )
                click.echo(
                    f"  [{strategy}] {label:>34}{round_note}"
                    f"  recall {result.blocking_recall:.4f}"
                    f"  ({result.co_blocked}/{result.gold_pairs}){cumulative}"
                    f"  {result.blocks} blocks  {result.elapsed_seconds:.0f}s"
                )

    if output_path:
        with open(output_path, "w") as handle:
            json.dump(results, handle, indent=2)
        click.echo(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# optimize  (GEPA student/teacher prompt optimization)
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--signature",
    type=click.Choice(["block-match", "entity-merge", "edge-resolve"], case_sensitive=False),
    default="block-match",
    help="ER signature to optimize with GEPA",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(BENCHMARK_DATASETS, case_sensitive=False),
    required=False,
    help="Benchmark dataset to block fully, then partition into train/val/holdout",
)
@click.option(
    "--trainset",
    type=click.Path(exists=True),
    required=False,
    help="JSONL file of labeled training examples",
)
@click.option(
    "--valset",
    type=click.Path(exists=True),
    required=False,
    help="JSONL file of labeled validation examples",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=False,
    help="Path to save the optimized DSPy program",
)
@click.option(
    "--student-model",
    type=str,
    default=None,
    help="Student/task LM (from config.yml models.student)",
)
@click.option(
    "--teacher-model",
    type=str,
    default=None,
    help="Teacher/reflection LM (from config.yml models.teacher)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for train/val/holdout sampling",
)
def optimize(
    signature: str,
    dataset: str | None,
    trainset: str | None,
    valset: str | None,
    output_path: str | None,
    student_model: str | None,
    teacher_model: str | None,
    seed: int | None,
) -> None:
    """Optimize an ER signature with GEPA.

    Uses the student/task LM for rollouts and the teacher LM as GEPA's
    reflection model. Requires VERTEX_AI_TOKEN for GPT OSS 120b and
    GEMINI_API_KEY for Gemini 3.5 Flash-Lite.

    With --dataset, randomly samples disjoint train, validation, and holdout
    records using the budgets in config.yml, keeping ground-truth match groups
    whole, then blocks within each split to build the BlockMatch examples.
    """
    import dspy

    from serf.dspy.optimize import (
        INPUT_FIELDS,
        SIGNATURES,
        load_jsonl_examples,
        optimize_module,
        prepare_dataset_splits,
    )
    from serf.eval.benchmarks import BenchmarkDataset
    from serf.eval.splits import get_all_split_sizes, get_split_sizes

    if not dataset and not trainset:
        raise click.UsageError("Provide --dataset or --trainset")

    setup_mlflow()

    student_model = student_model or serf_config.get("models.student")
    teacher_model = teacher_model or serf_config.get("models.teacher")
    seed = seed if seed is not None else int(serf_config.get("optimize.seed", 42))
    click.echo("GEPA optimization")
    click.echo(f"  Signature: {signature}")
    click.echo(f"  Student:   {student_model}")
    click.echo(f"  Teacher:   {teacher_model}")
    click.echo("  Split sizes (random match-group sampling; val and holdout first):")
    for name, sizes in get_all_split_sizes().items():
        click.echo(
            f"    {name}: {sizes.train_records} train records, "
            f"{sizes.val_records} val records, "
            f"{sizes.holdout_records} holdout records"
        )

    holdout_blocks: list[Any] = []
    if dataset:
        sizes = get_split_sizes(dataset)
        benchmark_data = BenchmarkDataset.download(dataset, output_path)
        left_entities, right_entities = benchmark_data.to_entities()
        entities = left_entities + right_entities
        click.echo(f"  Sampling splits from {len(entities)} records, then blocking each split...")
        train_examples, val_examples, holdout_blocks = prepare_dataset_splits(
            entities,
            benchmark_data.ground_truth,
            sizes=sizes,
            seed=seed,
        )
    else:
        input_fields = INPUT_FIELDS[signature]
        train_examples = load_jsonl_examples(str(trainset), input_fields)
        val_examples = load_jsonl_examples(valset, input_fields) if valset else None

    click.echo(f"  Trainset:  {len(train_examples)} block examples")
    if val_examples is not None:
        click.echo(f"  Valset:    {len(val_examples)} block examples")
    if holdout_blocks:
        holdout_size = sum(block.block_size for block in holdout_blocks)
        click.echo(f"  Holdout:   {len(holdout_blocks)} blocks ({holdout_size} records)")

    module = cast(dspy.Module, dspy.Predict(SIGNATURES[signature]))
    # GEPA resumes from any state it finds in log_dir, so each run keeps its own.
    optimized = optimize_module(
        module,
        trainset=train_examples,
        valset=val_examples,
        student_model=student_model,
        teacher_model=teacher_model,
        log_dir=output_path if dataset and output_path else None,
    )

    if output_path:
        os.makedirs(
            output_path if dataset else (os.path.dirname(output_path) or "."), exist_ok=True
        )
        program_path = os.path.join(output_path, f"{dataset}_gepa.json") if dataset else output_path
        optimized.save(program_path)
        click.echo(f"\nSaved optimized program to {program_path}")
        if holdout_blocks:
            holdout_path = os.path.join(output_path, f"{dataset}_holdout.jsonl")
            with open(holdout_path, "w", encoding="utf-8") as handle:
                for block in holdout_blocks:
                    handle.write(json.dumps(block.model_dump(mode="json")) + "\n")
            click.echo(f"Saved holdout blocks to {holdout_path}")


# ---------------------------------------------------------------------------
# benchmark  (single dataset, LLM matching)
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(BENCHMARK_DATASETS, case_sensitive=False),
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
    default=30,
    help="Target entities per block",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model for matching (from config.yml models.llm)",
)
@click.option(
    "--max-right-entities",
    type=int,
    default=None,
    help="Limit right table size for large datasets",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Max blocks to send through LLM matching (for testing)",
)
@click.option(
    "--concurrency",
    type=int,
    default=20,
    help="Number of concurrent LLM requests",
)
@click.option(
    "--max-iterations",
    type=int,
    default=serf_config.get("er.max_iterations", 3),
    help="Maximum ER iterations (re-block and re-match merged entities)",
)
@click.option(
    "--signature-mode",
    type=click.Choice(list(SIGNATURE_MODES), case_sensitive=False),
    default=SIGNATURE_MODE_GENERIC,
    help="Matching contract: the shared BlockMatch signature or this dataset's typed signature",
)
@click.option(
    "--sample-records",
    type=int,
    default=None,
    help="Sample this many records, keeping ground-truth match groups whole",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for record sampling (from config.yml optimize.seed)",
)
@click.option(
    "--embedding-tier",
    type=click.Choice(["low", "high"], case_sensitive=False),
    default="low",
    help="Blocking embedding: the small default or the large alternative",
)
@click.option(
    "--blocking-strategy",
    type=click.Choice(["name", "json"], case_sensitive=False),
    default=None,
    help="Embed the name alone or every field as JSON with field names inline",
)
def benchmark(
    dataset: str,
    output_path: str | None,
    target_block_size: int,
    model: str | None,
    max_right_entities: int | None,
    limit: int | None,
    concurrency: int,
    max_iterations: int,
    signature_mode: str,
    sample_records: int | None,
    seed: int | None,
    embedding_tier: str,
    blocking_strategy: str | None,
) -> None:
    """Run ER pipeline against a benchmark dataset and evaluate.

    Uses embeddings for blocking and LLM for matching.
    Requires VERTEX_AI_TOKEN for GPT OSS 120b (student) and GEMINI_API_KEY
    for Gemini 3.5 Flash-Lite (teacher/analyze).

    With --signature-mode per-dataset the block is matched with the typed DSPy
    signature written for this dataset instead of the shared BlockMatch one.
    With --sample-records the dataset is sampled by ground-truth match group, so
    gold pairs survive, and metrics are scored against the surviving pairs.
    """
    from serf.eval.benchmarks import BenchmarkDataset

    setup_mlflow()

    model = model or serf_config.get("models.llm")
    tier = embedding_tier.lower()
    embedding_model = str(serf_config.get(f"models.embedding_{tier}"))
    embedding_prompt = str(serf_config.get(f"models.embedding_{tier}_prompt", ""))
    strategy = (blocking_strategy or str(serf_config.get("er.blocking.strategy", "name"))).lower()

    click.echo(f"Running benchmark: {dataset}")
    click.echo(f"  Model: {model}")
    click.echo(f"  Signature mode: {signature_mode}")
    click.echo(f"  Embedding: {embedding_model} ({tier} tier, {strategy} blocking)")
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

    if sample_records:
        from serf.eval.sample import sample_records as sample_benchmark_records

        seed = seed if seed is not None else int(serf_config.get("optimize.seed", 42))
        record_sample = sample_benchmark_records(
            all_entities,
            benchmark_data.ground_truth,
            sample_records,
            seed=seed,
        )
        all_entities = record_sample.records
        benchmark_data.ground_truth = record_sample.ground_truth
        click.echo(
            f"  Sampled {len(all_entities)} of {record_sample.total} records "
            f"(seed {seed}) with {record_sample.gold_pairs} gold pairs retained"
        )

    click.echo(f"  Total entities: {len(all_entities)}")

    # Auto-scale block size for limited test runs
    effective_block_size = target_block_size
    if limit and limit <= 20 and target_block_size >= 20:
        effective_block_size = 5
        click.echo(f"  Auto-scaled target_block_size to {effective_block_size} for --limit={limit}")

    all_predicted_pairs: set[tuple[int, int]] = set()
    current_entities = all_entities
    iterations_run = 0

    for iteration in range(1, max_iterations + 1):
        if max_iterations > 1:
            click.echo(f"\n  === Iteration {iteration}/{max_iterations} ===")
        prev_count = len(current_entities)

        pairs, _resolved = _benchmark_llm_matching(
            current_entities,
            effective_block_size,
            model,
            limit,
            concurrency,
            dataset=dataset,
            signature_mode=signature_mode,
            iteration=iteration,
            embedding_model=embedding_model,
            embedding_prompt=embedding_prompt,
            blocking_strategy=strategy,
        )
        all_predicted_pairs.update(expand_pairs(pairs, entity_members(current_entities)))
        iterations_run = iteration

        merged = merge_matched_entities(current_entities, pairs)
        if max_iterations > 1:
            reduction_pct = (prev_count - len(merged)) / prev_count * 100 if prev_count > 0 else 0
            click.echo(
                f"    Entities: {prev_count} -> {len(merged)} ({reduction_pct:.1f}% reduction)"
            )

        if len(merged) >= prev_count or iteration == max_iterations:
            if len(merged) >= prev_count and max_iterations > 1 and iteration < max_iterations:
                click.echo("    Converged (no merges), stopping early")
            break

        current_entities = merged

    predicted_pairs = all_predicted_pairs
    metrics = benchmark_data.evaluate(predicted_pairs)
    elapsed = time.time() - start

    click.echo(f"\n  Benchmark Results ({elapsed:.1f}s):")
    results_df = pd.DataFrame(
        [
            {"Metric": "Precision", "Value": f"{metrics['precision']:.4f}"},
            {"Metric": "Recall", "Value": f"{metrics['recall']:.4f}"},
            {"Metric": "F1 Score", "Value": f"{metrics['f1_score']:.4f}"},
            {"Metric": "Predicted Pairs", "Value": str(len(predicted_pairs))},
            {"Metric": "Correct (TP)", "Value": str(metrics["true_positives"])},
            {"Metric": "Wrong (FP)", "Value": str(metrics["false_positives"])},
            {"Metric": "Iterations", "Value": str(iterations_run)},
        ]
    )
    click.echo(results_df.to_string(index=False))

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        suffix = "" if signature_mode == SIGNATURE_MODE_GENERIC else f"_{signature_mode}"
        results_file = os.path.join(output_path, f"{dataset}{suffix}_results.json")
        with open(results_file, "w") as f:
            json.dump(
                {
                    "dataset": dataset,
                    "model": model,
                    "signature_mode": signature_mode,
                    "sample_records": sample_records,
                    "seed": seed,
                    "max_iterations": max_iterations,
                    "iterations_run": iterations_run,
                    "gold_pairs_retained": len(benchmark_data.ground_truth),
                    "records": len(all_entities),
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
    default=None,
    help="LLM model for matching (from config.yml models.llm)",
)
@click.option(
    "--max-right-entities",
    type=int,
    default=5000,
    help="Limit right table size for large datasets",
)
def benchmark_all(
    output_path: str,
    model: str | None,
    max_right_entities: int,
) -> None:
    """Run LLM-based benchmarks on all available datasets.

    Requires VERTEX_AI_TOKEN for GPT OSS 120b (student) and GEMINI_API_KEY
    for Gemini 3.5 Flash-Lite (teacher/analyze).
    """
    from serf.eval.benchmarks import BenchmarkDataset

    setup_mlflow()

    model = model or serf_config.get("models.llm")
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
            target_block_size=30,
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


def _dataframe_to_entities(df: Any) -> list[Any]:
    """Convert a pandas DataFrame to Entity objects. Delegates to pipeline module."""
    from serf.pipeline import (
        _detect_name_field,
        _detect_text_fields,
        dataframe_to_entities,
    )

    name_field = _detect_name_field(df)
    text_fields = _detect_text_fields(df, name_field)
    return dataframe_to_entities(df, name_field, text_fields)


def _benchmark_llm_matching(
    all_entities: list[Any],
    target_block_size: int,
    model: str | None = None,
    limit: int | None = None,
    concurrency: int = 20,
    dataset: str | None = None,
    signature_mode: str = SIGNATURE_MODE_GENERIC,
    iteration: int = 1,
    embedding_model: str | None = None,
    embedding_prompt: str | None = None,
    blocking_strategy: str | None = None,
) -> tuple[set[tuple[int, int]], list[Any]]:
    """Run LLM-based matching for benchmarks.

    Embeddings are used for blocking only. Matching is done by LLM
    with concurrent async requests.

    Parameters
    ----------
    all_entities : list[Entity]
        All entities to match
    target_block_size : int
        Target block size for FAISS blocking
    model : str
        LLM model name
    limit : int | None
        Max blocks to process (for testing)
    concurrency : int
        Number of concurrent LLM requests
    dataset : str | None
        Benchmark dataset name, required for per-dataset signatures
    signature_mode : str
        ``generic`` for the shared BlockMatch signature, ``per-dataset`` for the
        typed signature written for this dataset
    iteration : int
        Current ER iteration, which tightens the target block size when
        ``er.blocking.auto_scale_by_iteration`` is on
    embedding_model : str | None
        Blocking embedding to use. Defaults to config.
    embedding_prompt : str | None
        Instruction prefix for that model. Defaults to config.
    blocking_strategy : str | None
        ``name`` or ``json``. Defaults to config.

    Returns
    -------
    tuple[set[tuple[int, int]], list[Entity]]
        Predicted match pairs and resolved entities for next iteration
    """
    from serf.block.pipeline import SemanticBlockingPipeline
    from serf.match.run import match_blocks

    max_block = min(
        serf_config.get("er.blocking.max_block_size", 100),
        target_block_size * 3,
    )
    auto_scale = bool(serf_config.get("er.blocking.auto_scale_by_iteration", True))
    pipeline = SemanticBlockingPipeline(
        model_name=embedding_model,
        target_block_size=target_block_size,
        max_block_size=max_block,
        iteration=iteration,
        auto_scale=auto_scale,
        embedding_prompt=embedding_prompt,
        blocking_strategy=blocking_strategy,
    )
    click.echo(
        f"\n  Blocking (target={target_block_size}, max={max_block}, "
        f"iteration={iteration}, auto_scale={auto_scale})..."
    )
    blocks, blocking_metrics = pipeline.run(all_entities)
    click.echo(
        f"    {blocking_metrics.total_blocks} blocks created "
        f"(avg {blocking_metrics.avg_block_size:.1f}, max {blocking_metrics.max_block_size})"
    )

    click.echo(f"  Matching with LLM ({model}, concurrency={concurrency}, limit={limit})...")
    outcome = match_blocks(
        blocks,
        signature_mode=signature_mode,
        dataset=dataset,
        model=model,
        concurrency=concurrency,
        limit=limit,
    )
    if outcome.single_source_blocks:
        click.echo(f"    {outcome.single_source_blocks} single-source blocks skipped")
    if outcome.dropped_candidates:
        click.echo(f"    {outcome.dropped_candidates} candidates dropped for unknown record ids")
    if outcome.failed_blocks:
        click.echo(
            f"    WARNING: {outcome.failed_blocks} blocks failed their LLM call, "
            f"recall is understated"
        )

    click.echo(f"    Predicted {len(outcome.predicted_pairs)} match pairs")
    return outcome.predicted_pairs, outcome.resolved_entities


if __name__ == "__main__":
    cli()
