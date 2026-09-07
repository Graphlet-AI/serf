"""Tests for the benchmark CLI command iteration logic."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner, Result

from serf.cli.main import _benchmark_llm_matching, cli
from serf.dspy.types import BlockResolution, Entity, MatchDecision


def _make_entities(n: int, id_offset: int = 0) -> list[Entity]:
    """Create n dummy entities starting at the given id offset."""
    return [
        Entity(id=id_offset + i, name=f"entity_{id_offset + i}", description=f"desc {i}")
        for i in range(n)
    ]


def _make_benchmark_data(
    left_n: int = 4, right_n: int = 4, gt_pairs: set[tuple[int, int]] | None = None
) -> MagicMock:
    """Create a mock BenchmarkDataset."""
    mock = MagicMock()
    left = _make_entities(left_n)
    right = _make_entities(right_n, id_offset=10000)
    mock.to_entities.return_value = (left, right)
    mock.ground_truth = gt_pairs or {(0, 10000)}
    mock.evaluate.return_value = {
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0,
        "true_positives": 1,
        "false_positives": 0,
    }
    return mock


# Shared patches for benchmark CLI tests
_BENCHMARK_PATCHES = [
    "serf.cli.main.setup_mlflow",
    "serf.cli.main._benchmark_llm_matching",
    "serf.eval.benchmarks.BenchmarkDataset",
    "serf.config.config",
]


def _run_benchmark_cli(
    mock_matching: MagicMock,
    mock_bd_cls: MagicMock,
    mock_config: MagicMock,
    args: list[str],
) -> Result:
    """Set up mocks and invoke the benchmark CLI command.

    Parameters
    ----------
    mock_matching : MagicMock
        Mock for _benchmark_llm_matching
    mock_bd_cls : MagicMock
        Mock for BenchmarkDataset class
    mock_config : MagicMock
        Mock for serf.config.config
    args : list[str]
        CLI arguments to pass to "benchmark" subcommand

    Returns
    -------
    Result
        CLI invocation result
    """
    mock_bd_cls.available_datasets.return_value = ["dblp-acm"]
    mock_bd_cls.download.return_value = _make_benchmark_data()
    mock_config.get.return_value = "test-model"

    runner = CliRunner()
    return runner.invoke(cli, ["benchmark"] + args, catch_exceptions=False)


# ── 1. --max-iterations argument is accepted and forwarded ──────────────


def test_benchmark_accepts_max_iterations_arg() -> None:
    """The benchmark command accepts --max-iterations and passes it through."""
    with (
        patch(_BENCHMARK_PATCHES[0]),
        patch(_BENCHMARK_PATCHES[1]) as mock_matching,
        patch(_BENCHMARK_PATCHES[2]) as mock_bd_cls,
        patch(_BENCHMARK_PATCHES[3]) as mock_config,
    ):
        entities = _make_entities(8)
        mock_matching.return_value = ({(0, 1)}, entities)

        result = _run_benchmark_cli(
            mock_matching,
            mock_bd_cls,
            mock_config,
            ["--dataset", "dblp-acm", "--max-iterations", "3"],
        )

    assert result.exit_code == 0
    assert "Benchmark Results" in result.output


# ── 2. Multiple iterations of entity resolution ────────────────────────


def test_benchmark_runs_multiple_iterations() -> None:
    """With max_iterations>1 and entities reducing, multiple iterations run."""
    with (
        patch(_BENCHMARK_PATCHES[0]),
        patch(_BENCHMARK_PATCHES[1]) as mock_matching,
        patch(_BENCHMARK_PATCHES[2]) as mock_bd_cls,
        patch(_BENCHMARK_PATCHES[3]) as mock_config,
    ):
        mock_matching.side_effect = [
            ({(0, 1)}, _make_entities(6)),
            ({(0, 1), (2, 3)}, _make_entities(4)),
            ({(0, 1), (2, 3)}, _make_entities(4)),
        ]

        result = _run_benchmark_cli(
            mock_matching,
            mock_bd_cls,
            mock_config,
            ["--dataset", "dblp-acm", "--max-iterations", "5"],
        )

    assert result.exit_code == 0
    assert "Iteration 1/" in result.output
    assert "Iteration 2/" in result.output
    assert mock_matching.call_count == 3


# ── 3. Early stop on convergence (no entity reduction) ──────────────────


def test_benchmark_stops_early_on_convergence() -> None:
    """When entity count does not decrease, the loop stops early."""
    with (
        patch(_BENCHMARK_PATCHES[0]),
        patch(_BENCHMARK_PATCHES[1]) as mock_matching,
        patch(_BENCHMARK_PATCHES[2]) as mock_bd_cls,
        patch(_BENCHMARK_PATCHES[3]) as mock_config,
    ):
        all_entities = _make_entities(8)
        mock_matching.return_value = ({(0, 1)}, all_entities)

        result = _run_benchmark_cli(
            mock_matching,
            mock_bd_cls,
            mock_config,
            ["--dataset", "dblp-acm", "--max-iterations", "5"],
        )

    assert result.exit_code == 0
    assert mock_matching.call_count == 1
    assert "Converged" in result.output


# ── 4. _benchmark_llm_matching returns (pairs, resolved_entities) ───────


def test_benchmark_llm_matching_return_type() -> None:
    """_benchmark_llm_matching returns a tuple of (set of pairs, list of entities)."""
    entities = _make_entities(4)
    resolved = _make_entities(3)

    resolution = BlockResolution(
        block_key="b0",
        matches=[
            MatchDecision(
                entity_a_id=0, entity_b_id=1, is_match=True, confidence=0.9, reasoning="same"
            )
        ],
        resolved_entities=resolved,
        was_resolved=True,
        original_count=4,
        resolved_count=3,
    )

    with (
        patch("asyncio.run", return_value=[resolution]),
        patch("serf.block.pipeline.SemanticBlockingPipeline") as mock_pipeline_cls,
        patch("serf.match.matcher.EntityMatcher"),
    ):
        blocking_metrics = MagicMock()
        blocking_metrics.total_blocks = 1
        mock_pipeline_cls.return_value.run.return_value = ([], blocking_metrics)

        pairs, result_entities = _benchmark_llm_matching(entities, target_block_size=10, model="m")

    assert isinstance(pairs, set)
    assert isinstance(result_entities, list)
    assert (0, 1) in pairs
    assert len(result_entities) == 3
    for e in result_entities:
        assert isinstance(e, Entity)


# ── 5. Iterations count is reported in output ───────────────────────────


def test_benchmark_reports_iteration_count() -> None:
    """The benchmark output includes the number of iterations run."""
    with (
        patch(_BENCHMARK_PATCHES[0]),
        patch(_BENCHMARK_PATCHES[1]) as mock_matching,
        patch(_BENCHMARK_PATCHES[2]) as mock_bd_cls,
        patch(_BENCHMARK_PATCHES[3]) as mock_config,
    ):
        mock_matching.side_effect = [
            ({(0, 1)}, _make_entities(6)),
            ({(0, 1)}, _make_entities(6)),
        ]

        result = _run_benchmark_cli(
            mock_matching,
            mock_bd_cls,
            mock_config,
            ["--dataset", "dblp-acm", "--max-iterations", "5"],
        )

    assert result.exit_code == 0
    assert "Iterations" in result.output
    assert mock_matching.call_count == 2
    for line in result.output.splitlines():
        if "Iterations" in line:
            assert "2" in line
            break
    else:
        raise AssertionError("Iterations row not found in output")
