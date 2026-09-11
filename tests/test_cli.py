"""Tests for the SERF CLI."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from serf.cli.main import BENCHMARK_DATASETS, cli
from serf.config import config


def test_cli_help() -> None:
    """Test that the CLI help output includes all commands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "SERF" in result.output
    assert "run" in result.output
    assert "analyze" in result.output
    assert "block" in result.output
    assert "match" in result.output
    assert "eval" in result.output
    assert "edges" in result.output
    assert "resolve" in result.output
    assert "benchmark" in result.output
    assert "benchmark-all" in result.output
    assert "download" in result.output
    assert "optimize" in result.output
    assert "profile-benchmark" in result.output
    assert "blocking-sweep" in result.output
    assert "mteb-rank" in result.output


def test_cli_version() -> None:
    """Test that version flag works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_analyze_help() -> None:
    """Test analyze command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--output" in result.output
    assert "--model" in result.output


def test_block_help() -> None:
    """Test block command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["block", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--method" in result.output
    assert "--target-block-size" in result.output


def test_match_help() -> None:
    """Test match command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["match", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--batch-size" in result.output


def test_eval_help() -> None:
    """Test eval command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--ground-truth" in result.output


def test_benchmark_help() -> None:
    """Test benchmark command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output
    assert "--model" in result.output
    assert "--max-right-entities" in result.output
    assert "walmart-amazon" in result.output
    assert "amazon-google" in result.output
    assert "--embedding-tier" in result.output
    assert "--blocking-strategy" in result.output


def test_blocking_sweep_help() -> None:
    """The sweep offers each single strategy and the union of both."""
    runner = CliRunner()
    result = runner.invoke(cli, ["blocking-sweep", "--help"])
    assert result.exit_code == 0
    assert "--blocking-strategy" in result.output
    assert "union" in result.output
    assert "--candidate-set" in result.output
    assert "--rounds" in result.output


def test_mteb_rank_help() -> None:
    """The MTEB ranking exposes category selection and sweep correlation."""
    runner = CliRunner()
    result = runner.invoke(cli, ["mteb-rank", "--help"])
    assert result.exit_code == 0
    assert "--category" in result.output
    assert "--candidate-set" in result.output
    assert "--sweep" in result.output


def test_profile_benchmark_help() -> None:
    """The profiler exposes dataset selection and both output formats."""
    runner = CliRunner()
    result = runner.invoke(cli, ["profile-benchmark", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output
    assert "--output" in result.output
    assert "--json" in result.output
    assert "--examples" in result.output
    assert "--top-values" in result.output
    assert "dblp-scholar" in result.output


def test_embedding_tiers_are_configured() -> None:
    """Both blocking tiers exist and the pipeline default is the low one."""
    low = config.get("models.embedding_low")
    high = config.get("models.embedding_high")
    assert low and high and low != high
    assert config.get("models.embedding") == low
    assert config.get("models.embedding_prompt") == config.get("models.embedding_low_prompt")


def test_benchmark_all_help() -> None:
    """Test benchmark-all command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark-all", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--max-right-entities" in result.output


def test_download_help() -> None:
    """Test download command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["download", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output
    assert "walmart-amazon" in result.output
    assert "amazon-google" in result.output


def test_resolve_help() -> None:
    """Test resolve command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["resolve", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--iteration" in result.output


def test_download_unknown_dataset() -> None:
    """Test download with unknown dataset name."""
    runner = CliRunner()
    result = runner.invoke(cli, ["download", "--dataset", "nonexistent"])
    assert result.exit_code == 2
    assert "Invalid value for '--dataset'" in result.output


def test_benchmark_unknown_dataset() -> None:
    """Test benchmark with unknown dataset name."""
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "--dataset", "nonexistent"])
    assert result.exit_code == 2
    assert "Invalid value for '--dataset'" in result.output


def test_optimize_help() -> None:
    """Test optimize command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["optimize", "--help"])
    assert result.exit_code == 0
    assert "--signature" in result.output
    assert "--trainset" in result.output
    assert "--dataset" in result.output
    assert "--student-model" in result.output
    assert "--teacher-model" in result.output
    assert "walmart-amazon" in result.output
    assert "amazon-google" in result.output


def test_optimize_requires_dataset_or_trainset() -> None:
    """Optimize without --dataset or --trainset fails."""
    runner = CliRunner()
    result = runner.invoke(cli, ["optimize"])
    assert result.exit_code != 0


def test_benchmark_datasets_include_deepmatcher() -> None:
    """CLI dataset choices include Walmart-Amazon and Amazon-Google."""
    assert "walmart-amazon" in BENCHMARK_DATASETS
    assert "amazon-google" in BENCHMARK_DATASETS
    assert "abt-buy" in BENCHMARK_DATASETS


def test_download_accepts_new_datasets() -> None:
    """download --dataset accepts the DeepMatcher product datasets."""
    runner = CliRunner()
    mock_ds = MagicMock()
    mock_ds.table_a = [0, 1]
    mock_ds.table_b = [0]
    mock_ds.ground_truth = set()
    with patch("serf.eval.benchmarks.BenchmarkDataset.download", return_value=mock_ds):
        for name in ("walmart-amazon", "amazon-google"):
            result = runner.invoke(cli, ["download", "--dataset", name])
            assert result.exit_code == 0, result.output
            assert "Left table" in result.output


def test_optimize_unknown_dataset() -> None:
    """optimize rejects dataset names that are not registered."""
    runner = CliRunner()
    result = runner.invoke(cli, ["optimize", "--dataset", "nonexistent"])
    assert result.exit_code == 2
    assert "Invalid value for '--dataset'" in result.output


def test_run_help() -> None:
    """Test run command help shows all options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--output" in result.output
    assert "--config" in result.output
    assert "--name-field" in result.output
    assert "--text-fields" in result.output
    assert "--model" in result.output
    assert "--max-iterations" in result.output
    assert "--convergence-threshold" in result.output
    assert "--target-block-size" in result.output
