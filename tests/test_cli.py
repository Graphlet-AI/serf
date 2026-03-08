"""Tests for the SERF CLI."""

from click.testing import CliRunner

from serf.cli.main import cli


def test_cli_help() -> None:
    """Test that the CLI help output includes all commands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "SERF" in result.output
    assert "analyze" in result.output
    assert "block" in result.output
    assert "match" in result.output
    assert "eval" in result.output
    assert "edges" in result.output
    assert "resolve" in result.output
    assert "benchmark" in result.output
    assert "benchmark-all" in result.output
    assert "download" in result.output


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
    assert "--similarity-threshold" in result.output
    assert "--use-llm" in result.output
    assert "--max-right-entities" in result.output


def test_benchmark_all_help() -> None:
    """Test benchmark-all command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark-all", "--help"])
    assert result.exit_code == 0
    assert "--similarity-threshold" in result.output
    assert "--max-right-entities" in result.output


def test_download_help() -> None:
    """Test download command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["download", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output


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
    assert result.exit_code == 0
    assert "Unknown dataset" in result.output


def test_benchmark_unknown_dataset() -> None:
    """Test benchmark with unknown dataset name."""
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "--dataset", "nonexistent"])
    assert result.exit_code == 0
    assert "Unknown dataset" in result.output
