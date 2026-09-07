"""Tests for benchmark dataset loading and evaluation."""

import os
import tempfile

import pandas as pd

from serf.dspy.types import Entity
from serf.eval.benchmarks import (
    RIGHT_ID_OFFSET,
    BenchmarkDataset,
    _detect_name_column,
    _get_text_columns,
)


def test_available_datasets_returns_expected_names() -> None:
    """Test that available datasets includes expected benchmark names."""
    names = BenchmarkDataset.available_datasets()
    assert "dblp-acm" in names
    assert "dblp-scholar" in names
    assert "abt-buy" in names


def test_benchmark_dataset_creation_with_mock_data() -> None:
    """Test creating a BenchmarkDataset with mock DataFrames."""
    table_a = pd.DataFrame({"id": [1, 2, 3], "title": ["Paper A", "Paper B", "Paper C"]})
    table_b = pd.DataFrame({"id": [1, 2, 3], "title": ["Paper A'", "Paper B'", "Paper D"]})
    ground_truth = {(1, 100001), (2, 100002)}

    ds = BenchmarkDataset(
        name="test",
        table_a=table_a,
        table_b=table_b,
        ground_truth=ground_truth,
        metadata={"domain": "test"},
    )
    assert ds.name == "test"
    assert len(ds.ground_truth) == 2


def test_evaluate_with_known_predictions() -> None:
    """Test evaluation against known ground truth."""
    table_a = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
    table_b = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
    ground_truth = {(1, 100001), (2, 100002)}

    ds = BenchmarkDataset("test", table_a, table_b, ground_truth, {})

    # Perfect predictions
    metrics = ds.evaluate({(1, 100001), (2, 100002)})
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0

    # Partial predictions
    metrics = ds.evaluate({(1, 100001)})
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5


def test_to_entities_produces_valid_entities() -> None:
    """Test that to_entities creates proper Entity objects."""
    table_a = pd.DataFrame(
        {"id": ["a1", "a2"], "title": ["Paper One", "Paper Two"], "authors": ["Auth A", "Auth B"]}
    )
    table_b = pd.DataFrame(
        {
            "id": ["b1", "b2"],
            "title": ["Paper One'", "Paper Three"],
            "authors": ["Auth A'", "Auth C"],
        }
    )

    ds = BenchmarkDataset("test", table_a, table_b, set(), {})
    left, right = ds.to_entities()

    assert len(left) == 2
    assert len(right) == 2
    assert all(isinstance(e, Entity) for e in left)
    assert all(isinstance(e, Entity) for e in right)

    # Left entities use row index as ID
    assert left[0].id == 0
    assert left[1].id == 1
    # Right entities are offset
    assert right[0].id == RIGHT_ID_OFFSET
    assert right[1].id == RIGHT_ID_OFFSET + 1

    # Names should be from title column
    assert left[0].name == "Paper One"
    assert right[0].name == "Paper One'"


def test_detect_name_column() -> None:
    """Test name column detection."""
    df = pd.DataFrame({"id": [1], "title": ["Test"], "year": [2024]})
    assert _detect_name_column(df) == "title"

    df2 = pd.DataFrame({"id": [1], "name": ["Test"], "category": ["A"]})
    assert _detect_name_column(df2) == "name"

    df3 = pd.DataFrame({"id": [1], "year": [2024]})
    assert _detect_name_column(df3) is None


def test_get_text_columns() -> None:
    """Test text column detection."""
    df = pd.DataFrame(
        {"id": [1, 2], "title": ["Test", "Other"], "year": [2024, 2025], "authors": ["Auth", "B"]}
    )
    text_cols = _get_text_columns(df)
    assert "title" in text_cols
    assert "authors" in text_cols


def test_load_from_deepmatcher_format() -> None:
    """Test loading from DeepMatcher format directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create DeepMatcher format files
        pd.DataFrame({"id": [1, 2], "title": ["A", "B"]}).to_csv(
            os.path.join(tmpdir, "tableA.csv"), index=False
        )
        pd.DataFrame({"id": [1, 2], "title": ["A'", "C"]}).to_csv(
            os.path.join(tmpdir, "tableB.csv"), index=False
        )
        pd.DataFrame({"ltable_id": [1, 2], "rtable_id": [1, 2], "label": [1, 0]}).to_csv(
            os.path.join(tmpdir, "train.csv"), index=False
        )
        pd.DataFrame({"ltable_id": [1], "rtable_id": [1], "label": [1]}).to_csv(
            os.path.join(tmpdir, "valid.csv"), index=False
        )
        pd.DataFrame({"ltable_id": [], "rtable_id": [], "label": []}).to_csv(  # type: ignore[arg-type]
            os.path.join(tmpdir, "test.csv"), index=False
        )

        ds = BenchmarkDataset.load("test", tmpdir)
        assert len(ds.table_a) == 2
        assert len(ds.table_b) == 2
        assert len(ds.ground_truth) == 1  # Only label=1 pairs


def test_load_raises_when_directory_missing() -> None:
    """Test that load raises FileNotFoundError for missing directory."""
    import pytest

    with pytest.raises(FileNotFoundError):
        BenchmarkDataset.load("test", "/nonexistent/path")


def test_download_raises_for_unknown_dataset() -> None:
    """Test that download raises ValueError for unknown dataset name."""
    import pytest

    with pytest.raises(ValueError):
        BenchmarkDataset.download("nonexistent-dataset")
