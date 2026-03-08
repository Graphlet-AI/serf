"""Tests for benchmark dataset module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from serf.eval.benchmarks import BenchmarkDataset


def test_available_datasets_returns_expected_names() -> None:
    """available_datasets returns list of registry keys."""
    names = BenchmarkDataset.available_datasets()
    expected = {"walmart-amazon", "abt-buy", "amazon-google", "dblp-acm", "dblp-scholar"}
    assert set(names) == expected
    assert len(names) == 5


def test_benchmark_dataset_creation_with_mock_data() -> None:
    """BenchmarkDataset can be created with mock DataFrames."""
    table_a = pd.DataFrame({"id": [1, 2], "title": ["A", "B"], "price": [10, 20]})
    table_b = pd.DataFrame({"id": [1, 2], "title": ["A", "C"], "price": [10, 30]})
    ground_truth = {(1, 1)}
    metadata = {"domain": "products", "difficulty": "easy"}

    ds = BenchmarkDataset(
        name="test-ds",
        table_a=table_a,
        table_b=table_b,
        ground_truth=ground_truth,
        metadata=metadata,
    )

    assert ds.name == "test-ds"
    assert len(ds.table_a) == 2
    assert len(ds.table_b) == 2
    assert ds.ground_truth == {(1, 1)}
    assert ds.metadata["domain"] == "products"


def test_evaluate_with_known_predictions() -> None:
    """evaluate returns correct precision, recall, f1 for known predictions."""
    table_a = pd.DataFrame({"id": [1, 2]})
    table_b = pd.DataFrame({"id": [1, 2]})
    ground_truth = {(1, 1), (2, 2)}
    ds = BenchmarkDataset(
        name="test",
        table_a=table_a,
        table_b=table_b,
        ground_truth=ground_truth,
        metadata={},
    )

    # Perfect predictions
    pred_perfect = {(1, 1), (2, 2)}
    result = ds.evaluate(pred_perfect)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0

    # Partial overlap
    pred_partial = {(1, 1), (3, 3)}
    result = ds.evaluate(pred_partial)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_to_entities_produces_valid_entity_objects() -> None:
    """to_entities returns valid Entity objects with correct structure."""
    table_a = pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Product A", "Product B"],
            "price": [10.0, 20.0],
        }
    )
    table_b = pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Product A", "Product C"],
            "price": [10.0, 30.0],
        }
    )
    ds = BenchmarkDataset(
        name="test",
        table_a=table_a,
        table_b=table_b,
        ground_truth=set(),
        metadata={},
    )

    left_entities, right_entities = ds.to_entities()

    assert len(left_entities) == 2
    assert len(right_entities) == 2

    # Left entities: ids as-is, l_ prefix on attributes
    assert left_entities[0].id == 1
    assert left_entities[0].name == "Product A"
    assert "l_title" in left_entities[0].attributes
    assert left_entities[0].attributes["l_title"] == "Product A"

    # Right entities: ids offset by 100000, r_ prefix on attributes
    assert right_entities[0].id == 100001
    assert right_entities[0].name == "Product A"
    assert "r_title" in right_entities[0].attributes
    assert right_entities[0].attributes["r_title"] == "Product A"


def test_load_from_fixture_data() -> None:
    """load reads dataset from directory with tableA, tableB, train/valid/test."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pd.DataFrame({"id": [1, 2], "title": ["A", "B"]}).to_csv(root / "tableA.csv", index=False)
        pd.DataFrame({"id": [1, 2], "title": ["A", "C"]}).to_csv(root / "tableB.csv", index=False)
        pd.DataFrame(
            {
                "ltable_id": [1],
                "rtable_id": [1],
                "label": [1],
            }
        ).to_csv(root / "train.csv", index=False)
        pd.DataFrame(
            {
                "ltable_id": [],
                "rtable_id": [],
                "label": [],
            }
        ).to_csv(root / "valid.csv", index=False)
        pd.DataFrame(
            {
                "ltable_id": [2],
                "rtable_id": [2],
                "label": [1],
            }
        ).to_csv(root / "test.csv", index=False)

        ds = BenchmarkDataset.load("walmart-amazon", tmp)

        assert ds.name == "walmart-amazon"
        assert len(ds.table_a) == 2
        assert len(ds.table_b) == 2
        assert ds.ground_truth == {(1, 1), (2, 2)}


def test_load_raises_when_directory_missing() -> None:
    """load raises FileNotFoundError when data_dir does not exist."""
    with pytest.raises(FileNotFoundError, match="Data directory not found"):
        BenchmarkDataset.load("walmart-amazon", "/nonexistent/path")


def test_download_raises_for_unknown_dataset() -> None:
    """download raises ValueError for unknown dataset name."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        BenchmarkDataset.download("unknown-dataset")
