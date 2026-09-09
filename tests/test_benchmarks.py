"""Tests for benchmark dataset loading and evaluation."""

import io
import os
import tempfile
import urllib.request
import zipfile
from unittest.mock import patch

import pandas as pd

from serf.dspy.types import Entity
from serf.eval.benchmarks import (
    DATASET_REGISTRY,
    RIGHT_ID_OFFSET,
    BenchmarkDataset,
    _detect_name_column,
    _find_zip_member,
    _get_text_columns,
)


def test_available_datasets_returns_expected_names() -> None:
    """Test that available datasets includes expected benchmark names."""
    names = BenchmarkDataset.available_datasets()
    assert set(names) == {
        "dblp-acm",
        "dblp-scholar",
        "abt-buy",
        "walmart-amazon",
        "amazon-google",
    }


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
        assert ds.ground_truth == {(0, RIGHT_ID_OFFSET)}


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


def test_find_zip_member_nested_exp_data() -> None:
    """DeepMatcher zips nest CSVs under exp_data/."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("exp_data/tableA.csv", "id\n1\n")
        zf.writestr("__MACOSX/tableA.csv", "skip\n")
        assert _find_zip_member(zf, "tableA.csv") == "exp_data/tableA.csv"
        assert _find_zip_member(zf, "missing.csv") is None


class _FakeUrlResponse:
    """Minimal urlopen context manager returning zip bytes."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_FakeUrlResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _deepmatcher_zip_bytes() -> bytes:
    """Build a nested DeepMatcher exp_data zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("exp_data/tableA.csv", "id,title\n10,Widget\n20,Gadget\n")
        zf.writestr("exp_data/tableB.csv", "id,title\n30,Widget\n40,Other\n")
        zf.writestr("exp_data/train.csv", "ltable_id,rtable_id,label\n10,30,1\n20,40,0\n")
        zf.writestr("exp_data/valid.csv", "ltable_id,rtable_id,label\n")
        zf.writestr("exp_data/test.csv", "ltable_id,rtable_id,label\n")
    return buf.getvalue()


def test_download_deepmatcher_nested_zip() -> None:
    """Download loads DeepMatcher tableA/tableB and labeled matches from a zip."""
    data = _deepmatcher_zip_bytes()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(urllib.request, "urlopen", return_value=_FakeUrlResponse(data)):
            ds = BenchmarkDataset.download("walmart-amazon", tmpdir)
        assert len(ds.table_a) == 2
        assert len(ds.table_b) == 2
        assert ds.ground_truth == {(0, RIGHT_ID_OFFSET)}
        assert str(ds.table_a.iloc[0]["title"]) == "Widget"


def test_download_amazon_google_uses_deepmatcher_registry() -> None:
    """amazon-google is registered as DeepMatcher (no Leipzig mapping_name)."""
    assert "mapping_name" not in DATASET_REGISTRY["amazon-google"]
    assert "mapping_name" not in DATASET_REGISTRY["walmart-amazon"]
    assert "mapping_name" in DATASET_REGISTRY["dblp-acm"]
