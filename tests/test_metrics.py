"""Tests for entity resolution evaluation metrics."""

import pytest

from serf.dspy.types import Entity
from serf.eval.metrics import (
    cluster_f1,
    evaluate_resolution,
    f1_score,
    pair_completeness,
    precision,
    recall,
    reduction_ratio,
    validate_source_uuids,
)


def test_precision_perfect_predictions() -> None:
    """Precision is 1.0 when all predictions are correct."""
    pred = {(1, 2), (3, 4)}
    true = {(1, 2), (3, 4)}
    assert precision(pred, true) == 1.0


def test_precision_no_correct_predictions() -> None:
    """Precision is 0.0 when no predictions are correct."""
    pred = {(1, 2), (3, 4)}
    true = {(5, 6), (7, 8)}
    assert precision(pred, true) == 0.0


def test_precision_partial() -> None:
    """Precision is correct for partial overlap."""
    pred = {(1, 2), (3, 4), (5, 6)}
    true = {(1, 2), (3, 4), (7, 8)}
    assert precision(pred, true) == pytest.approx(2 / 3)


def test_precision_empty_predicted_returns_zero() -> None:
    """Precision returns 0.0 when predicted_pairs is empty."""
    pred: set[tuple[int, int]] = set()
    true = {(1, 2)}
    assert precision(pred, true) == 0.0


def test_recall_perfect_predictions() -> None:
    """Recall is 1.0 when all true pairs are found."""
    pred = {(1, 2), (3, 4)}
    true = {(1, 2), (3, 4)}
    assert recall(pred, true) == 1.0


def test_recall_no_correct_predictions() -> None:
    """Recall is 0.0 when no true pairs are found."""
    pred = {(1, 2), (3, 4)}
    true = {(5, 6), (7, 8)}
    assert recall(pred, true) == 0.0


def test_recall_partial() -> None:
    """Recall is correct for partial overlap."""
    pred = {(1, 2), (3, 4)}
    true = {(1, 2), (3, 4), (5, 6)}
    assert recall(pred, true) == pytest.approx(2 / 3)


def test_recall_empty_true_returns_zero() -> None:
    """Recall returns 0.0 when true_pairs is empty."""
    pred = {(1, 2)}
    true: set[tuple[int, int]] = set()
    assert recall(pred, true) == 0.0


def test_f1_score_known_inputs() -> None:
    """F1 is harmonic mean of precision and recall."""
    pred = {(1, 2), (3, 4)}
    true = {(1, 2), (5, 6)}
    p = 1 / 2
    r = 1 / 2
    expected = 2 * p * r / (p + r)
    assert f1_score(pred, true) == pytest.approx(expected)


def test_f1_score_perfect() -> None:
    """F1 is 1.0 when predictions match ground truth exactly."""
    pred = {(1, 2), (3, 4)}
    true = {(1, 2), (3, 4)}
    assert f1_score(pred, true) == 1.0


def test_f1_score_empty_both_returns_zero() -> None:
    """F1 returns 0.0 when both sets are empty."""
    pred: set[tuple[int, int]] = set()
    true: set[tuple[int, int]] = set()
    assert f1_score(pred, true) == 0.0


def test_pair_completeness_all_retained() -> None:
    """Pair completeness is 1.0 when all true pairs retained after blocking."""
    blocked = {(1, 2), (3, 4)}
    true = {(1, 2), (3, 4)}
    assert pair_completeness(blocked, true) == 1.0


def test_pair_completeness_partial() -> None:
    """Pair completeness is correct for partial retention."""
    blocked = {(1, 2)}
    true = {(1, 2), (3, 4)}
    assert pair_completeness(blocked, true) == 0.5


def test_pair_completeness_empty_true_returns_zero() -> None:
    """Pair completeness returns 0.0 when true_pairs is empty."""
    blocked = {(1, 2)}
    true: set[tuple[int, int]] = set()
    assert pair_completeness(blocked, true) == 0.0


def test_reduction_ratio() -> None:
    """Reduction ratio is 1 - (blocked / total)."""
    assert reduction_ratio(100, 1000) == pytest.approx(0.9)
    assert reduction_ratio(0, 100) == 1.0
    assert reduction_ratio(100, 100) == 0.0


def test_reduction_ratio_empty_total_returns_zero() -> None:
    """Reduction ratio returns 0.0 when total_possible_pairs is 0."""
    assert reduction_ratio(0, 0) == 0.0
    assert reduction_ratio(10, 0) == 0.0


def test_cluster_f1_matching_clusters() -> None:
    """Cluster F1 is 1.0 when predicted clusters match true clusters."""
    pred_clusters = {0: {1, 2, 3}, 1: {4, 5}}
    true_clusters = {0: {1, 2, 3}, 1: {4, 5}}
    assert cluster_f1(pred_clusters, true_clusters) == 1.0


def test_cluster_f1_non_matching_clusters() -> None:
    """Cluster F1 is 0.0 when no overlap between predicted and true pairs."""
    pred_clusters = {0: {1, 2}, 1: {3, 4}}
    true_clusters = {0: {5, 6}, 1: {7, 8}}
    assert cluster_f1(pred_clusters, true_clusters) == 0.0


def test_cluster_f1_partial_overlap() -> None:
    """Cluster F1 is correct for partial cluster overlap."""
    pred_clusters = {0: {1, 2, 3}}
    true_clusters = {0: {1, 2, 4}}
    pred_pairs = {(1, 2), (1, 3), (2, 3)}
    true_pairs = {(1, 2), (1, 4), (2, 4)}
    expected = f1_score(pred_pairs, true_pairs)
    assert cluster_f1(pred_clusters, true_clusters) == pytest.approx(expected)


def test_evaluate_resolution_returns_all_keys() -> None:
    """evaluate_resolution returns dict with precision, recall, f1_score."""
    pred = {(1, 2), (3, 4)}
    true = {(1, 2), (5, 6)}
    result = evaluate_resolution(pred, true)
    assert {"precision", "recall", "f1_score"}.issubset(set(result.keys()))
    assert "precision" in result
    assert "recall" in result
    assert "f1_score" in result


def test_evaluate_resolution_values_match_individual_functions() -> None:
    """evaluate_resolution values match individual metric functions."""
    pred = {(1, 2), (3, 4)}
    true = {(1, 2), (5, 6)}
    result = evaluate_resolution(pred, true)
    assert result["precision"] == precision(pred, true)
    assert result["recall"] == recall(pred, true)
    assert result["f1_score"] == f1_score(pred, true)


def test_edge_case_empty_predicted_sets() -> None:
    """Empty predicted sets yield 0.0 for precision and recall."""
    pred: set[tuple[int, int]] = set()
    true = {(1, 2)}
    assert precision(pred, true) == 0.0
    assert recall(pred, true) == 0.0
    assert f1_score(pred, true) == 0.0


def test_edge_case_empty_true_sets() -> None:
    """Empty true sets yield 0.0 for recall and f1."""
    pred = {(1, 2)}
    true: set[tuple[int, int]] = set()
    assert precision(pred, true) == 0.0
    assert recall(pred, true) == 0.0
    assert f1_score(pred, true) == 0.0


def test_normalize_pairs_handles_ordering() -> None:
    """Pairs (a,b) and (b,a) are treated as the same."""
    pred = {(2, 1), (4, 3)}
    true = {(1, 2), (3, 4)}
    assert precision(pred, true) == 1.0
    assert recall(pred, true) == 1.0


def test_validate_source_uuids_all_valid() -> None:
    """validate_source_uuids passes when all source_uuids are known."""
    entities = [
        Entity(id=0, name="A", source_uuids=["uuid-1", "uuid-2"]),
        Entity(id=1, name="B", source_uuids=["uuid-3"]),
    ]
    historical = {"uuid-1", "uuid-2", "uuid-3", "uuid-4"}
    result = validate_source_uuids(entities, historical)
    assert result["total_entities"] == 2
    assert result["total_source_uuids"] == 3
    assert result["valid_source_uuids"] == 3
    assert result["invalid_source_uuids"] == 0
    assert result["coverage_pct"] == 100.0
    assert result["passed"] is True
    assert result["missing_uuids"] == []


def test_validate_source_uuids_with_invalid() -> None:
    """validate_source_uuids detects invalid source_uuids."""
    entities = [
        Entity(id=0, name="A", source_uuids=["uuid-1", "uuid-bad"]),
        Entity(id=1, name="B", source_uuids=["uuid-2"]),
    ]
    historical = {"uuid-1", "uuid-2"}
    result = validate_source_uuids(entities, historical)
    assert result["total_source_uuids"] == 3
    assert result["valid_source_uuids"] == 2
    assert result["invalid_source_uuids"] == 1
    assert result["passed"] is False
    assert "uuid-bad" in result["missing_uuids"]


def test_validate_source_uuids_empty_entities() -> None:
    """validate_source_uuids handles entities with no source_uuids."""
    entities = [
        Entity(id=0, name="A"),
        Entity(id=1, name="B"),
    ]
    result = validate_source_uuids(entities, set())
    assert result["total_source_uuids"] == 0
    assert result["coverage_pct"] == 100.0
    assert result["passed"] is True
