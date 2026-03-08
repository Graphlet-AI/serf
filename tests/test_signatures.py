"""Tests for DSPy signature definitions."""

import dspy

from serf.dspy.signatures import AnalyzeDataset, BlockMatch, EdgeResolve, EntityMerge


def test_block_match_signature_fields() -> None:
    """Test BlockMatch has the expected input/output fields."""
    assert "block_records" in BlockMatch.input_fields
    assert "schema_info" in BlockMatch.input_fields
    assert "few_shot_examples" in BlockMatch.input_fields
    assert "resolution" in BlockMatch.output_fields


def test_entity_merge_signature_fields() -> None:
    """Test EntityMerge has the expected input/output fields."""
    assert "entity_a" in EntityMerge.input_fields
    assert "entity_b" in EntityMerge.input_fields
    assert "merged" in EntityMerge.output_fields


def test_edge_resolve_signature_fields() -> None:
    """Test EdgeResolve has the expected input/output fields."""
    assert "edge_block" in EdgeResolve.input_fields
    assert "resolved_edges" in EdgeResolve.output_fields


def test_analyze_dataset_signature_fields() -> None:
    """Test AnalyzeDataset has the expected input/output fields."""
    assert "dataset_summary" in AnalyzeDataset.input_fields
    assert "profile" in AnalyzeDataset.output_fields


def test_block_match_can_create_predict() -> None:
    """Test that BlockMatch can be used with dspy.Predict."""
    predictor = dspy.Predict(BlockMatch)
    assert predictor is not None


def test_entity_merge_can_create_predict() -> None:
    """Test that EntityMerge can be used with dspy.Predict."""
    predictor = dspy.Predict(EntityMerge)
    assert predictor is not None


def test_edge_resolve_can_create_predict() -> None:
    """Test that EdgeResolve can be used with dspy.Predict."""
    predictor = dspy.Predict(EdgeResolve)
    assert predictor is not None


def test_analyze_dataset_can_create_predict() -> None:
    """Test that AnalyzeDataset can be used with dspy.Predict."""
    predictor = dspy.Predict(AnalyzeDataset)
    assert predictor is not None
