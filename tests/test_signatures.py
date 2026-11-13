"""Tests for DSPy signature definitions."""

import dspy

from serf.dspy.schemas.base import (
    AddressList,
    NameList,
    SingularAddress,
    SingularName,
)
from serf.dspy.signatures import (
    AnalyzeDataset,
    BlockMatch,
    DeduplicateListValues,
    EdgeResolve,
    EntityMerge,
    GenerateERConfig,
    SelectBestValue,
)


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


def test_generate_er_config_signature_fields() -> None:
    """Test GenerateERConfig has the expected input/output fields."""
    assert "dataset_profile" in GenerateERConfig.input_fields
    assert "sample_records" in GenerateERConfig.input_fields
    assert "er_config_yaml" in GenerateERConfig.output_fields


def test_generate_er_config_can_create_predict() -> None:
    """Test that GenerateERConfig can be used with dspy.Predict."""
    predictor = dspy.Predict(GenerateERConfig)
    assert predictor is not None


def test_select_best_value_signature_fields() -> None:
    """Test SelectBestValue has the expected input/output fields."""
    assert "field_name" in SelectBestValue.input_fields
    assert "field_type" in SelectBestValue.input_fields
    assert "field_description" in SelectBestValue.input_fields
    assert "candidate_values" in SelectBestValue.input_fields
    assert "best_value" in SelectBestValue.output_fields


def test_select_best_value_can_create_predict() -> None:
    """Test that SelectBestValue can be used with dspy.Predict."""
    predictor = dspy.Predict(SelectBestValue)
    assert predictor is not None


def test_deduplicate_list_values_signature_fields() -> None:
    """Test DeduplicateListValues has the expected input/output fields."""
    assert "field_name" in DeduplicateListValues.input_fields
    assert "field_type" in DeduplicateListValues.input_fields
    assert "field_description" in DeduplicateListValues.input_fields
    assert "candidate_values" in DeduplicateListValues.input_fields
    assert "deduplicated_values" in DeduplicateListValues.output_fields
    assert DeduplicateListValues.output_fields["deduplicated_values"].annotation == list[str]


def test_dspy_field_annotations_work_directly_in_custom_signature() -> None:
    """Reusable DSPy Annotated field types carry merge semantics into a dspy.Signature."""

    class MergeContactSignature(dspy.Signature):
        """Merge contact records."""

        records: list[dict[str, object]] = dspy.InputField()
        primary_name: SingularName
        primary_address: SingularAddress
        known_names: NameList
        known_addresses: AddressList

    assert MergeContactSignature.output_fields["primary_name"].annotation is str
    assert MergeContactSignature.output_fields["primary_address"].annotation is str
    assert MergeContactSignature.output_fields["known_names"].annotation == list[str]
    assert MergeContactSignature.output_fields["known_addresses"].annotation == list[str]
    assert (
        "individual names"
        in (MergeContactSignature.output_fields["known_names"].description or "").lower()
    )
    assert (
        "individual addresses"
        in (MergeContactSignature.output_fields["known_addresses"].description or "").lower()
    )
