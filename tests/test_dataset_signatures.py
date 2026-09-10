"""Tests for the per-dataset DSPy signatures and their registry."""

from typing import get_args, get_origin

import dspy
import pytest

from serf.dspy.dataset_signatures import (
    DATASET_SIGNATURES,
    SIGNATURE_MODE_GENERIC,
    SIGNATURE_MODE_PER_DATASET,
    SIGNATURE_MODES,
    AbtBuyBlockMatch,
    AmazonGoogleBlockMatch,
    DblpAcmBlockMatch,
    DblpScholarBlockMatch,
    WalmartAmazonBlockMatch,
    get_dataset_spec,
)
from serf.dspy.signatures import BlockMatch
from serf.eval.benchmarks import DATASET_REGISTRY

DATASETS = sorted(DATASET_SIGNATURES)


def test_registry_covers_every_benchmark_dataset() -> None:
    """Every dataset in the benchmark registry has a per-dataset signature."""
    assert set(DATASET_SIGNATURES) == set(DATASET_REGISTRY)


@pytest.mark.parametrize(
    ("dataset", "signature"),
    [
        ("dblp-acm", DblpAcmBlockMatch),
        ("dblp-scholar", DblpScholarBlockMatch),
        ("abt-buy", AbtBuyBlockMatch),
        ("amazon-google", AmazonGoogleBlockMatch),
        ("walmart-amazon", WalmartAmazonBlockMatch),
    ],
)
def test_registry_returns_the_right_signature(
    dataset: str, signature: type[dspy.Signature]
) -> None:
    """The registry maps each dataset name onto its own signature."""
    spec = get_dataset_spec(dataset)

    assert spec.signature is signature
    assert spec.dataset == dataset


def test_get_dataset_spec_rejects_unknown_datasets() -> None:
    """An unknown dataset name raises rather than silently falling back."""
    with pytest.raises(ValueError, match="No per-dataset signature"):
        get_dataset_spec("cora")


@pytest.mark.parametrize("dataset", DATASETS)
def test_signature_fields_are_typed_per_side(dataset: str) -> None:
    """Each signature takes one typed list per source and returns typed candidates."""
    spec = get_dataset_spec(dataset)
    signature = spec.signature

    assert set(signature.input_fields) == {spec.left_field, spec.right_field}
    assert set(signature.output_fields) == {spec.candidates_field}
    for field_name, item_type in (
        (spec.left_field, spec.left_type),
        (spec.right_field, spec.right_type),
    ):
        annotation = signature.input_fields[field_name].annotation
        assert get_origin(annotation) is list
        assert get_args(annotation) == (item_type,)
    candidates = signature.output_fields[spec.candidates_field].annotation
    assert get_origin(candidates) is list
    assert get_args(candidates) == (spec.candidate_type,)


@pytest.mark.parametrize("dataset", DATASETS)
def test_signature_side_types_are_distinct(dataset: str) -> None:
    """The two sides of a dataset are modelled by two different classes."""
    spec = get_dataset_spec(dataset)

    assert spec.left_type is not spec.right_type


@pytest.mark.parametrize("dataset", DATASETS)
def test_signature_can_create_predict(dataset: str) -> None:
    """Every per-dataset signature works with dspy.Predict."""
    predictor = dspy.Predict(get_dataset_spec(dataset).signature)

    assert predictor is not None


@pytest.mark.parametrize("dataset", DATASETS)
def test_signature_instructions_carry_the_field_guide(dataset: str) -> None:
    """Instructions describe every field of both sides, since DSPy drops the schema."""
    spec = get_dataset_spec(dataset)
    instructions = spec.signature.instructions

    assert "Fields of every" in instructions
    for side in (spec.left_type, spec.right_type):
        for name in side.model_fields:
            assert f"- {name}:" in instructions


@pytest.mark.parametrize("dataset", DATASETS)
def test_signature_instructions_forbid_same_source_pairs(dataset: str) -> None:
    """The shared block rules make the bipartite structure of the task explicit."""
    instructions = " ".join(get_dataset_spec(dataset).signature.instructions.split())

    assert "two records from the same source are never a match" in instructions


def test_signature_modes_are_generic_and_per_dataset() -> None:
    """The two selectable modes are the generic and per-dataset contracts."""
    assert SIGNATURE_MODES == (SIGNATURE_MODE_GENERIC, SIGNATURE_MODE_PER_DATASET)


def test_generic_block_match_signature_is_unchanged() -> None:
    """The shared BlockMatch contract is untouched by the per-dataset work."""
    assert set(BlockMatch.input_fields) == {
        "block_records",
        "schema_info",
        "few_shot_examples",
    }
    assert set(BlockMatch.output_fields) == {"resolution"}
