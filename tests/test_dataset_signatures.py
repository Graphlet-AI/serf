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


@pytest.mark.parametrize("dataset", DATASETS)
def test_signature_instructions_warn_about_the_one_attribute_difference(dataset: str) -> None:
    """The cross-dataset finding from BENCHMARKS.md reaches every prompt.

    The hardest non-match differs from a true match on one short attribute in all
    five tasks, so the rule belongs in the shared block rules rather than in five
    separate docstrings.
    """
    instructions = " ".join(get_dataset_spec(dataset).signature.instructions.split())

    assert "differ from true matches on exactly one attribute" in instructions


@pytest.mark.parametrize("dataset", DATASETS)
def test_signature_instructions_bound_every_veto_by_coverage(dataset: str) -> None:
    """A discriminative attribute may only rule a pair out when it is present.

    Stating the agreement rates without the coverage rates turns each of them
    into a veto the data does not support: `modelno` decides Walmart-Amazon but
    is unusable on 31.8% of its gold pairs, `manufacturer` is unusable on 82.2%
    of Amazon-Google's, and price on 79.4% of Abt-Buy's. Measured over all five
    datasets, the vetoing phrasing cost 3.7 F1 points against the prompts it
    replaced, almost all of it recall.
    """
    instructions = " ".join(get_dataset_spec(dataset).signature.instructions.split())

    assert "A value that is missing on either side is not a disagreement" in instructions
    assert "Never reject a pair for failing a test it had no way to take" in instructions


# One measured finding per dataset that the profiler in `serf profile-benchmark`
# established and that the signature used to contradict. Each phrase is the
# instruction the measurement forced; losing one is a silent regression back to
# an instruction the data says is wrong.
MEASURED_FINDINGS: dict[str, tuple[str, ...]] = {
    "dblp-acm": (
        # Year is equal on 100% of gold pairs and 12.8% of near misses, so the
        # earlier "within one year" tolerance let the journal-extension false
        # positive straight through.
        "Require the years to be equal",
        "Do not allow a year of slack",
        # Venue agrees on 0% of gold pairs as a string and is a five-row bijection.
        "Never compare venue as a string",
    ),
    "dblp-scholar": (
        # Scholar writes 2002.0, so string equality fires on 0% of gold pairs
        # and numeric comparison on 99.96%.
        "Compare the year as a number, never as a string",
        # Whitespace is dropped in ~1% of Scholar titles, which zeroes word-token
        # overlap on pairs that are the same paper.
        "Compare titles by their characters, not by their words",
    ),
    "abt-buy": (
        # Containment finds 82% of gold pairs against 47% for exact equality, at
        # a 2.5% near-miss rate. This is the strongest single finding measured.
        "Then test containment, not equality",
        # 85% of near misses also carry codes on both sides, so presence is not
        # evidence; only the containment relation is.
        "mere presence of a code on each side is worth nothing",
    ),
    "amazon-google": (
        # Only 15% of pairs carry a code, so hunting for one is wasted effort.
        "no model codes to fall back on",
        # Price is within 25% on 79.9% of gold pairs against 24.0% of near misses
        # and is comparable on 89.6% of them.
        "most useful attribute here after the title",
    ),
    "walmart-amazon": (
        # Amazon's modelno holds descriptive text often enough that inequality
        # cannot be trusted before the value is checked.
        "A value with no digits and more than one word is prose, not a code",
        # modelno is unusable on 31.8% of gold pairs, so treating its absence as
        # a rejection costs a third of the recall on this task.
        "Blank or prose on either side means the model number is silent, not negative",
        # Category agrees on 4.4% of gold pairs against 2.2% of near misses, and
        # the Walmart label is frequently wrong outright.
        "Ignore the category field completely",
    ),
}


@pytest.mark.parametrize("dataset", DATASETS)
def test_signature_instructions_carry_the_measured_findings(dataset: str) -> None:
    """Each dataset's prompt states what profiling measured, not what was assumed."""
    instructions = " ".join(get_dataset_spec(dataset).signature.instructions.split())

    for finding in MEASURED_FINDINGS[dataset]:
        assert finding in instructions, f"{dataset} lost the measured finding: {finding}"


def test_dblp_acm_no_longer_tolerates_a_year_of_slack() -> None:
    """The DBLP-ACM year rule is equality, because the measurement says so.

    DBLP-Scholar legitimately allows a year of slack, since Scholar may have
    crawled a preprint. DBLP-ACM does not: both sources are curated catalogues
    of the same venues and every gold pair shares a year, so tolerance there only
    admits the conference-paper-against-journal-extension false positive.
    """
    acm = " ".join(DblpAcmBlockMatch.instructions.split())
    scholar = " ".join(DblpScholarBlockMatch.instructions.split())

    assert "Allow one year of slack" not in acm
    assert "Allow one year of slack" in scholar


def test_price_guidance_matches_how_useful_price_measured_per_dataset() -> None:
    """Price is a tie-breaker where it separates the populations, ignored where it does not.

    On Amazon-Google price is the sharpest non-title attribute (79.9% of gold
    pairs within 25% against 24.0% of near misses). On Abt-Buy four gold pairs in
    five have no comparable price at all. Both prompts have to say which case
    they are in.
    """
    amazon_google = " ".join(AmazonGoogleBlockMatch.instructions.split())
    abt_buy = " ".join(AbtBuyBlockMatch.instructions.split())

    assert "tie-breaker" in amazon_google
    assert "no comparable price at all" in abt_buy


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
