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


# DBLP-Scholar and Walmart-Amazon keep the shorter literature-derived prompt.
# The profiling holds for them too, but writing it into the instructions cost
# recall in four successive framings (0.889/0.903/0.897/0.881 against 0.919, and
# 0.800/0.851/0.870/0.855 against 0.892), because the attribute each finding
# turns on is missing on a large share of their gold pairs. Their findings stay
# in BENCHMARKS.md as documentation of the data.
RETAINED_LITERATURE_PROMPTS = frozenset({"dblp-scholar", "walmart-amazon"})

REWRITTEN_PROMPTS = sorted(set(DATASETS) - RETAINED_LITERATURE_PROMPTS)


@pytest.mark.parametrize("dataset", REWRITTEN_PROMPTS)
def test_signature_instructions_warn_about_the_one_attribute_difference(dataset: str) -> None:
    """The cross-dataset finding from BENCHMARKS.md reaches the prompts it helps.

    The hardest non-match differs from a true match on one short attribute in all
    five tasks, but "so compare that field exactly" is only safe where the field
    is there to compare. Carried in the shared block rules it cost DBLP-Scholar
    and Walmart-Amazon recall, so it sits in the three docstrings that measured a
    gain from it instead.
    """
    instructions = " ".join(get_dataset_spec(dataset).signature.instructions.split())

    assert "differ from true matches on exactly one attribute" in instructions


@pytest.mark.parametrize("dataset", sorted(RETAINED_LITERATURE_PROMPTS))
def test_retained_prompts_do_not_carry_the_one_attribute_rule(dataset: str) -> None:
    """The two datasets it hurt must not pick it back up through the shared rules."""
    instructions = " ".join(get_dataset_spec(dataset).signature.instructions.split())

    assert "differ from true matches on exactly one attribute" not in instructions


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
}


@pytest.mark.parametrize("dataset", sorted(MEASURED_FINDINGS))
def test_signature_instructions_carry_the_measured_findings(dataset: str) -> None:
    """Each rewritten prompt states what profiling measured, not what was assumed."""
    instructions = " ".join(get_dataset_spec(dataset).signature.instructions.split())

    for finding in MEASURED_FINDINGS[dataset]:
        assert finding in instructions, f"{dataset} lost the measured finding: {finding}"


@pytest.mark.parametrize("dataset", sorted(RETAINED_LITERATURE_PROMPTS))
def test_retained_prompts_say_why_the_measured_rewrite_was_dropped(dataset: str) -> None:
    """Keeping the shorter prompt is a measured decision, so the prompt records it.

    Without the note the next reader sees two datasets whose instructions ignore
    BENCHMARKS.md and reasonably assumes nobody got to them.
    """
    doc = get_dataset_spec(dataset).signature.__doc__ or ""

    assert "Measured rewrite deliberately not applied here" in doc
    assert "BENCHMARKS.md" in doc


def test_measured_findings_and_retained_prompts_cover_every_dataset() -> None:
    """Every dataset is either rewritten from the profiling or knowingly left alone."""
    assert set(MEASURED_FINDINGS) | RETAINED_LITERATURE_PROMPTS == set(DATASETS)
    assert not set(MEASURED_FINDINGS) & RETAINED_LITERATURE_PROMPTS


def test_dblp_acm_no_longer_tolerates_a_year_of_slack() -> None:
    """The DBLP-ACM year rule is equality, because the measurement says so.

    DBLP-Scholar legitimately allows a year of slack, since Scholar may have
    crawled a preprint. DBLP-ACM does not: both sources are curated catalogues
    of the same venues and every gold pair shares a year, so tolerance there only
    admits the conference-paper-against-journal-extension false positive.
    """
    acm = " ".join(DblpAcmBlockMatch.instructions.split())
    scholar = " ".join(DblpScholarBlockMatch.instructions.split())

    assert "Require the years to be equal" in acm
    assert "Do not allow a year of slack" in acm
    assert "agree within about a year" in scholar


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
