"""Tests for per-dataset block matching."""

from typing import Any
from unittest.mock import patch

import dspy
import pytest

from serf.dspy.dataset_signatures import (
    SIGNATURE_MODE_GENERIC,
    SIGNATURE_MODE_PER_DATASET,
    WalmartAmazonBlockMatch,
    get_dataset_spec,
)
from serf.dspy.schemas import (
    AmazonElectronicsProduct,
    WalmartAmazonCandidate,
    WalmartProduct,
)
from serf.dspy.signatures import BlockMatch
from serf.dspy.types import Entity, EntityBlock
from serf.match.dataset_matcher import SINGLE_SOURCE_SKIP_REASON, DatasetMatcher, typed_sides
from serf.match.matcher import EntityMatcher
from serf.match.run import collect_pairs, create_matcher

RIGHT_ID_OFFSET = 100000


def _walmart_entity(entity_id: int) -> Entity:
    """Build a left-side Walmart entity."""
    return Entity(
        id=entity_id,
        name="draper infrared remote transmitter",
        attributes={
            "l_id": str(entity_id),
            "l_title": "draper infrared remote transmitter",
            "l_category": "electronics - general",
            "l_brand": "draper",
            "l_modelno": "121066",
            "l_price": "58.45",
        },
    )


def _amazon_entity(entity_id: int) -> Entity:
    """Build a right-side Amazon entity."""
    return Entity(
        id=entity_id,
        name="draper 121066 infrared remote transmitter",
        attributes={
            "r_id": str(entity_id - RIGHT_ID_OFFSET),
            "r_title": "draper 121066 infrared remote transmitter",
            "r_category": "home audio accessories",
            "r_brand": "draper",
            "r_modelno": "121066",
            "r_price": "52.10",
        },
    )


def _block(entities: list[Entity]) -> EntityBlock:
    """Wrap entities in a block."""
    return EntityBlock(
        block_key="b0",
        block_size=len(entities),
        entities=entities,
    )


class _StubPredict:
    """Stand-in for dspy.Predict that returns fixed candidates."""

    def __init__(self, candidates: list[WalmartAmazonCandidate]) -> None:
        self.candidates = candidates
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dspy.Prediction:
        """Record the call and return the fixed candidates."""
        self.calls.append(kwargs)
        return dspy.Prediction(candidates=self.candidates)


def _candidate(left_id: int, right_id: int) -> WalmartAmazonCandidate:
    """Build a matched candidate over two mapped record ids."""
    return WalmartAmazonCandidate(
        left=WalmartProduct(record_id=left_id, source_id="7", modelno="121066"),
        right=AmazonElectronicsProduct(record_id=right_id, source_id="12", modelno="121066"),
        is_match=True,
        confidence=0.95,
        justification="identical model number 121066",
    )


def test_dataset_matcher_uses_the_dataset_signature() -> None:
    """The matcher predicts with the dataset's signature, not with BlockMatch."""
    matcher = DatasetMatcher("walmart-amazon")

    assert matcher.predictor.signature is WalmartAmazonBlockMatch
    assert matcher.predictor.signature is not BlockMatch


def test_typed_sides_splits_a_block_by_source() -> None:
    """typed_sides maps each entity onto its own side's model."""
    spec = get_dataset_spec("walmart-amazon")
    block = _block([_walmart_entity(3), _amazon_entity(RIGHT_ID_OFFSET + 4)])

    left, right = typed_sides(block, spec)

    assert [type(record) for record in left] == [WalmartProduct]
    assert [type(record) for record in right] == [AmazonElectronicsProduct]
    assert left[0].modelno == "121066"
    assert right[0].price == pytest.approx(52.10)


def test_resolve_block_returns_matches_with_original_ids() -> None:
    """Candidates over mapped ids come back as matches over the original ids."""
    matcher = DatasetMatcher("walmart-amazon")
    block = _block([_walmart_entity(7), _amazon_entity(RIGHT_ID_OFFSET + 12)])
    stub = _StubPredict([_candidate(0, 1)])
    matcher._predictor = stub  # type: ignore[assignment]

    with patch.object(EntityMatcher, "_ensure_lm", return_value=None):
        resolution = matcher.resolve_block(block)

    assert len(resolution.matches) == 1
    match = resolution.matches[0]
    assert {match.entity_a_id, match.entity_b_id} == {7, RIGHT_ID_OFFSET + 12}
    assert match.reasoning == "identical model number 121066"
    assert resolution.was_resolved is True
    assert len(resolution.resolved_entities) == 2
    assert all(entity.uuid for entity in resolution.resolved_entities)


def test_resolve_block_feeds_each_source_its_own_input_field() -> None:
    """Each side of the block is passed as its own typed input field."""
    matcher = DatasetMatcher("walmart-amazon")
    block = _block([_walmart_entity(1), _amazon_entity(RIGHT_ID_OFFSET + 2)])
    stub = _StubPredict([])
    matcher._predictor = stub  # type: ignore[assignment]

    with patch.object(EntityMatcher, "_ensure_lm", return_value=None):
        matcher.resolve_block(block)

    call = stub.calls[0]
    assert set(call) == {"walmart_records", "amazon_records"}
    assert isinstance(call["walmart_records"][0], WalmartProduct)
    assert isinstance(call["amazon_records"][0], AmazonElectronicsProduct)


def test_single_source_block_skips_the_llm_call() -> None:
    """A block holding one source only cannot contain a cross-source pair."""
    matcher = DatasetMatcher("walmart-amazon")
    block = _block([_walmart_entity(1), _walmart_entity(2)])
    stub = _StubPredict([_candidate(0, 1)])
    matcher._predictor = stub  # type: ignore[assignment]

    with patch.object(EntityMatcher, "_ensure_lm", return_value=None):
        resolution = matcher.resolve_block(block)

    assert stub.calls == []
    assert resolution.matches == []
    assert matcher.single_source_blocks == 1
    assert all(
        entity.match_skip_reason == SINGLE_SOURCE_SKIP_REASON
        for entity in resolution.resolved_entities
    )


def test_candidates_with_unknown_record_ids_are_dropped() -> None:
    """A hallucinated record id is dropped instead of becoming a match."""
    matcher = DatasetMatcher("walmart-amazon")
    block = _block([_walmart_entity(5), _amazon_entity(RIGHT_ID_OFFSET + 6)])
    stub = _StubPredict([_candidate(0, 99)])
    matcher._predictor = stub  # type: ignore[assignment]

    with patch.object(EntityMatcher, "_ensure_lm", return_value=None):
        resolution = matcher.resolve_block(block)

    assert resolution.matches == []
    assert matcher.unknown_record_ids == 1


def test_llm_failure_falls_back_to_error_recovery() -> None:
    """An exception from the LM marks the block as error_recovery."""
    matcher = DatasetMatcher("walmart-amazon")
    block = _block([_walmart_entity(1), _amazon_entity(RIGHT_ID_OFFSET + 2)])

    def _raise(**_kwargs: Any) -> dspy.Prediction:
        raise RuntimeError("endpoint exploded")

    matcher._predictor = _raise  # type: ignore[assignment]

    with patch.object(EntityMatcher, "_ensure_lm", return_value=None):
        resolution = matcher.resolve_block(block)

    assert resolution.matches == []
    assert all(
        entity.match_skip_reason == "error_recovery" for entity in resolution.resolved_entities
    )


def test_create_matcher_defaults_to_the_generic_signature() -> None:
    """The default mode still returns the generic BlockMatch matcher."""
    matcher = create_matcher()

    assert isinstance(matcher, EntityMatcher)
    assert not isinstance(matcher, DatasetMatcher)
    assert matcher.predictor.signature is BlockMatch


def test_create_matcher_per_dataset_returns_a_dataset_matcher() -> None:
    """Per-dataset mode returns a matcher bound to that dataset's signature."""
    matcher = create_matcher(signature_mode=SIGNATURE_MODE_PER_DATASET, dataset="abt-buy")

    assert isinstance(matcher, DatasetMatcher)
    assert matcher.dataset == "abt-buy"


def test_create_matcher_requires_a_dataset_for_per_dataset_mode() -> None:
    """Per-dataset mode without a dataset name is a usage error."""
    with pytest.raises(ValueError, match="requires a dataset name"):
        create_matcher(signature_mode=SIGNATURE_MODE_PER_DATASET)


def test_create_matcher_rejects_unknown_modes() -> None:
    """An unknown signature mode is rejected."""
    with pytest.raises(ValueError, match="Unknown signature mode"):
        create_matcher(signature_mode="embedding")


def test_create_matcher_passes_the_trained_prompt_request_through() -> None:
    """The benchmark flag has to reach the matcher that loads the program."""
    matcher = create_matcher(
        signature_mode=SIGNATURE_MODE_PER_DATASET,
        dataset="abt-buy",
        trained_prompts=True,
        trained_dir="somewhere",
    )

    assert isinstance(matcher, DatasetMatcher)
    assert matcher.trained_prompts is True
    assert matcher.trained_dir == "somewhere"


def test_create_matcher_rejects_trained_prompts_for_the_generic_signature() -> None:
    """`serf train` writes per-dataset programs; the shared signature has none."""
    with pytest.raises(ValueError, match="requires signature_mode 'per-dataset'"):
        create_matcher(trained_prompts=True)


def test_matcher_defaults_to_the_signature_as_written() -> None:
    """Trained prompts are opt-in, so an untrained repo behaves exactly as before."""
    matcher = DatasetMatcher("abt-buy")

    assert matcher.trained_prompts is False
    assert matcher.predictor.signature is get_dataset_spec("abt-buy").signature


def test_matcher_matches_with_the_trained_instructions_when_asked(tmp_path: Any) -> None:
    """A trained program replaces the docstring without anyone editing the source."""
    spec = get_dataset_spec("abt-buy")
    trained = dspy.Predict(spec.signature)
    trained.signature = spec.signature.with_instructions("Trained abt-buy instructions.")
    trained.save(str(tmp_path / "abt-buy_gepa.json"))

    matcher = DatasetMatcher("abt-buy", trained_prompts=True, trained_dir=str(tmp_path))

    signature = matcher.predictor.signature
    assert signature is not None
    assert signature.instructions == "Trained abt-buy instructions."
    assert list(signature.input_fields) == [spec.left_field, spec.right_field]


def test_matcher_falls_back_to_the_docstring_when_no_program_was_trained(tmp_path: Any) -> None:
    """Asking for a prompt that was never trained must not fail the run."""
    matcher = DatasetMatcher("abt-buy", trained_prompts=True, trained_dir=str(tmp_path))

    assert matcher.predictor.signature is get_dataset_spec("abt-buy").signature


def test_generic_mode_constant_is_the_default() -> None:
    """The generic mode is the documented default for the CLI."""
    assert SIGNATURE_MODE_GENERIC == "generic"


def test_collect_pairs_counts_blocks_that_fell_back_to_error_recovery() -> None:
    """A failed block is counted so a degraded run is not read as a real one."""
    matcher = DatasetMatcher("walmart-amazon")
    good_block = _block([_walmart_entity(1), _amazon_entity(RIGHT_ID_OFFSET + 2)])
    bad_block = _block([_walmart_entity(3), _amazon_entity(RIGHT_ID_OFFSET + 4)])

    matcher._predictor = _StubPredict([_candidate(0, 1)])  # type: ignore[assignment]
    with patch.object(EntityMatcher, "_ensure_lm", return_value=None):
        good = matcher.resolve_block(good_block)

    def _raise(**_kwargs: Any) -> dspy.Prediction:
        raise RuntimeError("endpoint exploded")

    matcher._predictor = _raise  # type: ignore[assignment]
    with patch.object(EntityMatcher, "_ensure_lm", return_value=None):
        bad = matcher.resolve_block(bad_block)

    outcome = collect_pairs([good, bad])

    assert outcome.failed_blocks == 1
    assert outcome.predicted_pairs == {(1, RIGHT_ID_OFFSET + 2)}


def test_collect_pairs_reports_no_failures_for_healthy_blocks() -> None:
    """A run where every block answered reports zero failed blocks."""
    matcher = DatasetMatcher("walmart-amazon")
    block = _block([_walmart_entity(1), _amazon_entity(RIGHT_ID_OFFSET + 2)])
    matcher._predictor = _StubPredict([_candidate(0, 1)])  # type: ignore[assignment]

    with patch.object(EntityMatcher, "_ensure_lm", return_value=None):
        resolution = matcher.resolve_block(block)

    assert collect_pairs([resolution]).failed_blocks == 0
