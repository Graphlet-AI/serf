"""Tests for the embedding blocking sweep."""

from unittest.mock import MagicMock, patch

from serf.dspy.types import BlockingMetrics, Entity, EntityBlock
from serf.eval.blocking_sweep import (
    BlockingSweepResult,
    evaluate_blocking,
    evaluate_blocking_rounds,
)


def _entity(entity_id: int, name: str) -> Entity:
    """Build a minimal entity."""
    return Entity(id=entity_id, name=name, entity_type="product")


def _block(entities: list[Entity]) -> EntityBlock:
    """Build a block holding the given entities."""
    return EntityBlock(
        block_key=f"block_{entities[0].id}",
        block_key_type="semantic",
        block_size=len(entities),
        entities=entities,
    )


def _run_evaluate(
    blocks: list[EntityBlock], ground_truth: set[tuple[int, int]]
) -> BlockingSweepResult:
    """Run evaluate_blocking against a pipeline stubbed to return ``blocks``."""
    entities = [e for block in blocks for e in block.entities]
    pipeline = MagicMock()
    pipeline.run.return_value = (blocks, BlockingMetrics(total_blocks=len(blocks)))

    with patch("serf.eval.blocking_sweep.SemanticBlockingPipeline", return_value=pipeline):
        return evaluate_blocking(
            entities=entities,
            ground_truth=ground_truth,
            dataset="test",
            model_name="test-model",
            prompt="query: ",
            target_block_size=30,
            max_block_size=100,
        )


def test_co_blocked_pairs_are_counted() -> None:
    """A gold pair inside one block counts toward recall."""
    blocks = [_block([_entity(1, "acme widget"), _entity(2, "acme widget pro")])]
    result = _run_evaluate(blocks, {(1, 2)})

    assert result.co_blocked == 1
    assert result.blocking_recall == 1.0
    assert result.gold_pairs == 1


def test_split_pairs_are_missed() -> None:
    """A gold pair straddling two blocks is a blocking miss."""
    blocks = [_block([_entity(1, "acme widget")]), _block([_entity(2, "acme widget pro")])]
    result = _run_evaluate(blocks, {(1, 2)})

    assert result.co_blocked == 0
    assert result.blocking_recall == 0.0


def test_recall_is_the_co_blocked_share() -> None:
    """Recall is the co-blocked share of gold pairs, not of records."""
    blocks = [
        _block([_entity(1, "a"), _entity(2, "a pro")]),
        _block([_entity(3, "b"), _entity(4, "c")]),
    ]
    result = _run_evaluate(blocks, {(1, 2), (3, 5), (4, 6)})

    assert result.co_blocked == 1
    assert result.blocking_recall == 1 / 3


def test_empty_ground_truth_scores_zero() -> None:
    """No gold pairs yields zero rather than a division error."""
    blocks = [_block([_entity(1, "a"), _entity(2, "b")])]
    result = _run_evaluate(blocks, set())

    assert result.gold_pairs == 0
    assert result.blocking_recall == 0.0


def test_rounds_stop_early_when_nothing_merges() -> None:
    """A round that co-blocks no gold pair ends the loop."""
    entities = [_entity(1, "a"), _entity(2, "b")]
    pipeline = MagicMock()
    pipeline.run.return_value = (
        [_block([entities[0]]), _block([entities[1]])],
        BlockingMetrics(total_blocks=2),
    )

    with patch("serf.eval.blocking_sweep.SemanticBlockingPipeline", return_value=pipeline):
        results = evaluate_blocking_rounds(
            entities=entities,
            ground_truth={(1, 2)},
            dataset="test",
            model_name="test-model",
            prompt="",
            target_block_size=30,
            max_block_size=100,
            rounds=3,
        )

    assert len(results) == 1
    assert results[0].cumulative_recall == 0.0


def test_rounds_accumulate_recall_across_rounds() -> None:
    """A pair separated in round one but co-blocked later counts as recovered."""
    a, b, c, d = _entity(1, "a"), _entity(2, "b"), _entity(3, "c"), _entity(4, "d")
    first = ([_block([a, b]), _block([c, d])], BlockingMetrics(total_blocks=2))
    second = ([_block([a, c])], BlockingMetrics(total_blocks=1))

    pipeline = MagicMock()
    pipeline.run.side_effect = [first, second]

    with patch("serf.eval.blocking_sweep.SemanticBlockingPipeline", return_value=pipeline):
        results = evaluate_blocking_rounds(
            entities=[a, b, c, d],
            ground_truth={(1, 2), (1, 3)},
            dataset="test",
            model_name="test-model",
            prompt="",
            target_block_size=30,
            max_block_size=100,
            rounds=2,
        )

    assert results[0].cumulative_recall == 0.5
    assert results[1].cumulative_recall == 1.0
    assert results[1].round_number == 2


def test_prompt_is_passed_to_the_pipeline() -> None:
    """The instruction prefix reaches the blocking pipeline."""
    entities = [_entity(1, "a"), _entity(2, "b")]
    pipeline = MagicMock()
    pipeline.run.return_value = ([_block(entities)], BlockingMetrics(total_blocks=1))

    with patch(
        "serf.eval.blocking_sweep.SemanticBlockingPipeline", return_value=pipeline
    ) as mock_pipeline:
        evaluate_blocking(
            entities=entities,
            ground_truth={(1, 2)},
            dataset="test",
            model_name="test-model",
            prompt="query: ",
            target_block_size=30,
            max_block_size=100,
        )

    assert mock_pipeline.call_args.kwargs["embedding_prompt"] == "query: "
    assert mock_pipeline.call_args.kwargs["model_name"] == "test-model"
