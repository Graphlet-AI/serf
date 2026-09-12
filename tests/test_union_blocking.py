"""Tests for union blocking, where the name and JSON views are both kept.

JSON blocking loses to name blocking on its own almost everywhere, but that
only rules it out as a *replacement*. As an augmentation the question is
different: does a pair caught by either view get caught, and what does keeping
both cost the matcher?
"""

from unittest.mock import MagicMock, patch

import numpy as np

from serf.block.pipeline import SemanticBlockingPipeline, count_blocked_pairs
from serf.dspy.types import BlockingMetrics, Entity, EntityBlock
from serf.eval.blocking_sweep import block_membership, evaluate_blocking


def _entity(entity_id: int, name: str, brand: str = "") -> Entity:
    """Build an entity carrying one extra attribute for the JSON view."""
    return Entity(
        id=entity_id,
        name=name,
        entity_type="product",
        attributes={"l_name": name, "l_brand": brand} if brand else {"l_name": name},
    )


def _block(key: str, entities: list[Entity]) -> EntityBlock:
    """Build a block holding the given entities."""
    return EntityBlock(
        block_key=key,
        block_key_type="semantic",
        block_size=len(entities),
        entities=entities,
    )


def _run_union(
    entities: list[Entity], assignments: list[dict[str, list[str]]]
) -> tuple[list[EntityBlock], BlockingMetrics]:
    """Run the pipeline in union mode against stubbed clustering."""
    clusters = iter(assignments)
    with (
        patch(
            "serf.block.pipeline.embed_in_subprocess",
            side_effect=lambda texts, **_: np.zeros((len(texts), 4), dtype=np.float32),
        ),
        patch(
            "serf.block.pipeline.cluster_in_subprocess",
            side_effect=lambda *_args, **_kwargs: next(clusters),
        ),
    ):
        pipeline = SemanticBlockingPipeline(
            model_name="test-model", auto_scale=False, blocking_strategy="union"
        )
        return pipeline.run(entities)


def test_union_embeds_the_name_and_the_json_record() -> None:
    """Union blocking runs the embedder twice, once per view."""
    entities = [_entity(1, "acme widget", "acme"), _entity(2, "globex gadget", "globex")]
    seen: list[list[str]] = []

    clusters = iter([{"block_0": ["1", "2"]}, {"block_0": ["1", "2"]}])
    with (
        patch(
            "serf.block.pipeline.embed_in_subprocess",
            side_effect=lambda texts, **_: (
                seen.append(list(texts)) or np.zeros((len(texts), 4), dtype=np.float32)
            ),
        ),
        patch(
            "serf.block.pipeline.cluster_in_subprocess",
            side_effect=lambda *_args, **_kwargs: next(clusters),
        ),
    ):
        SemanticBlockingPipeline(
            model_name="test-model", auto_scale=False, blocking_strategy="union"
        ).run(entities)

    assert seen[0] == ["acme widget", "globex gadget"]
    assert seen[1] == [
        '{"brand": "acme", "name": "acme widget"}',
        '{"brand": "globex", "name": "globex gadget"}',
    ]


def test_union_block_keys_name_the_view_they_came_from() -> None:
    """Keys stay distinct so a block is traceable to the view that built it."""
    entities = [_entity(1, "a", "x"), _entity(2, "b", "y")]
    blocks, _ = _run_union(
        entities, [{"block_0": ["1"], "block_1": ["2"]}, {"block_0": ["1", "2"]}]
    )

    assert [block.block_key for block in blocks] == ["name_block_0", "name_block_1", "json_block_0"]


def test_a_single_view_leaves_block_keys_alone() -> None:
    """Only the union namespaces its keys, so existing keys do not move."""
    with (
        patch(
            "serf.block.pipeline.embed_in_subprocess",
            return_value=np.zeros((1, 4), dtype=np.float32),
        ),
        patch(
            "serf.block.pipeline.cluster_in_subprocess",
            return_value={"block_0": ["1"]},
        ),
    ):
        blocks, _ = SemanticBlockingPipeline(model_name="m", auto_scale=False).run(
            [_entity(1, "a")]
        )

    assert blocks[0].block_key == "block_0"


def test_blocked_pairs_are_billed_once_across_views() -> None:
    """A pair both views block is one comparison, not two."""
    entities = [_entity(1, "a"), _entity(2, "b"), _entity(3, "c")]
    # The name view groups 1+2, the JSON view groups 1+2 again and adds 1+3.
    blocks, metrics = _run_union(
        entities, [{"block_0": ["1", "2"], "block_1": ["3"]}, {"block_0": ["1", "2", "3"]}]
    )

    assert metrics.total_blocks == 3
    assert metrics.blocked_pairs == 3
    assert count_blocked_pairs(blocks) == 3


def test_blocked_pairs_sum_when_blocks_are_disjoint() -> None:
    """Disjoint blocks take the cheap arithmetic path and get the same answer."""
    blocks = [
        _block("block_0", [_entity(1, "a"), _entity(2, "b"), _entity(3, "c")]),
        _block("block_1", [_entity(4, "d"), _entity(5, "e")]),
    ]

    assert count_blocked_pairs(blocks) == 3 + 1


def test_membership_is_a_set_of_blocks_per_record() -> None:
    """A record blocked under two views belongs to two blocks."""
    shared = _entity(1, "a")
    blocks = [
        _block("name_0", [shared, _entity(2, "b")]),
        _block("json_0", [shared, _entity(3, "c")]),
    ]

    membership = block_membership(blocks)

    assert membership[1] == {0, 1}
    assert membership[2] == {0}
    assert membership[3] == {1}


def test_recall_counts_a_pair_caught_by_either_view() -> None:
    """Overlapping blocks mean co-blocked is "share any block", not "share the block"."""
    a, b, c = _entity(1, "a"), _entity(2, "b"), _entity(3, "c")
    blocks = [_block("name_0", [a, b]), _block("json_0", [a, c])]

    pipeline = MagicMock()
    pipeline.run.return_value = (blocks, BlockingMetrics(total_blocks=2, blocked_pairs=2))

    with patch("serf.eval.blocking_sweep.SemanticBlockingPipeline", return_value=pipeline):
        result = evaluate_blocking(
            entities=[a, b, c],
            ground_truth={(1, 2), (1, 3), (2, 3)},
            dataset="test",
            model_name="test-model",
            prompt="",
            target_block_size=30,
            max_block_size=100,
            strategy="union",
        )

    assert result.co_blocked == 2
    assert result.blocking_recall == 2 / 3
    assert result.blocked_pairs == 2
    assert result.strategy == "union"
