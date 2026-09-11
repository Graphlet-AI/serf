"""Tests for the semantic blocking pipeline."""

from unittest.mock import patch

import numpy as np

from serf.block.pipeline import SemanticBlockingPipeline
from serf.block.subprocess_embed import cluster_in_subprocess
from serf.dspy.types import Entity


def _entities() -> list[Entity]:
    """Build two entities to block."""
    return [
        Entity(id=1, name="acme widget", entity_type="product"),
        Entity(id=2, name="globex gadget", entity_type="product"),
    ]


def test_pipeline_passes_prompt_to_the_embedder() -> None:
    """The configured instruction prefix reaches the embedding subprocess."""
    entities = _entities()

    with (
        patch(
            "serf.block.pipeline.embed_in_subprocess",
            return_value=np.zeros((2, 4), dtype=np.float32),
        ) as mock_embed,
        patch(
            "serf.block.pipeline.cluster_in_subprocess",
            return_value={"block_0": ["1", "2"]},
        ),
    ):
        pipeline = SemanticBlockingPipeline(
            model_name="test-model", embedding_prompt="query: ", auto_scale=False
        )
        pipeline.run(entities)

    assert mock_embed.call_args.kwargs["prompt"] == "query: "
    assert mock_embed.call_args.kwargs["model_name"] == "test-model"


def test_pipeline_prompt_defaults_to_config() -> None:
    """Omitting the prefix falls back to the config value rather than None."""
    pipeline = SemanticBlockingPipeline(model_name="test-model")
    assert isinstance(pipeline.embedding_prompt, str)


def test_clustering_handles_more_ids_than_argv_allows() -> None:
    """Entity ids go through a file, so the 128 KB argv cap does not bind."""
    count = 20000
    ids = [str(i) for i in range(count)]
    embeddings = np.random.randn(count, 8).astype(np.float32)

    blocks = cluster_in_subprocess(embeddings, ids, target_block_size=30)

    assert sum(len(members) for members in blocks.values()) == count


def test_cluster_count_tracks_the_target_block_size() -> None:
    """Cluster count follows n / target rather than a sqrt(n) cap.

    A sqrt(n) cap made target_block_size unreachable above n = target squared,
    leaving blocks an order of magnitude larger than asked for.
    """
    count = 20000
    target = 30
    ids = [str(i) for i in range(count)]
    embeddings = np.random.randn(count, 8).astype(np.float32)

    blocks = cluster_in_subprocess(embeddings, ids, target_block_size=target)

    assert len(blocks) > count**0.5 * 2
    assert count / len(blocks) < target * 2


def test_pipeline_embeds_names_only() -> None:
    """Blocking text is the bare entity name when no blocking fields are set."""
    entities = _entities()

    with (
        patch(
            "serf.block.pipeline.embed_in_subprocess",
            return_value=np.zeros((2, 4), dtype=np.float32),
        ) as mock_embed,
        patch(
            "serf.block.pipeline.cluster_in_subprocess",
            return_value={"block_0": ["1", "2"]},
        ),
    ):
        SemanticBlockingPipeline(model_name="test-model", auto_scale=False).run(entities)

    assert mock_embed.call_args.args[0] == ["acme widget", "globex gadget"]


def test_pipeline_json_strategy_embeds_every_field() -> None:
    """The json strategy sends the full record, not just the name."""
    entities = [
        Entity(
            id=1,
            name="acme widget",
            entity_type="product",
            attributes={"l_id": "7", "l_title": "acme widget", "l_brand": "acme"},
        )
    ]

    with (
        patch(
            "serf.block.pipeline.embed_in_subprocess",
            return_value=np.zeros((1, 4), dtype=np.float32),
        ) as mock_embed,
        patch(
            "serf.block.pipeline.cluster_in_subprocess",
            return_value={"block_0": ["1"]},
        ),
    ):
        SemanticBlockingPipeline(
            model_name="test-model", auto_scale=False, blocking_strategy="json"
        ).run(entities)

    assert mock_embed.call_args.args[0] == ['{"brand": "acme", "title": "acme widget"}']


def test_pipeline_strategy_defaults_to_config() -> None:
    """Omitting the strategy falls back to the configured one."""
    pipeline = SemanticBlockingPipeline(model_name="test-model")
    assert pipeline.blocking_strategy in {"name", "json", "union"}


def test_pipeline_passes_trust_remote_code_to_the_embedder() -> None:
    """Models with custom architectures can opt into running their own code."""
    with (
        patch(
            "serf.block.pipeline.embed_in_subprocess",
            return_value=np.zeros((2, 4), dtype=np.float32),
        ) as mock_embed,
        patch(
            "serf.block.pipeline.cluster_in_subprocess",
            return_value={"block_0": ["1", "2"]},
        ),
    ):
        SemanticBlockingPipeline(
            model_name="test-model", auto_scale=False, embedding_trust_remote_code=True
        ).run(_entities())

    assert mock_embed.call_args.kwargs["trust_remote_code"] is True
