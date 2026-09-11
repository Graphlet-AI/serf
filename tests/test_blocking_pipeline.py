"""Tests for the semantic blocking pipeline."""

from unittest.mock import patch

import numpy as np

from serf.block.pipeline import SemanticBlockingPipeline
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
