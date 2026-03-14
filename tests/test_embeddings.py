"""Tests for entity embedding.

Note: These tests use a small model for speed. The default Qwen3-Embedding-0.6B
is too large for unit tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from serf.block.embeddings import EntityEmbedder, get_torch_device


def test_get_torch_device() -> None:
    """Test torch device detection returns a valid device string."""
    device = get_torch_device()
    assert device in ("cuda", "mps", "cpu")


@patch("serf.block.embeddings.SentenceTransformer")
def test_entity_embedder_init(mock_st: MagicMock) -> None:
    """Test EntityEmbedder initialization with mock model."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_st.return_value = mock_model

    embedder = EntityEmbedder(model_name="test-model", device="cpu")
    assert embedder.model_name == "test-model"
    assert embedder.device == "cpu"
    assert embedder.embedding_dim == 384


@patch("serf.block.embeddings.SentenceTransformer")
def test_entity_embedder_embed(mock_st: MagicMock) -> None:
    """Test EntityEmbedder.embed returns correct shape."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
    mock_st.return_value = mock_model

    embedder = EntityEmbedder(model_name="test-model", device="cpu")
    result = embedder.embed(["text1", "text2", "text3"])

    assert result.shape == (3, 384)
    assert result.dtype == np.float32
    mock_model.encode.assert_called_once()
