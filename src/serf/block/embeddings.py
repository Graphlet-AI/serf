"""Entity embedder using sentence-transformers.

Wraps sentence-transformers models for computing entity embeddings
used in semantic blocking.
"""

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from serf.config import config
from serf.logs import get_logger

logger = get_logger(__name__)


def get_torch_device() -> str:
    """Auto-detect the best available torch device.

    Returns
    -------
    str
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EntityEmbedder:
    """Compute embeddings for entity text using sentence-transformers.

    Parameters
    ----------
    model_name : str | None
        Hugging Face model name. Defaults to config models.embedding.
    device : str | None
        Torch device. Auto-detected if None.
    normalize : bool
        Whether to L2-normalize embeddings.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        if model_name is None:
            model_name = config.get("models.embedding", "intfloat/multilingual-e5-large")
        if device is None:
            device = get_torch_device()

        self.model_name = model_name
        self.device = device
        self.normalize = normalize

        logger.info(f"Loading embedding model {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def embed(self, texts: list[str], batch_size: int = 64) -> NDArray[np.float32]:
        """Compute embeddings for a list of texts.

        Parameters
        ----------
        texts : list[str]
            Texts to embed
        batch_size : int
            Batch size for encoding

        Returns
        -------
        NDArray[np.float32]
            Embeddings matrix of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            device="cpu",  # Always encode on CPU — FAISS segfaults with MPS tensors
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)
