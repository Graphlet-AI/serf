"""FAISS-based blocking for entity resolution.

Uses FAISS IndexIVFFlat to cluster entity embeddings into blocks
for efficient pairwise comparison.
"""

import math

import faiss
import numpy as np
from numpy.typing import NDArray

from serf.logs import get_logger

logger = get_logger(__name__)


class FAISSBlocker:
    """Cluster entity embeddings into blocks using FAISS IVF.

    Parameters
    ----------
    target_block_size : int
        Target number of entities per block
    max_distance : float | None
        Maximum inner product distance for inclusion in a block
    iteration : int
        Current ER iteration (used for auto-scaling)
    auto_scale : bool
        Whether to auto-scale target_block_size by iteration
    """

    def __init__(
        self,
        target_block_size: int = 50,
        max_distance: float | None = None,
        iteration: int = 1,
        auto_scale: bool = True,
    ) -> None:
        self.target_block_size = target_block_size
        self.max_distance = max_distance
        self.iteration = iteration
        self.auto_scale = auto_scale

        # Auto-scale: tighter clusters in later rounds
        if auto_scale and iteration > 1:
            self.effective_target = max(10, target_block_size // iteration)
        else:
            self.effective_target = target_block_size

        logger.info(
            f"FAISSBlocker: target={target_block_size}, "
            f"effective={self.effective_target}, iteration={iteration}"
        )

    def block(
        self,
        embeddings: NDArray[np.float32],
        ids: list[str],
    ) -> dict[str, list[str]]:
        """Cluster embeddings into blocks.

        Parameters
        ----------
        embeddings : NDArray[np.float32]
            Embedding matrix of shape (n, dim)
        ids : list[str]
            Entity identifiers corresponding to each embedding row

        Returns
        -------
        dict[str, list[str]]
            Mapping from block_key to list of entity IDs in that block
        """
        n = len(ids)
        dim = embeddings.shape[1]

        if n == 0:
            return {}

        if n <= self.effective_target:
            # Everything fits in one block
            return {"block_0": list(ids)}

        # Calculate number of clusters
        nlist = max(1, n // self.effective_target)
        nlist = min(nlist, int(math.sqrt(n)))
        nlist = max(1, nlist)

        logger.info(f"FAISS: n={n}, dim={dim}, nlist={nlist}")

        # Normalize embeddings for inner product
        faiss.normalize_L2(embeddings)

        # Build IVF index
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train and add vectors
        index.train(embeddings)  # type: ignore[call-arg]
        index.add(embeddings)  # type: ignore[call-arg]

        # Assign each vector to its nearest centroid
        _, assignments = index.quantizer.search(embeddings, 1)

        # Build blocks from assignments
        blocks: dict[str, list[str]] = {}
        for i, cluster_id in enumerate(assignments.flatten()):
            block_key = f"block_{int(cluster_id)}"
            if block_key not in blocks:
                blocks[block_key] = []
            blocks[block_key].append(ids[i])

        logger.info(
            f"Created {len(blocks)} blocks, "
            f"sizes: min={min(len(v) for v in blocks.values())}, "
            f"max={max(len(v) for v in blocks.values())}, "
            f"avg={n / len(blocks):.1f}"
        )

        return blocks
