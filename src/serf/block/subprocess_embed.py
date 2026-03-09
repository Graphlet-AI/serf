"""Subprocess-isolated embedding and FAISS clustering.

Runs PyTorch embedding and FAISS clustering in a separate subprocess
to avoid memory conflicts between PyTorch MPS and FAISS on macOS.
This is the pattern proven in the Abzu production system.

The main process communicates with the subprocess via temporary files
(numpy .npy for embeddings, JSON for block assignments).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from serf.logs import get_logger

logger = get_logger(__name__)

# Inline Python script for embedding — runs in a fresh subprocess
EMBED_SCRIPT = """
import json
import sys
import numpy as np

def main():
    args = json.loads(sys.argv[1])
    texts_file = args["texts_file"]
    output_file = args["output_file"]
    model_name = args["model_name"]

    with open(texts_file) as f:
        texts = json.load(f)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    np.save(output_file, np.ascontiguousarray(embeddings, dtype=np.float32))

if __name__ == "__main__":
    main()
"""

# Inline Python script for FAISS clustering — runs in a fresh subprocess
FAISS_SCRIPT = """
import json
import math
import sys
import numpy as np

def main():
    args = json.loads(sys.argv[1])
    embeddings_file = args["embeddings_file"]
    output_file = args["output_file"]
    ids = args["ids"]
    target_block_size = args["target_block_size"]

    import faiss

    embeddings = np.load(embeddings_file)
    n, dim = embeddings.shape

    if n == 0:
        with open(output_file, "w") as f:
            json.dump({}, f)
        return

    if n <= target_block_size:
        with open(output_file, "w") as f:
            json.dump({"block_0": ids}, f)
        return

    nlist = max(1, n // target_block_size)
    nlist = min(nlist, int(math.sqrt(n)))
    nlist = max(1, nlist)

    faiss.normalize_L2(embeddings)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)

    _, assignments = index.quantizer.search(embeddings, 1)

    blocks = {}
    for i, cluster_id in enumerate(assignments.flatten()):
        block_key = f"block_{int(cluster_id)}"
        if block_key not in blocks:
            blocks[block_key] = []
        blocks[block_key].append(ids[i])

    with open(output_file, "w") as f:
        json.dump(blocks, f)

if __name__ == "__main__":
    main()
"""


def embed_in_subprocess(
    texts: list[str],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
) -> NDArray[np.float32]:
    """Compute embeddings in an isolated subprocess.

    Avoids PyTorch MPS / FAISS memory conflicts on macOS by running
    the sentence-transformer model in a separate process.

    Parameters
    ----------
    texts : list[str]
        Texts to embed
    model_name : str
        HuggingFace model name

    Returns
    -------
    NDArray[np.float32]
        Embeddings matrix (n, dim)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        texts_file = str(Path(tmpdir) / "texts.json")
        output_file = str(Path(tmpdir) / "embeddings.npy")

        with open(texts_file, "w") as f:
            json.dump(texts, f)

        args = json.dumps(
            {
                "texts_file": texts_file,
                "output_file": output_file,
                "model_name": model_name,
            }
        )

        logger.info(f"Embedding {len(texts)} texts in subprocess (model={model_name})")
        result = subprocess.run(
            [sys.executable, "-c", EMBED_SCRIPT, args],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Embedding subprocess failed:\n{result.stderr}")
            raise RuntimeError(f"Embedding subprocess failed: {result.stderr[:500]}")

        embeddings: NDArray[np.float32] = np.load(output_file)
        logger.info(f"Embeddings computed: shape={embeddings.shape}")
        return embeddings


def cluster_in_subprocess(
    embeddings: NDArray[np.float32],
    ids: list[str],
    target_block_size: int = 30,
) -> dict[str, list[str]]:
    """Cluster embeddings using FAISS in an isolated subprocess.

    Avoids FAISS segfaults caused by MPS memory conflicts on macOS.

    Parameters
    ----------
    embeddings : NDArray[np.float32]
        Embedding matrix (n, dim)
    ids : list[str]
        Entity IDs corresponding to embedding rows
    target_block_size : int
        Target entities per cluster

    Returns
    -------
    dict[str, list[str]]
        Mapping from block_key to list of entity IDs
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings_file = str(Path(tmpdir) / "embeddings.npy")
        output_file = str(Path(tmpdir) / "blocks.json")

        np.save(embeddings_file, embeddings)

        args = json.dumps(
            {
                "embeddings_file": embeddings_file,
                "output_file": output_file,
                "ids": ids,
                "target_block_size": target_block_size,
            }
        )

        logger.info(f"Clustering {len(ids)} entities in subprocess (target={target_block_size})")
        result = subprocess.run(
            [sys.executable, "-c", FAISS_SCRIPT, args],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"FAISS subprocess failed:\n{result.stderr}")
            raise RuntimeError(f"FAISS subprocess failed: {result.stderr[:500]}")

        with open(output_file) as f:
            blocks: dict[str, list[str]] = json.load(f)

        logger.info(f"Created {len(blocks)} blocks")
        return blocks
