"""Tests for FAISS-based blocking."""

import numpy as np

from serf.block.faiss_blocker import FAISSBlocker


def test_blocker_single_block() -> None:
    """Test that small datasets get a single block."""
    blocker = FAISSBlocker(target_block_size=100)
    embeddings = np.random.randn(10, 64).astype(np.float32)
    ids = [f"id_{i}" for i in range(10)]

    blocks = blocker.block(embeddings, ids)
    # With only 10 items and target 100, should be 1 block
    assert len(blocks) == 1
    assert len(list(blocks.values())[0]) == 10


def test_blocker_multiple_blocks() -> None:
    """Test creating multiple blocks from a larger dataset."""
    blocker = FAISSBlocker(target_block_size=10)
    # Create 100 entities with 64-dim embeddings
    embeddings = np.random.randn(100, 64).astype(np.float32)
    ids = [f"id_{i}" for i in range(100)]

    blocks = blocker.block(embeddings, ids)
    # Should create multiple blocks
    assert len(blocks) > 1
    # All entities should be assigned
    all_ids = []
    for block_ids in blocks.values():
        all_ids.extend(block_ids)
    assert sorted(all_ids) == sorted(ids)


def test_blocker_empty_input() -> None:
    """Test blocking with empty input."""
    blocker = FAISSBlocker()
    embeddings = np.zeros((0, 64), dtype=np.float32)
    blocks = blocker.block(embeddings, [])
    assert blocks == {}


def test_blocker_auto_scale() -> None:
    """Test auto-scaling target block size by iteration."""
    blocker_iter1 = FAISSBlocker(target_block_size=50, iteration=1)
    blocker_iter3 = FAISSBlocker(target_block_size=50, iteration=3)

    assert blocker_iter1.effective_target == 50
    assert blocker_iter3.effective_target == 16  # 50 // 3 = 16


def test_blocker_no_auto_scale() -> None:
    """Test disabling auto-scaling."""
    blocker = FAISSBlocker(target_block_size=50, iteration=3, auto_scale=False)
    assert blocker.effective_target == 50


def test_blocker_min_target_size() -> None:
    """Test that effective target doesn't go below 10."""
    blocker = FAISSBlocker(target_block_size=20, iteration=10)
    assert blocker.effective_target == 10  # max(10, 20 // 10) = 10


def test_blocker_preserves_all_ids() -> None:
    """Test that all input IDs appear in output blocks."""
    blocker = FAISSBlocker(target_block_size=5)
    n = 50
    embeddings = np.random.randn(n, 32).astype(np.float32)
    ids = [f"entity_{i}" for i in range(n)]

    blocks = blocker.block(embeddings, ids)
    output_ids = set()
    for block_ids in blocks.values():
        output_ids.update(block_ids)
    assert output_ids == set(ids)


def test_blocker_block_keys_format() -> None:
    """Test that block keys have expected format."""
    blocker = FAISSBlocker(target_block_size=10)
    embeddings = np.random.randn(50, 32).astype(np.float32)
    ids = [f"id_{i}" for i in range(50)]

    blocks = blocker.block(embeddings, ids)
    for key in blocks:
        assert key.startswith("block_")
