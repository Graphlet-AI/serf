"""Tests for blocked train/val/holdout sampling."""

from serf.config import config
from serf.dspy.types import Entity, EntityBlock
from serf.eval.splits import (
    get_all_split_sizes,
    get_split_sizes,
    sample_blocked_splits,
)


def _entity(entity_id: int) -> Entity:
    """Build a minimal entity."""
    return Entity(id=entity_id, name=f"e{entity_id}", description="", entity_type="entity")


def _block(key: str, ids: list[int]) -> EntityBlock:
    """Build a block from entity IDs."""
    entities = [_entity(i) for i in ids]
    return EntityBlock(block_key=key, block_size=len(entities), entities=entities)


def _blocks(count: int, size: int) -> list[EntityBlock]:
    """Build ``count`` disjoint blocks of ``size`` entities each."""
    return [_block(str(i), list(range(i * size, i * size + size))) for i in range(count)]


def _split_ids(blocks: list[EntityBlock]) -> set[int]:
    """Collect entity IDs across blocks."""
    return {entity.id for block in blocks for entity in block.entities}


def test_split_sizes_are_1k_train_blocks_per_dataset() -> None:
    """Every benchmark dataset uses 1000 train blocks, 500 val, 1000 holdout."""
    expected = get_all_split_sizes()
    assert set(expected) == {
        "walmart-amazon",
        "abt-buy",
        "amazon-google",
        "dblp-acm",
        "dblp-scholar",
    }
    for name, sizes in expected.items():
        assert sizes.train_blocks == 1000, name
        assert sizes.val_records == 500, name
        assert sizes.holdout_records == 1000, name
        assert get_split_sizes(name) == sizes
    assert config.get("benchmarks.train_blocks") == 1000
    assert config.get("benchmarks.val_records") == 500
    assert config.get("benchmarks.holdout_records") == 1000


def test_sample_blocked_splits_is_disjoint_and_sized() -> None:
    """Train, val, and holdout blocks do not share entities."""
    blocks = _blocks(20, 5)
    splits = sample_blocked_splits(
        blocks,
        train_blocks=4,
        val_records=10,
        holdout_records=20,
        seed=0,
        min_block_size=2,
    )
    assert len(splits.train_blocks) == 4
    assert len(splits.val_blocks) == 2
    assert len(splits.holdout_blocks) == 4

    train_ids = _split_ids(splits.train_blocks)
    val_ids = _split_ids(splits.val_blocks)
    holdout_ids = _split_ids(splits.holdout_blocks)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(holdout_ids)
    assert val_ids.isdisjoint(holdout_ids)
    assert len(train_ids) == 20


def test_sample_blocked_splits_reserves_val_and_holdout_when_blocks_are_scarce() -> None:
    """DBLP-ACM case: 82 blocks cover every record, yet val and holdout are real blocks."""
    blocks = _blocks(82, 60)
    splits = sample_blocked_splits(
        blocks,
        train_blocks=1000,
        val_records=500,
        holdout_records=1000,
        seed=42,
        min_block_size=2,
    )
    assert len(splits.val_blocks) == 9
    assert len(splits.holdout_blocks) == 17
    assert len(splits.train_blocks) == 82 - 9 - 17

    train_ids = _split_ids(splits.train_blocks)
    val_ids = _split_ids(splits.val_blocks)
    holdout_ids = _split_ids(splits.holdout_blocks)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(holdout_ids)
    assert val_ids.isdisjoint(holdout_ids)
    assert len(train_ids) + len(val_ids) + len(holdout_ids) == 82 * 60

    assert all(block.block_size == 60 for block in splits.val_blocks)
    assert all(block.block_key_type == "semantic" for block in splits.val_blocks)


def test_sample_blocked_splits_caps_train_blocks() -> None:
    """Train stops at train_blocks even when more blocks are left over."""
    blocks = _blocks(20, 4)
    splits = sample_blocked_splits(
        blocks,
        train_blocks=3,
        val_records=4,
        holdout_records=4,
        seed=1,
        min_block_size=2,
    )
    assert len(splits.val_blocks) == 1
    assert len(splits.holdout_blocks) == 1
    assert len(splits.train_blocks) == 3


def test_sample_blocked_splits_fills_val_before_holdout_and_train() -> None:
    """Val takes whole blocks until its record budget is met, even with few blocks."""
    blocks = _blocks(2, 3)
    splits = sample_blocked_splits(
        blocks,
        train_blocks=1000,
        val_records=500,
        holdout_records=1000,
        seed=1,
        min_block_size=2,
    )
    assert len(splits.val_blocks) == 2
    assert splits.holdout_blocks == []
    assert splits.train_blocks == []


def test_sample_blocked_splits_skips_singletons() -> None:
    """Sampling ignores blocks smaller than min_block_size."""
    blocks = [_block("pair", [0, 1]), _block("s1", [2]), _block("s2", [3])]
    splits = sample_blocked_splits(
        blocks,
        train_blocks=10,
        val_records=2,
        holdout_records=2,
        seed=2,
        min_block_size=2,
    )
    assert len(splits.val_blocks) == 1
    assert splits.val_blocks[0].block_key == "pair"
    assert splits.holdout_blocks == []
    assert splits.train_blocks == []


def test_sample_blocked_splits_is_deterministic() -> None:
    """The same seed yields the same split."""
    blocks = _blocks(20, 2)
    a = sample_blocked_splits(blocks, train_blocks=5, val_records=4, holdout_records=6, seed=7)
    b = sample_blocked_splits(blocks, train_blocks=5, val_records=4, holdout_records=6, seed=7)
    assert [blk.block_key for blk in a.train_blocks] == [blk.block_key for blk in b.train_blocks]
    assert [blk.block_key for blk in a.val_blocks] == [blk.block_key for blk in b.val_blocks]
    assert [blk.block_key for blk in a.holdout_blocks] == [
        blk.block_key for blk in b.holdout_blocks
    ]


def test_sample_blocked_splits_reports_record_counts() -> None:
    """Split record counts sum to the entities in the returned blocks."""
    blocks = _blocks(10, 5)
    splits = sample_blocked_splits(
        blocks,
        train_blocks=100,
        val_records=10,
        holdout_records=10,
        seed=5,
        min_block_size=2,
    )
    assert splits.val_record_count == sum(b.block_size for b in splits.val_blocks)
    assert splits.holdout_record_count == sum(b.block_size for b in splits.holdout_blocks)
    assert splits.train_record_count == sum(b.block_size for b in splits.train_blocks)
