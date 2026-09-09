"""Tests for blocked train/val/holdout sampling."""

from serf.config import config
from serf.dspy.types import Entity, EntityBlock
from serf.eval.splits import (
    chunk_records,
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
    """Train blocks, val records, and holdout records do not overlap."""
    entities = [_entity(i) for i in range(100)]
    blocks = [_block(str(i), list(range(i * 5, i * 5 + 5))) for i in range(20)]
    splits = sample_blocked_splits(
        entities,
        blocks,
        train_blocks=4,
        val_records=10,
        holdout_records=20,
        seed=0,
        min_block_size=2,
    )
    assert len(splits.train_blocks) == 4
    assert len(splits.val_records) == 10
    assert len(splits.holdout_records) == 20

    train_ids = {e.id for b in splits.train_blocks for e in b.entities}
    val_ids = {e.id for e in splits.val_records}
    holdout_ids = {e.id for e in splits.holdout_records}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(holdout_ids)
    assert val_ids.isdisjoint(holdout_ids)
    assert len(train_ids) == 20


def test_sample_blocked_splits_caps_when_data_is_small() -> None:
    """Requested sizes are capped at what the blocked dataset contains."""
    entities = [_entity(i) for i in range(10)]
    blocks = [_block("a", [0, 1, 2]), _block("b", [3, 4, 5])]
    splits = sample_blocked_splits(
        entities,
        blocks,
        train_blocks=1000,
        val_records=500,
        holdout_records=1000,
        seed=1,
        min_block_size=2,
    )
    assert len(splits.train_blocks) == 2
    remaining = 10 - 6
    assert len(splits.holdout_records) + len(splits.val_records) == remaining


def test_sample_blocked_splits_skips_singletons() -> None:
    """Train sampling ignores blocks smaller than min_block_size."""
    entities = [_entity(i) for i in range(6)]
    blocks = [_block("pair", [0, 1]), _block("s1", [2]), _block("s2", [3])]
    splits = sample_blocked_splits(
        entities,
        blocks,
        train_blocks=10,
        val_records=2,
        holdout_records=2,
        seed=2,
        min_block_size=2,
    )
    assert len(splits.train_blocks) == 1
    assert splits.train_blocks[0].block_key == "pair"


def test_sample_blocked_splits_is_deterministic() -> None:
    """The same seed yields the same split."""
    entities = [_entity(i) for i in range(40)]
    blocks = [_block(str(i), [i * 2, i * 2 + 1]) for i in range(20)]
    a = sample_blocked_splits(
        entities, blocks, train_blocks=5, val_records=4, holdout_records=6, seed=7
    )
    b = sample_blocked_splits(
        entities, blocks, train_blocks=5, val_records=4, holdout_records=6, seed=7
    )
    assert [blk.block_key for blk in a.train_blocks] == [blk.block_key for blk in b.train_blocks]
    assert [e.id for e in a.val_records] == [e.id for e in b.val_records]
    assert [e.id for e in a.holdout_records] == [e.id for e in b.holdout_records]


def test_chunk_records_packs_fixed_size_blocks() -> None:
    """Val records are packed into blocks of the requested size."""
    records = [_entity(i) for i in range(7)]
    chunks = chunk_records(records, block_size=3, prefix="val")
    assert len(chunks) == 3
    assert chunks[0].block_size == 3
    assert chunks[1].block_size == 3
    assert chunks[2].block_size == 1
    assert chunks[0].block_key == "val_0"
