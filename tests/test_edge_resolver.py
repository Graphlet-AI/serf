"""Tests for EdgeResolver."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from serf.edge.resolver import EdgeResolver


def test_group_edges_groups_correctly() -> None:
    """Test group_edges groups correctly by (src_id, dst_id, type)."""
    edges = [
        {"src_id": 1, "dst_id": 2, "type": "owns", "weight": 1},
        {"src_id": 1, "dst_id": 2, "type": "owns", "weight": 2},
        {"src_id": 1, "dst_id": 3, "type": "owns", "weight": 1},
    ]
    resolver = EdgeResolver()
    groups = resolver.group_edges(edges)

    assert len(groups) == 2
    block_sizes = [len(g) for g in groups.values()]
    assert 2 in block_sizes
    assert 1 in block_sizes


def test_group_edges_accepts_src_dst_alternatives() -> None:
    """Test group_edges accepts src/dst as well as src_id/dst_id."""
    edges = [
        {"src": "a", "dst": "b", "type": "link"},
        {"src": "a", "dst": "b", "type": "link"},
    ]
    resolver = EdgeResolver()
    groups = resolver.group_edges(edges)

    assert len(groups) == 1
    assert len(list(groups.values())[0]) == 2


@pytest.mark.asyncio
async def test_singleton_edges_pass_through() -> None:
    """Test singleton edges pass through in resolve_all."""
    edges = [
        {"src_id": 1, "dst_id": 2, "type": "owns"},
    ]
    resolver = EdgeResolver()
    result = await resolver.resolve_all(edges)

    assert len(result) == 1
    assert result[0]["src_id"] == 1
    assert result[0]["dst_id"] == 2


@pytest.mark.asyncio
@patch("serf.edge.resolver.dspy.Predict")
async def test_resolve_edge_block_with_mocked_dspy(mock_predict_cls: MagicMock) -> None:
    """Test resolve_edge_block with mocked DSPy."""
    mock_instance = MagicMock()
    mock_instance.return_value = MagicMock(
        resolved_edges='[{"src_id": 1, "dst_id": 2, "type": "owns", "merged": true}]'
    )
    mock_predict_cls.return_value = mock_instance

    edges: list[dict[str, Any]] = [
        {"src_id": 1, "dst_id": 2, "type": "owns", "weight": 1},
        {"src_id": 1, "dst_id": 2, "type": "owns", "weight": 2},
    ]
    resolver = EdgeResolver()
    result = await resolver.resolve_edge_block("test_key", edges)

    assert len(result) == 1
    assert result[0]["merged"] is True
    mock_instance.assert_called_once()
