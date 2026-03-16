"""Edge resolution after entity merging."""

import asyncio
import json
from typing import Any

import dspy

from serf.dspy.signatures import EdgeResolve
from serf.logs import get_logger

logger = get_logger(__name__)


class EdgeResolver:
    """Resolve duplicate edges after entity merging.

    When entities are merged, edges pointing to merged nodes create
    duplicates. This resolver groups edges by (src, dst, type) and
    uses an LLM to intelligently merge duplicates.
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        """Initialize the edge resolver.

        Parameters
        ----------
        max_concurrent : int, optional
            Maximum number of concurrent LLM calls for resolving edge blocks
        """
        self.max_concurrent = max_concurrent
        self._predictor = dspy.Predict(EdgeResolve)

    def group_edges(self, edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Group edges by (src_id, dst_id, type) key.

        Parameters
        ----------
        edges : list[dict[str, Any]]
            List of edge dicts with src_id, dst_id, type (or src, dst, type)

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Map from group key to list of edges in that group
        """
        groups: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            src = edge.get("src_id", edge.get("src"))
            dst = edge.get("dst_id", edge.get("dst"))
            etype = edge.get("type", edge.get("edge_type", ""))
            key = json.dumps([src, dst, etype], sort_keys=True)
            if key not in groups:
                groups[key] = []
            groups[key].append(edge)
        return groups

    async def resolve_edge_block(
        self, block_key: str, edges: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Resolve a single block of duplicate edges.

        On error, return original edges unchanged.

        Parameters
        ----------
        block_key : str
            Key identifying this block (e.g. JSON of [src, dst, type])
        edges : list[dict[str, Any]]
            Edges in this block

        Returns
        -------
        list[dict[str, Any]]
            Resolved edges (deduplicated/merged), or original on error
        """
        if len(edges) <= 1:
            return edges

        try:
            # Treat edge data as untrusted — delimit clearly from instructions
            edge_block_json = json.dumps(edges, default=str)
            result = await asyncio.to_thread(self._predictor, edge_block=edge_block_json)
            resolved = json.loads(result.resolved_edges)
            if isinstance(resolved, list):
                return resolved
            return edges
        except Exception as e:
            logger.warning("Edge resolution failed for block %s: %s", block_key, e)
            return edges

    async def resolve_all(self, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resolve all duplicate edges.

        Groups edges, then resolves each group concurrently.
        Singleton groups (1 edge) are passed through.

        Parameters
        ----------
        edges : list[dict[str, Any]]
            All edges to resolve

        Returns
        -------
        list[dict[str, Any]]
            Resolved edges
        """
        groups = self.group_edges(edges)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def resolve_with_semaphore(
            key: str, block: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            async with semaphore:
                return await self.resolve_edge_block(key, block)

        tasks = [resolve_with_semaphore(key, block) for key, block in groups.items()]
        results = await asyncio.gather(*tasks)
        return [edge for block_result in results for edge in block_result]
