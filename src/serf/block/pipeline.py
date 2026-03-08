"""Semantic blocking pipeline.

Orchestrates the embed → cluster → split workflow for creating
entity blocks for matching.
"""

from serf.block.embeddings import EntityEmbedder
from serf.block.faiss_blocker import FAISSBlocker
from serf.dspy.types import BlockingMetrics, Entity, EntityBlock
from serf.logs import get_logger

logger = get_logger(__name__)


def split_oversized_block(block: EntityBlock, max_block_size: int) -> list[EntityBlock]:
    """Split a block that exceeds the maximum size into sub-blocks.

    Parameters
    ----------
    block : EntityBlock
        The oversized block to split
    max_block_size : int
        Maximum entities per block

    Returns
    -------
    list[EntityBlock]
        List of smaller blocks
    """
    if block.block_size <= max_block_size:
        return [block]

    sub_blocks = []
    entities = block.entities
    for i in range(0, len(entities), max_block_size):
        chunk = entities[i : i + max_block_size]
        sub_block = EntityBlock(
            block_key=f"{block.block_key}_sub{i // max_block_size}",
            block_key_type=block.block_key_type,
            block_size=len(chunk),
            entities=chunk,
        )
        sub_blocks.append(sub_block)

    return sub_blocks


class SemanticBlockingPipeline:
    """Orchestrates semantic blocking: embed → cluster → split.

    Parameters
    ----------
    model_name : str | None
        Embedding model name. Defaults to config.
    target_block_size : int
        Target entities per block
    max_block_size : int
        Maximum entities per block (oversized blocks are split)
    iteration : int
        Current ER iteration
    auto_scale : bool
        Whether to auto-scale target_block_size by iteration
    """

    def __init__(
        self,
        model_name: str | None = None,
        target_block_size: int = 30,
        max_block_size: int = 100,
        iteration: int = 1,
        auto_scale: bool = True,
        blocking_fields: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.target_block_size = target_block_size
        self.max_block_size = max_block_size
        self.iteration = iteration
        self.auto_scale = auto_scale
        self.blocking_fields = blocking_fields
        self._embedder: EntityEmbedder | None = None
        self._blocker: FAISSBlocker | None = None

    @property
    def embedder(self) -> EntityEmbedder:
        """Lazy-load the embedder."""
        if self._embedder is None:
            self._embedder = EntityEmbedder(model_name=self.model_name)
        return self._embedder

    @property
    def blocker(self) -> FAISSBlocker:
        """Lazy-load the blocker."""
        if self._blocker is None:
            self._blocker = FAISSBlocker(
                target_block_size=self.target_block_size,
                iteration=self.iteration,
                auto_scale=self.auto_scale,
            )
        return self._blocker

    def run(self, entities: list[Entity]) -> tuple[list[EntityBlock], BlockingMetrics]:
        """Run the full blocking pipeline.

        Parameters
        ----------
        entities : list[Entity]
            Entities to block

        Returns
        -------
        tuple[list[EntityBlock], BlockingMetrics]
            Tuple of (blocks, metrics)
        """
        if not entities:
            return [], BlockingMetrics()

        logger.info(f"Blocking {len(entities)} entities")

        # Build entity lookup
        entity_map = {str(e.id): e for e in entities}
        ids = [str(e.id) for e in entities]

        # Embed (name-only by default, configurable via blocking_fields)
        texts = [e.text_for_embedding(self.blocking_fields) for e in entities]
        logger.info("Computing embeddings...")
        embeddings = self.embedder.embed(texts)

        # Cluster
        logger.info("Clustering with FAISS...")
        block_assignments = self.blocker.block(embeddings, ids)

        # Build EntityBlocks
        blocks: list[EntityBlock] = []
        singleton_count = 0

        for block_key, entity_ids in block_assignments.items():
            block_entities = [entity_map[eid] for eid in entity_ids]
            block = EntityBlock(
                block_key=block_key,
                block_key_type="semantic",
                block_size=len(block_entities),
                entities=block_entities,
            )

            if block.block_size == 1:
                singleton_count += 1

            # Split oversized blocks
            sub_blocks = split_oversized_block(block, self.max_block_size)
            blocks.extend(sub_blocks)

        # Compute metrics
        block_sizes = [b.block_size for b in blocks]
        total_entities = sum(block_sizes)
        n = len(entities)
        total_possible_pairs = n * (n - 1) // 2
        blocked_pairs = sum(s * (s - 1) // 2 for s in block_sizes)

        metrics = BlockingMetrics(
            total_blocks=len(blocks),
            total_entities=total_entities,
            avg_block_size=total_entities / len(blocks) if blocks else 0.0,
            max_block_size=max(block_sizes) if block_sizes else 0,
            singleton_blocks=singleton_count,
            pair_completeness=0.0,  # Requires ground truth to compute
            reduction_ratio=(
                1.0 - blocked_pairs / total_possible_pairs if total_possible_pairs > 0 else 0.0
            ),
        )

        logger.info(
            f"Blocking complete: {metrics.total_blocks} blocks, "
            f"avg size {metrics.avg_block_size:.1f}, "
            f"max size {metrics.max_block_size}, "
            f"reduction ratio {metrics.reduction_ratio:.4f}"
        )

        return blocks, metrics
