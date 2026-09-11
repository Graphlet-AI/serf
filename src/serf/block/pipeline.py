"""Semantic blocking pipeline.

Orchestrates the embed → cluster → split workflow for creating
entity blocks for matching.

Uses subprocess isolation for PyTorch embedding and FAISS clustering
to avoid memory conflicts (MPS/FAISS segfault) on macOS. This is
the pattern proven in the Abzu production system.
"""

from serf.block.subprocess_embed import cluster_in_subprocess, embed_in_subprocess
from serf.config import config
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


def count_blocked_pairs(blocks: list[EntityBlock]) -> int:
    """Count the distinct record pairs that share at least one block.

    This is the matcher's bill. Summing ``size * (size - 1) / 2`` over blocks
    only gives the right answer while the blocks are disjoint, which stops
    being true once a record is blocked under more than one view of itself.

    Parameters
    ----------
    blocks : list[EntityBlock]
        Blocks to count over

    Returns
    -------
    int
        Number of distinct pairs inside at least one block
    """
    sizes = [block.block_size for block in blocks]
    if sum(sizes) == len({entity.id for block in blocks for entity in block.entities}):
        return sum(size * (size - 1) // 2 for size in sizes)

    seen: set[int] = set()
    for block in blocks:
        ids = sorted(int(entity.id) for entity in block.entities)
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                seen.add((left << 32) | right)
    return len(seen)


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
    blocking_fields : list[str] | None
        Additional fields appended to the name before embedding
    embedding_prompt : str | None
        Instruction prefix prepended to every text. Defaults to config.
    embedding_trust_remote_code : bool | None
        Execute the modelling code shipped in the model repository, which some
        architectures require. Defaults to config.
    blocking_strategy : str | None
        ``"name"`` embeds the name alone, ``"json"`` embeds every field as a
        JSON object with the field names inline, and ``"union"`` runs both and
        keeps the blocks from each, so a pair only has to be caught once by
        either view. Defaults to config.
    """

    def __init__(
        self,
        model_name: str | None = None,
        target_block_size: int = 30,
        max_block_size: int = 100,
        iteration: int = 1,
        auto_scale: bool = True,
        blocking_fields: list[str] | None = None,
        embedding_prompt: str | None = None,
        embedding_trust_remote_code: bool | None = None,
        blocking_strategy: str | None = None,
    ) -> None:
        if model_name is None:
            model_name = config.get("models.embedding")
        if embedding_prompt is None:
            embedding_prompt = config.get("models.embedding_prompt", "")
        if embedding_trust_remote_code is None:
            embedding_trust_remote_code = bool(
                config.get("models.embedding_trust_remote_code", False)
            )
        self.model_name = model_name
        self.embedding_prompt = embedding_prompt
        self.embedding_trust_remote_code = embedding_trust_remote_code
        self.blocking_strategy = blocking_strategy or str(
            config.get("er.blocking.strategy", "name")
        )
        self.target_block_size = target_block_size
        self.max_block_size = max_block_size
        self.iteration = iteration
        self.auto_scale = auto_scale
        self.blocking_fields = blocking_fields

    def _embedding_texts(self, entities: list[Entity], strategy: str) -> list[str]:
        """Return the text to embed for each entity under one strategy.

        Parameters
        ----------
        entities : list[Entity]
            Entities to render
        strategy : str
            ``"name"`` or ``"json"``

        Returns
        -------
        list[str]
            One text per entity, in the order given
        """
        if strategy == "json":
            return [e.json_for_embedding() for e in entities]
        return [e.text_for_embedding(self.blocking_fields) for e in entities]

    def _blocks_for(
        self,
        entities: list[Entity],
        strategy: str,
        target_block_size: int,
        key_prefix: str = "",
    ) -> list[EntityBlock]:
        """Embed, cluster and split one view of the entities.

        Parameters
        ----------
        entities : list[Entity]
            Entities to block
        strategy : str
            ``"name"`` or ``"json"``
        target_block_size : int
            Target entities per block
        key_prefix : str
            Prepended to every block key, so blocks built from different views
            stay distinguishable when they are combined

        Returns
        -------
        list[EntityBlock]
            Blocks from this view, already split to ``max_block_size``
        """
        entity_map = {str(e.id): e for e in entities}
        embeddings = embed_in_subprocess(
            self._embedding_texts(entities, strategy),
            model_name=self.model_name,
            prompt=self.embedding_prompt,
            trust_remote_code=self.embedding_trust_remote_code,
        )
        assignments = cluster_in_subprocess(
            embeddings, [str(e.id) for e in entities], target_block_size=target_block_size
        )

        blocks: list[EntityBlock] = []
        for block_key, entity_ids in assignments.items():
            block = EntityBlock(
                block_key=f"{key_prefix}{block_key}",
                block_key_type="semantic",
                block_size=len(entity_ids),
                entities=[entity_map[eid] for eid in entity_ids],
            )
            blocks.extend(split_oversized_block(block, self.max_block_size))
        return blocks

    def run(self, entities: list[Entity]) -> tuple[list[EntityBlock], BlockingMetrics]:
        """Run the full blocking pipeline using subprocess isolation.

        Embedding and FAISS clustering run in separate subprocesses
        to avoid PyTorch MPS / FAISS memory conflicts on macOS.

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

        effective_target = self.target_block_size
        if self.auto_scale and self.iteration > 1:
            effective_target = max(10, self.target_block_size // self.iteration)

        views = ("name", "json") if self.blocking_strategy == "union" else (self.blocking_strategy,)
        blocks: list[EntityBlock] = []
        for view in views:
            blocks.extend(
                self._blocks_for(
                    entities,
                    strategy=view,
                    target_block_size=effective_target,
                    key_prefix=f"{view}_" if len(views) > 1 else "",
                )
            )

        block_sizes = [b.block_size for b in blocks]
        total_entities = sum(block_sizes)
        n = len(entities)
        total_possible_pairs = n * (n - 1) // 2
        blocked_pairs = count_blocked_pairs(blocks)

        metrics = BlockingMetrics(
            total_blocks=len(blocks),
            total_entities=total_entities,
            avg_block_size=total_entities / len(blocks) if blocks else 0.0,
            max_block_size=max(block_sizes) if block_sizes else 0,
            singleton_blocks=sum(1 for size in block_sizes if size == 1),
            pair_completeness=0.0,  # Requires ground truth to compute
            reduction_ratio=(
                1.0 - blocked_pairs / total_possible_pairs if total_possible_pairs > 0 else 0.0
            ),
            blocked_pairs=blocked_pairs,
        )

        logger.info(
            f"Blocking complete: {metrics.total_blocks} blocks, "
            f"avg size {metrics.avg_block_size:.1f}, "
            f"max size {metrics.max_block_size}, "
            f"{metrics.blocked_pairs} pairs to compare, "
            f"reduction ratio {metrics.reduction_ratio:.4f}"
        )

        return blocks, metrics
