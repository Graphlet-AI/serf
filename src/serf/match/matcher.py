"""Entity matcher using DSPy BlockMatch for block resolution."""

import asyncio
import json
import os
from uuid import uuid4

import dspy

from serf.config import config
from serf.dspy.baml_adapter import BAMLAdapter
from serf.dspy.signatures import BlockMatch
from serf.dspy.types import BlockResolution, EntityBlock
from serf.logs import get_logger
from serf.match.few_shot import get_default_few_shot_examples
from serf.match.uuid_mapper import UUIDMapper

logger = get_logger(__name__)

SCHEMA_INFO = (
    "Entity: id (int), name (str), description (str), entity_type (str), "
    "attributes (dict), source_ids (list[int] of merged entity IDs). "
    "Lowest id becomes master; merge source_ids from all matched entities."
)


class EntityMatcher:
    """Resolves entity blocks via LLM using DSPy BlockMatch.

    Uses UUIDMapper for ID mapping, few-shot examples, and async processing
    with rate limiting. On LLM failure, marks all entities with
    match_skip_reason='error_recovery'.
    """

    def __init__(
        self,
        model: str | None = None,
        batch_size: int | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        """Initialize the matcher.

        Parameters
        ----------
        model : str | None
            LLM model name. Defaults to config models.llm.
        batch_size : int | None
            Batch size for processing. Defaults to config er.matching.batch_size.
        max_concurrent : int | None
            Max concurrent LLM calls. Defaults to config er.matching.max_concurrent.
        """
        self.model = model or config.get("models.llm", "gemini/gemini-2.0-flash")
        self.batch_size = batch_size or config.get("er.matching.batch_size", 10)
        self.max_concurrent = max_concurrent or config.get("er.matching.max_concurrent", 20)
        self._predictor: dspy.Predict | None = None
        self._lm: dspy.LM | None = None
        self._adapter = BAMLAdapter()

    def _ensure_lm(self) -> dspy.LM:
        """Get or create the LM instance."""
        if self._lm is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable required")
            temperature = config.get("er.matching.temperature", 0.0)
            self._lm = dspy.LM(
                self.model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=8192,
            )
        return self._lm

    @property
    def predictor(self) -> dspy.Predict:
        """Lazy-load the BlockMatch predictor."""
        if self._predictor is None:
            self._predictor = dspy.Predict(BlockMatch)
        return self._predictor

    def resolve_block(self, block: EntityBlock) -> BlockResolution:
        """Process a single block through the LLM.

        Parameters
        ----------
        block : EntityBlock
            Block of entities to resolve

        Returns
        -------
        BlockResolution
            Resolution with merged and non-matched entities
        """
        mapper = UUIDMapper()
        mapped_block = mapper.map_block(block)

        block_records = json.dumps(
            [e.model_dump(mode="json") for e in mapped_block.entities],
            indent=2,
        )
        few_shot = get_default_few_shot_examples()

        try:
            lm = self._ensure_lm()
            with dspy.context(lm=lm, adapter=self._adapter):
                result = self.predictor(
                    block_records=block_records,
                    schema_info=SCHEMA_INFO,
                    few_shot_examples=few_shot,
                )
            resolution = result.resolution
        except Exception as e:
            logger.error(f"LLM failure for block {block.block_key}: {e}")
            resolution = self._error_recovery_resolution(block)
            return self._assign_uuids(resolution)

        resolution = mapper.unmap_block(resolution, block)
        resolution = self._assign_uuids(resolution)
        return resolution

    def _error_recovery_resolution(self, block: EntityBlock) -> BlockResolution:
        """Build resolution with all entities marked error_recovery.

        Parameters
        ----------
        block : EntityBlock
            Original block

        Returns
        -------
        BlockResolution
            Pass-through resolution with error_recovery
        """
        entities = []
        for e in block.entities:
            entities.append(
                e.model_copy(
                    update={
                        "match_skip": True,
                        "match_skip_reason": "error_recovery",
                    }
                )
            )
        return BlockResolution(
            block_key=block.block_key,
            matches=[],
            resolved_entities=entities,
            was_resolved=False,
            original_count=len(entities),
            resolved_count=len(entities),
        )

    def _assign_uuids(self, resolution: BlockResolution) -> BlockResolution:
        """Assign new UUIDs to resolved entities.

        Parameters
        ----------
        resolution : BlockResolution
            Resolution with entities

        Returns
        -------
        BlockResolution
            Resolution with UUIDs assigned
        """
        entities = []
        for e in resolution.resolved_entities:
            entities.append(e.model_copy(update={"uuid": str(uuid4())}))
        return resolution.model_copy(update={"resolved_entities": entities})

    async def resolve_blocks(
        self,
        blocks: list[EntityBlock],
        limit: int | None = None,
    ) -> list[BlockResolution]:
        """Process blocks with async concurrency and rate limiting.

        Fires up to max_concurrent LLM calls simultaneously using
        asyncio.Semaphore for rate limiting and tqdm for progress.

        Parameters
        ----------
        blocks : list[EntityBlock]
            Blocks to resolve
        limit : int | None
            Max number of blocks to process (for testing). None = all.

        Returns
        -------
        list[BlockResolution]
            Resolutions for each block
        """
        from tqdm import tqdm

        if limit is not None:
            blocks = blocks[:limit]

        total = len(blocks)
        logger.info(f"Processing {total} blocks with {self.max_concurrent} concurrent LLM calls")

        semaphore = asyncio.Semaphore(self.max_concurrent)
        progress = tqdm(total=total, desc="Matching blocks", unit="block")

        async def process_one(block: EntityBlock) -> BlockResolution:
            async with semaphore:
                result = await asyncio.to_thread(self.resolve_block, block)
                progress.update(1)
                return result

        tasks = [process_one(b) for b in blocks]
        results = list(await asyncio.gather(*tasks))
        progress.close()
        return results
