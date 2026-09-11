"""Block matching with per-dataset, strongly-typed DSPy signatures.

``EntityMatcher`` sends every dataset through the one generic ``BlockMatch``
signature. ``DatasetMatcher`` sends a block through the signature written for
that specific dataset instead, feeding each source's records as their own typed
list and reading typed candidate pairs back out.

Blocking is untouched: blocks still mix both sources. This matcher splits the
block it is handed by source, because gold matches in every benchmark task cross
the two sources by construction.
"""

from typing import Any, cast

import dspy

from serf.dspy.dataset_signatures import DatasetSignatureSpec, get_dataset_spec
from serf.dspy.schemas.base import (
    SIDE_LEFT,
    SIDE_RIGHT,
    EntityMatchCandidate,
    EntitySide,
    entity_side,
)
from serf.dspy.trained import load_trained_predictor
from serf.dspy.types import BlockResolution, Entity, EntityBlock, MatchDecision
from serf.logs import get_logger
from serf.match.matcher import EntityMatcher
from serf.match.uuid_mapper import UUIDMapper

logger = get_logger(__name__)

SINGLE_SOURCE_SKIP_REASON = "single_source_block"


class DatasetMatcher(EntityMatcher):
    """Resolves blocks with the DSPy signature written for one benchmark dataset.

    Reuses ``EntityMatcher`` for the LM, the async concurrency and the error
    recovery path, and overrides only how a single block is turned into a prompt
    and how the result is read back.

    Parameters
    ----------
    dataset : str
        Benchmark dataset name with a per-dataset signature
    model : str | None
        LLM model name. Defaults to config models.llm.
    batch_size : int | None
        Batch size for processing. Defaults to config er.matching.batch_size.
    max_concurrent : int | None
        Max concurrent LLM calls. Defaults to config er.matching.max_concurrent.
    trained_prompts : bool
        Load the instructions ``serf train`` wrote for this dataset instead of
        the ones in the signature's docstring
    trained_dir : str | None
        Directory holding trained programs. Defaults to config
        ``optimize.trained_dir``.
    """

    def __init__(
        self,
        dataset: str,
        model: str | None = None,
        batch_size: int | None = None,
        max_concurrent: int | None = None,
        trained_prompts: bool = False,
        trained_dir: str | None = None,
    ) -> None:
        super().__init__(model=model, batch_size=batch_size, max_concurrent=max_concurrent)
        self.dataset = dataset
        self.spec: DatasetSignatureSpec = get_dataset_spec(dataset)
        self.trained_prompts = trained_prompts
        self.trained_dir = trained_dir
        self.single_source_blocks = 0
        self.unknown_record_ids = 0

    @property
    def predictor(self) -> dspy.Predict:
        """Lazy-load the per-dataset predictor, trained instructions included."""
        if self._predictor is None:
            trained = (
                load_trained_predictor(self.dataset, self.trained_dir)
                if self.trained_prompts
                else None
            )
            self._predictor = trained or cast(dspy.Predict, dspy.Predict(self.spec.signature))
        return self._predictor

    def resolve_block(self, block: EntityBlock, iteration: int = 1) -> BlockResolution:
        """Match one block with this dataset's typed signature.

        Parameters
        ----------
        block : EntityBlock
            Block of entities from both sources
        iteration : int
            Current pipeline iteration number

        Returns
        -------
        BlockResolution
            Resolution whose ``matches`` hold the pairs the LLM judged to match
        """
        mapper = UUIDMapper()
        mapped_block = mapper.map_block(block)
        left_entities = [e for e in mapped_block.entities if entity_side(e) == SIDE_LEFT]
        right_entities = [e for e in mapped_block.entities if entity_side(e) == SIDE_RIGHT]

        if not left_entities or not right_entities:
            self.single_source_blocks += 1
            logger.debug(
                f"Block {block.block_key} holds records from one source only "
                f"({len(left_entities)} left, {len(right_entities)} right); no cross-source "
                "pair is possible, skipping the LLM call"
            )
            return self._assign_uuids(self._single_source_resolution(block))

        left_records = [self.spec.left_type.from_entity(e) for e in left_entities]
        right_records = [self.spec.right_type.from_entity(e) for e in right_entities]
        inputs: dict[str, Any] = {
            self.spec.left_field: left_records,
            self.spec.right_field: right_records,
        }

        try:
            lm = self._ensure_lm()
            with dspy.context(lm=lm, adapter=self._adapter):
                result = self.predictor(**inputs)
            candidates: list[EntityMatchCandidate] = (
                getattr(result, self.spec.candidates_field, None) or []
            )
        except Exception as e:
            logger.error(f"LLM failure for block {block.block_key}: {e}")
            return self._assign_uuids(self._error_recovery_resolution(block, iteration))

        known_ids = {e.id for e in mapped_block.entities}
        resolution = BlockResolution(
            block_key=block.block_key,
            matches=self._candidate_matches(candidates, known_ids, block.block_key),
            resolved_entities=list(mapped_block.entities),
            was_resolved=False,
            original_count=mapped_block.block_size,
            resolved_count=mapped_block.block_size,
        )
        resolution.was_resolved = bool(resolution.matches)
        resolution = mapper.unmap_block(resolution, block)
        return self._assign_uuids(resolution)

    def _candidate_matches(
        self,
        candidates: list[EntityMatchCandidate],
        known_ids: set[int],
        block_key: str,
    ) -> list[MatchDecision]:
        """Convert typed candidates into match decisions.

        Parameters
        ----------
        candidates : list[EntityMatchCandidate]
            Candidate pairs returned by the LLM
        known_ids : set[int]
            Record ids that actually exist in the block
        block_key : str
            Block identifier, for logging

        Returns
        -------
        list[MatchDecision]
            One decision per candidate the LLM marked as a match
        """
        matches: list[MatchDecision] = []
        for candidate in candidates:
            if not candidate.is_match:
                continue
            left_id = candidate.left.record_id
            right_id = candidate.right.record_id
            if left_id not in known_ids or right_id not in known_ids:
                self.unknown_record_ids += 1
                logger.warning(
                    f"Block {block_key}: candidate ({left_id}, {right_id}) references a "
                    "record id that is not in the block; dropping it"
                )
                continue
            if left_id == right_id:
                continue
            matches.append(
                MatchDecision(
                    entity_a_id=left_id,
                    entity_b_id=right_id,
                    is_match=True,
                    confidence=candidate.confidence,
                    reasoning=candidate.justification,
                )
            )
        return matches

    def _single_source_resolution(self, block: EntityBlock) -> BlockResolution:
        """Build a pass-through resolution for a block with only one source in it.

        Parameters
        ----------
        block : EntityBlock
            Original block

        Returns
        -------
        BlockResolution
            Pass-through resolution marked ``single_source_block``
        """
        entities: list[Entity] = [
            e.model_copy(
                update={
                    "match_skip": True,
                    "match_skip_reason": SINGLE_SOURCE_SKIP_REASON,
                }
            )
            for e in block.entities
        ]
        return BlockResolution(
            block_key=block.block_key,
            matches=[],
            resolved_entities=entities,
            was_resolved=False,
            original_count=len(entities),
            resolved_count=len(entities),
        )


def typed_sides(block: EntityBlock, spec: DatasetSignatureSpec) -> tuple[list[Any], list[Any]]:
    """Split a block into its two typed sides.

    Exposed for tests and for callers that want the typed view of a block without
    running a match.

    Parameters
    ----------
    block : EntityBlock
        Block of entities from both sources
    spec : DatasetSignatureSpec
        Typed contract for the dataset

    Returns
    -------
    tuple[list[Any], list[Any]]
        Left-side records and right-side records
    """
    left: list[EntitySide] = [
        spec.left_type.from_entity(e) for e in block.entities if entity_side(e) == SIDE_LEFT
    ]
    right: list[EntitySide] = [
        spec.right_type.from_entity(e) for e in block.entities if entity_side(e) == SIDE_RIGHT
    ]
    return left, right
