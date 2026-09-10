"""Run LLM block matching over blocks and collect the predicted pairs.

Holds the business logic the benchmark CLI needs: pick the matcher for the
requested signature mode, run it over the blocks, and turn the block resolutions
into the pair set the evaluation metrics expect.
"""

import asyncio
from dataclasses import dataclass, field

from serf.dspy.dataset_signatures import (
    SIGNATURE_MODE_GENERIC,
    SIGNATURE_MODE_PER_DATASET,
    SIGNATURE_MODES,
)
from serf.dspy.types import BlockResolution, Entity, EntityBlock
from serf.logs import get_logger
from serf.match.dataset_matcher import DatasetMatcher
from serf.match.matcher import EntityMatcher

logger = get_logger(__name__)


@dataclass
class MatchOutcome:
    """Result of matching a set of blocks.

    Parameters
    ----------
    predicted_pairs : set[tuple[int, int]]
        Predicted match pairs, smaller id first
    resolved_entities : list[Entity]
        Entities returned by the matcher, for the next iteration
    resolutions : list[BlockResolution]
        Raw per-block resolutions
    single_source_blocks : int
        Blocks skipped because they held records from only one source
    dropped_candidates : int
        Candidates dropped because they referenced an unknown record id
    """

    predicted_pairs: set[tuple[int, int]] = field(default_factory=set)
    resolved_entities: list[Entity] = field(default_factory=list)
    resolutions: list[BlockResolution] = field(default_factory=list)
    single_source_blocks: int = 0
    dropped_candidates: int = 0


def create_matcher(
    signature_mode: str = SIGNATURE_MODE_GENERIC,
    dataset: str | None = None,
    model: str | None = None,
    concurrency: int | None = None,
) -> EntityMatcher:
    """Build the matcher for a signature mode.

    Parameters
    ----------
    signature_mode : str
        ``generic`` for the shared ``BlockMatch`` signature, ``per-dataset`` for
        the typed signature written for ``dataset``
    dataset : str | None
        Benchmark dataset name, required for ``per-dataset``
    model : str | None
        LLM model name. Defaults to config models.llm.
    concurrency : int | None
        Max concurrent LLM calls. Defaults to config er.matching.max_concurrent.

    Returns
    -------
    EntityMatcher
        Generic matcher, or a ``DatasetMatcher`` for per-dataset signatures

    Raises
    ------
    ValueError
        If the mode is unknown, or ``per-dataset`` is requested without a dataset
    """
    if signature_mode not in SIGNATURE_MODES:
        raise ValueError(f"Unknown signature mode: {signature_mode}. Available: {SIGNATURE_MODES}")
    if signature_mode == SIGNATURE_MODE_PER_DATASET:
        if not dataset:
            raise ValueError("signature_mode 'per-dataset' requires a dataset name")
        return DatasetMatcher(dataset, model=model, max_concurrent=concurrency)
    return EntityMatcher(model=model, max_concurrent=concurrency)


def match_blocks(
    blocks: list[EntityBlock],
    signature_mode: str = SIGNATURE_MODE_GENERIC,
    dataset: str | None = None,
    model: str | None = None,
    concurrency: int | None = None,
    limit: int | None = None,
    iteration: int = 1,
) -> MatchOutcome:
    """Match blocks with the LLM and collect the predicted pairs.

    Parameters
    ----------
    blocks : list[EntityBlock]
        Blocks to match
    signature_mode : str
        ``generic`` or ``per-dataset``
    dataset : str | None
        Benchmark dataset name, required for ``per-dataset``
    model : str | None
        LLM model name
    concurrency : int | None
        Max concurrent LLM calls
    limit : int | None
        Max blocks to send to the LLM
    iteration : int
        Current pipeline iteration number

    Returns
    -------
    MatchOutcome
        Predicted pairs, resolved entities, and per-run diagnostics
    """
    matcher = create_matcher(
        signature_mode=signature_mode,
        dataset=dataset,
        model=model,
        concurrency=concurrency,
    )
    resolutions = asyncio.run(matcher.resolve_blocks(blocks, limit=limit, iteration=iteration))
    outcome = collect_pairs(resolutions)
    if isinstance(matcher, DatasetMatcher):
        outcome.single_source_blocks = matcher.single_source_blocks
        outcome.dropped_candidates = matcher.unknown_record_ids
        logger.info(
            f"Per-dataset matching on {dataset}: {matcher.single_source_blocks} single-source "
            f"blocks skipped, {matcher.unknown_record_ids} candidates dropped for unknown ids"
        )
    return outcome


def collect_pairs(resolutions: list[BlockResolution]) -> MatchOutcome:
    """Extract predicted pairs and resolved entities from block resolutions.

    Pairs come from explicit match decisions and from merged entities'
    ``source_ids``, because the generic signature may merge without emitting a
    match decision.

    Parameters
    ----------
    resolutions : list[BlockResolution]
        Per-block resolutions

    Returns
    -------
    MatchOutcome
        Predicted pairs and resolved entities
    """
    predicted_pairs: set[tuple[int, int]] = set()
    resolved_entities: list[Entity] = []
    for resolution in resolutions:
        for match in resolution.matches:
            if match.is_match:
                left, right = match.entity_a_id, match.entity_b_id
                predicted_pairs.add((min(left, right), max(left, right)))
        for entity in resolution.resolved_entities:
            for source_id in entity.source_ids or []:
                predicted_pairs.add((min(entity.id, source_id), max(entity.id, source_id)))
        resolved_entities.extend(resolution.resolved_entities)
    return MatchOutcome(
        predicted_pairs=predicted_pairs,
        resolved_entities=resolved_entities,
        resolutions=list(resolutions),
    )
