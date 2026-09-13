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
from serf.match.partition import partition_from_pairs
from serf.merge.canonical import IdAllocator
from serf.merge.merger import EntityMerger

logger = get_logger(__name__)

ERROR_RECOVERY_REASON = "error_recovery"


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
    failed_blocks : int
        Blocks whose LLM call failed and fell back to error recovery, so they
        contributed no matches
    recovered_records : int
        Records the matcher left out of every group and that were put back as
        unmatched rather than dropped from the dataset
    """

    predicted_pairs: set[tuple[int, int]] = field(default_factory=set)
    resolved_entities: list[Entity] = field(default_factory=list)
    resolutions: list[BlockResolution] = field(default_factory=list)
    single_source_blocks: int = 0
    dropped_candidates: int = 0
    failed_blocks: int = 0
    recovered_records: int = 0


def create_matcher(
    signature_mode: str = SIGNATURE_MODE_GENERIC,
    dataset: str | None = None,
    model: str | None = None,
    concurrency: int | None = None,
    trained_prompts: bool = False,
    trained_dir: str | None = None,
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
    trained_prompts : bool
        Match with the instructions ``serf train`` wrote for this dataset.
        Per-dataset signatures only; the generic signature has no trained form.
    trained_dir : str | None
        Directory holding trained programs. Defaults to config
        ``optimize.trained_dir``.

    Returns
    -------
    EntityMatcher
        Generic matcher, or a ``DatasetMatcher`` for per-dataset signatures

    Raises
    ------
    ValueError
        If the mode is unknown, ``per-dataset`` is requested without a dataset,
        or trained prompts are requested for the generic signature
    """
    if signature_mode not in SIGNATURE_MODES:
        raise ValueError(f"Unknown signature mode: {signature_mode}. Available: {SIGNATURE_MODES}")
    if signature_mode == SIGNATURE_MODE_PER_DATASET:
        if not dataset:
            raise ValueError("signature_mode 'per-dataset' requires a dataset name")
        return DatasetMatcher(
            dataset,
            model=model,
            max_concurrent=concurrency,
            trained_prompts=trained_prompts,
            trained_dir=trained_dir,
        )
    if trained_prompts:
        raise ValueError(
            "trained_prompts requires signature_mode 'per-dataset'; `serf train` writes a "
            "program per dataset signature, and the shared BlockMatch signature has none"
        )
    return EntityMatcher(model=model, max_concurrent=concurrency)


def match_blocks(
    blocks: list[EntityBlock],
    signature_mode: str = SIGNATURE_MODE_GENERIC,
    dataset: str | None = None,
    model: str | None = None,
    concurrency: int | None = None,
    limit: int | None = None,
    iteration: int = 1,
    trained_prompts: bool = False,
    trained_dir: str | None = None,
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
    trained_prompts : bool
        Match with the instructions ``serf train`` wrote for this dataset
    trained_dir : str | None
        Directory holding trained programs

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
        trained_prompts=trained_prompts,
        trained_dir=trained_dir,
    )
    resolutions = asyncio.run(matcher.resolve_blocks(blocks, limit=limit, iteration=iteration))
    outcome = collect_pairs(resolutions)
    if outcome.failed_blocks:
        logger.warning(
            f"{outcome.failed_blocks} of {len(resolutions)} blocks failed their LLM call and "
            f"contributed no matches, so recall for this run is understated"
        )
    if isinstance(matcher, DatasetMatcher):
        outcome.single_source_blocks = matcher.single_source_blocks
        outcome.dropped_candidates = matcher.unknown_record_ids
        logger.info(
            f"Per-dataset matching on {dataset}: {matcher.single_source_blocks} single-source "
            f"blocks skipped, {matcher.unknown_record_ids} candidates dropped for unknown ids"
        )
    return outcome


def entity_members(
    entities: list[Entity], original_ids: set[int] | None = None
) -> dict[int, set[int]]:
    """Map each entity id to the original record ids it stands for.

    An entity merged in an earlier iteration keeps the records it absorbed in
    ``source_ids``, so a later match against that entity is really a match
    against every one of them.

    A merged entity carries a minted id that belongs to no source record, and
    a pair naming one would be a pair about a record the ground truth has
    never heard of. Pass ``original_ids`` to keep the expansion inside the
    universe the data actually has.

    Parameters
    ----------
    entities : list[Entity]
        Entities handed to the current iteration
    original_ids : set[int] | None
        Ids of the records the run started from. When given, minted ids are
        excluded from the expansion.

    Returns
    -------
    dict[int, set[int]]
        Entity id to the set of record ids it covers, including its own
    """
    members: dict[int, set[int]] = {}
    for entity in entities:
        covered = {entity.id, *(entity.source_ids or [])}
        members[entity.id] = covered if original_ids is None else covered & original_ids
    return members


def expand_pairs(pairs: set[tuple[int, int]], members: dict[int, set[int]]) -> set[tuple[int, int]]:
    """Expand pairs of merged entities into pairs of original records.

    Matching two entities that each already stand for several records asserts a
    match between every record on one side and every record on the other, which
    is what the benchmark ground truth is expressed over.

    Parameters
    ----------
    pairs : set[tuple[int, int]]
        Pairs over current entity ids
    members : dict[int, set[int]]
        Entity id to the record ids it covers

    Returns
    -------
    set[tuple[int, int]]
        Pairs over original record ids, smaller id first
    """
    expanded: set[tuple[int, int]] = set()
    for left, right in pairs:
        for left_member in members.get(left, {left}):
            for right_member in members.get(right, {right}):
                if left_member != right_member:
                    expanded.add((min(left_member, right_member), max(left_member, right_member)))
    return expanded


def merge_matched_entities(
    entities: list[Entity],
    pairs: set[tuple[int, int]],
    allocator: IdAllocator | None = None,
) -> list[Entity]:
    """Collapse every connected component of matched entities into one entity.

    The result is what the next iteration re-blocks: a record that was matched
    is now carried by its merged entity, whose name and attributes are the
    fullest of the group, so it can land in a different block and meet records
    the first round of blocking kept away from it.

    A merge produces a new entity, so it gets a minted id and every id it
    absorbed, transitively, moves into ``source_ids``. An entity that merged
    with nothing keeps its own id.

    Parameters
    ----------
    entities : list[Entity]
        Entities handed to the current iteration
    pairs : set[tuple[int, int]]
        Pairs the matcher predicted over those entities
    allocator : IdAllocator | None
        Source of minted ids. One seeded from every id and lineage entry in
        ``entities`` is built when omitted, which is correct as long as
        ``entities`` is the whole working set.

    Returns
    -------
    list[Entity]
        One entity per connected component, sorted by id
    """
    groups = partition_from_pairs(pairs, {e.id for e in entities})
    by_id = {e.id: e for e in entities}
    if allocator is None:
        allocator = IdAllocator(
            {
                entity_id
                for entity in entities
                for entity_id in (entity.id, *(entity.source_ids or []))
            }
        )

    merger = EntityMerger()
    merged: list[Entity] = []
    for group in groups:
        members = [by_id[entity_id] for entity_id in group]
        combined = merger.merge_entities(members)
        if len(members) == 1:
            merged.append(combined)
            continue
        lineage = sorted(
            {
                entity_id
                for member in members
                for entity_id in (member.id, *(member.source_ids or []))
            }
        )
        merged.append(combined.model_copy(update={"id": allocator.mint(), "source_ids": lineage}))
    return sorted(merged, key=lambda e: e.id)


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
    failed_blocks = 0
    recovered_records = 0
    for resolution in resolutions:
        if any(e.match_skip_reason == ERROR_RECOVERY_REASON for e in resolution.resolved_entities):
            failed_blocks += 1
        recovered_records += len(resolution.recovered_ids)
        if resolution.groups:
            for group in resolution.groups:
                for index, left in enumerate(group):
                    for right in group[index + 1 :]:
                        predicted_pairs.add((min(left, right), max(left, right)))
        else:
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
        failed_blocks=failed_blocks,
        recovered_records=recovered_records,
    )
