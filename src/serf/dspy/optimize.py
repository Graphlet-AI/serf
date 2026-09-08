"""GEPA-based reflective optimization for SERF's BlockMatch signature.

Builds labeled blocks from a benchmark's ground truth, optimizes BlockMatch's
instructions with dspy.GEPA against a held-out validation split (the test
split stays sealed until final evaluation, per docs/RESEARCH_LOOP.md Section 5
rule 5), and reports the optimized program's F1 against the hand-written
baseline.
"""

# isort: off
# numpy must load before dspy in a fresh process: dspy lazily proxies the
# numpy module, and if something later does `from numpy.typing import X`
# before numpy has been imported for real, the lazy proxy re-execs numpy's
# __init__ into an already-partially-loaded module and corrupts its C
# extension state. serf.block.pipeline (imported below) does exactly that.
import numpy  # noqa: F401
import dspy

# isort: on
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from serf.block.pipeline import SemanticBlockingPipeline
from serf.dspy.adapters import RobustXMLAdapter
from serf.dspy.signatures import BlockMatch
from serf.dspy.types import BlockResolution, Entity, EntityBlock
from serf.eval.benchmarks import BenchmarkDataset
from serf.logs import get_logger
from serf.match.few_shot import get_default_few_shot_examples
from serf.match.uuid_mapper import UUIDMapper

if TYPE_CHECKING:
    from serf.dspy.budget import TrackedLM

logger = get_logger(__name__)


def build_candidate_blocks(
    dataset: BenchmarkDataset,
    target_block_size: int = 30,
    sample_size: int | None = None,
    seed: int = 0,
    require_true_pair: bool = True,
) -> list[EntityBlock]:
    """Block a (sub)sample of a benchmark, for use as GEPA training data.

    Parameters
    ----------
    dataset : BenchmarkDataset
        Benchmark providing entities and ground truth
    target_block_size : int
        Target entities per block for the real FAISS blocking pipeline
    sample_size : int | None
        If set, sample roughly this many true-pair entities plus an equal
        number of random distractor entities before blocking (~2x total),
        to bound cost while still presenting genuine ambiguity. None blocks
        the full dataset.
    seed : int
        Random seed for sampling
    require_true_pair : bool
        If True (default), drop blocks with no true pair at all -- trivial
        to resolve (the answer is always "nothing merges") and, in
        isolation, teaches the optimizer little. Set False for a
        substantially larger, more production-realistic training set that
        also exercises correct non-merging on blocks with no match at all.

    Returns
    -------
    list[EntityBlock]
        Candidate blocks, optionally filtered to those with >=1 true pair
    """
    left, right = dataset.to_entities()
    all_entities = left + right

    if sample_size is not None:
        rng = random.Random(seed)
        gt_list = sorted(dataset.ground_truth)
        rng.shuffle(gt_list)
        sampled_ids: set[int] = set()
        for a, b in gt_list:
            if len(sampled_ids) >= sample_size:
                break
            sampled_ids.add(a)
            sampled_ids.add(b)
        # Add an equal number of random "distractor" entities. Sampling only
        # true-pair members produces artificially easy blocks (every entity
        # already has an obvious partner) and was measured to hit a perfect
        # F1 ceiling on both the baseline and GEPA-optimized program, leaving
        # no signal for the optimizer. Real production blocks mix matches
        # with plausible near-miss non-matches; this approximates that.
        remaining = [e.id for e in all_entities if e.id not in sampled_ids]
        rng.shuffle(remaining)
        sampled_ids.update(remaining[: len(sampled_ids)])
        entity_by_id = {e.id: e for e in all_entities}
        all_entities = [entity_by_id[i] for i in sampled_ids if i in entity_by_id]

    # Cap max_block_size at target_block_size itself (rather than the
    # pipeline's normal 100 default): with a small, true-pair-biased sample,
    # unsupervised clustering produces a few outsized, disproportionately
    # dense blocks (observed: 71- and 41-entity blocks from a target of 30),
    # and those are exactly the blocks that most often fail to parse -- this
    # is itself evidence for RESEARCH_LOOP.md's E1 block-size hypothesis.
    pipeline = SemanticBlockingPipeline(
        target_block_size=target_block_size, max_block_size=target_block_size
    )
    blocks, _ = pipeline.run(all_entities)

    if not require_true_pair:
        return blocks

    labeled_blocks = []
    for block in blocks:
        ids_in_block = {e.id for e in block.entities}
        has_true_pair = any(
            a in ids_in_block and b in ids_in_block for a, b in dataset.ground_truth
        )
        if has_true_pair:
            labeled_blocks.append(block)
    return labeled_blocks


def gold_resolution_for_block(
    block: EntityBlock, ground_truth: set[tuple[int, int]]
) -> BlockResolution:
    """Compute the correct BlockResolution for a block from ground truth pairs.

    Groups the block's entities into connected components under the
    ground-truth match relation restricted to this block, then applies the
    MDM convention (docs/ID_INVARIANTS.md Section 3): the lowest id in each
    component becomes the master, all others go into its source_ids.

    Parameters
    ----------
    block : EntityBlock
        Block of entities (real, pre-mapping ids)
    ground_truth : set[tuple[int, int]]
        All true (left_id, right_id) matching pairs for the dataset

    Returns
    -------
    BlockResolution
        Gold resolution: matches, merged/standalone resolved_entities
    """
    ids_in_block = {e.id for e in block.entities}
    relevant_pairs = [(a, b) for a, b in ground_truth if a in ids_in_block and b in ids_in_block]

    parent: dict[int, int] = {e.id: e.id for e in block.entities}

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in relevant_pairs:
        union(a, b)

    components: dict[int, list[int]] = {}
    for e in block.entities:
        components.setdefault(find(e.id), []).append(e.id)

    entity_by_id = {e.id: e for e in block.entities}
    resolved_entities: list[Entity] = []
    matches = []
    for member_ids in components.values():
        member_ids.sort()
        master_id = member_ids[0]
        master = entity_by_id[master_id]
        if len(member_ids) > 1:
            resolved_entities.append(master.model_copy(update={"source_ids": member_ids[1:]}))
            for other_id in member_ids[1:]:
                matches.append(
                    {"entity_a_id": master_id, "entity_b_id": other_id, "is_match": True}
                )
        else:
            resolved_entities.append(master.model_copy())

    from serf.dspy.types import MatchDecision

    return BlockResolution(
        block_key=block.block_key,
        matches=[
            MatchDecision(
                entity_a_id=m["entity_a_id"],
                entity_b_id=m["entity_b_id"],
                is_match=True,
                confidence=1.0,
                reasoning="ground truth",
            )
            for m in matches
        ],
        resolved_entities=resolved_entities,
        was_resolved=any(len(v) > 1 for v in components.values()),
        original_count=len(block.entities),
        resolved_count=len(resolved_entities),
    )


def resolution_to_pairs(resolution: BlockResolution) -> set[tuple[int, int]]:
    """Extract all implied (a, b) match pairs from a resolution's source_ids.

    Parameters
    ----------
    resolution : BlockResolution
        A resolution (gold or predicted)

    Returns
    -------
    set[tuple[int, int]]
        Normalized (min_id, max_id) pairs implied by every merge
    """
    pairs: set[tuple[int, int]] = set()
    for e in resolution.resolved_entities:
        for sid in e.source_ids or []:
            if sid != e.id:
                pairs.add((min(e.id, sid), max(e.id, sid)))
    return pairs


def build_gepa_examples(
    blocks: list[EntityBlock], ground_truth: set[tuple[int, int]]
) -> list[dspy.Example]:
    """Build GEPA-ready examples: one per block, ids mapped exactly as
    EntityMatcher would map them for a real LLM call, with a mapped gold
    resolution as the label.

    Parameters
    ----------
    blocks : list[EntityBlock]
        Labeled blocks (real, pre-mapping ids)
    ground_truth : set[tuple[int, int]]
        All true (left_id, right_id) matching pairs for the dataset

    Returns
    -------
    list[dspy.Example]
        Examples with inputs block_records/schema_info/few_shot_examples
        and label resolution, all in block-local mapped-id space
    """
    from serf.match.matcher import SCHEMA_INFO

    few_shot = get_default_few_shot_examples()
    examples = []
    for block in blocks:
        gold = gold_resolution_for_block(block, ground_truth)
        mapper = UUIDMapper()
        mapped_block = mapper.map_block(block)
        mapped_gold = _map_gold_resolution(gold, mapper)

        block_records = "\n".join(str(e.model_dump(mode="json")) for e in mapped_block.entities)
        example = dspy.Example(
            block_records=block_records,
            schema_info=SCHEMA_INFO,
            few_shot_examples=few_shot,
            resolution=mapped_gold,
        ).with_inputs("block_records", "schema_info", "few_shot_examples")
        examples.append(example)
    return examples


def _map_gold_resolution(gold: BlockResolution, mapper: UUIDMapper) -> BlockResolution:
    """Re-express a gold resolution's real ids as the mapper's mapped ids.

    Parameters
    ----------
    gold : BlockResolution
        Gold resolution in real (pre-mapping) id space
    mapper : UUIDMapper
        A mapper that has already run map_block on the corresponding block

    Returns
    -------
    BlockResolution
        Gold resolution with ids translated into mapped-id space
    """
    mapped_entities = []
    for e in gold.resolved_entities:
        mapped_id = mapper._id_to_int[e.id]
        mapped_source_ids = [mapper._id_to_int[sid] for sid in (e.source_ids or [])]
        mapped_entities.append(
            e.model_copy(update={"id": mapped_id, "source_ids": mapped_source_ids or None})
        )
    return gold.model_copy(update={"resolved_entities": mapped_entities})


def er_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: object = None,
    pred_name: str | None = None,
    pred_trace: object = None,
) -> dspy.Prediction:
    """Score a BlockMatch prediction against gold and explain the score.

    Parameters
    ----------
    gold : dspy.Example
        Example with the gold `resolution`
    pred : dspy.Prediction
        Model output with a predicted `resolution`
    trace : object
        Unused; part of the GEPA metric protocol
    pred_name : str | None
        Unused; part of the GEPA metric protocol
    pred_trace : object
        Unused; part of the GEPA metric protocol

    Returns
    -------
    dspy.Prediction
        `score` (pairwise F1 in [0, 1]) and `feedback` (text explaining
        missed/extra merges, for the reflection_lm to read)
    """
    from serf.eval.metrics import f1_score

    try:
        gold_pairs = resolution_to_pairs(gold.resolution)
        pred_pairs = resolution_to_pairs(pred.resolution)
    except Exception as e:
        return dspy.Prediction(score=0.0, feedback=f"Failed to parse resolution: {e}")

    score = f1_score(pred_pairs, gold_pairs)
    missed = gold_pairs - pred_pairs
    extra = pred_pairs - gold_pairs
    feedback_parts = [f"Pairwise F1: {score:.3f}."]
    if missed:
        feedback_parts.append(f"Missed {len(missed)} true match(es): {sorted(missed)[:10]}.")
    if extra:
        feedback_parts.append(
            f"Incorrectly merged {len(extra)} non-match(es): {sorted(extra)[:10]}."
        )
    if not missed and not extra:
        feedback_parts.append("All matches correct.")
    return dspy.Prediction(score=score, feedback=" ".join(feedback_parts))


def _build_tracked_lm(model: str, temperature: float, max_tokens: int) -> "TrackedLM":
    """Build a TrackedLM for either the Gemini Developer API or gpt-oss-*-maas
    on Vertex AI, routed by model name (docs/SERF_LONG_SHOT_PLAN.md Section 7.9).

    Parameters
    ----------
    model : str
        Model identifier, e.g. "gemini/gemini-3.5-flash-lite" or
        "openai/gpt-oss-120b-maas"
    temperature : float
        Sampling temperature
    max_tokens : int
        Max output tokens

    Returns
    -------
    TrackedLM
        A TrackedLM ready to use, billed against the matching named ledger

    Raises
    ------
    ValueError
        If a gpt-oss-*-maas model is requested without GOOGLE_CLOUD_PROJECT
        set (no GEMINI_API_KEY substitute exists for Vertex AI; see
        docs/SERF_LONG_SHOT_PLAN.md Section 7.9)
    google.auth.exceptions.DefaultCredentialsError
        If GOOGLE_CLOUD_PROJECT is set but Application Default Credentials
        are not configured (no `gcloud auth application-default login` and
        no GOOGLE_APPLICATION_CREDENTIALS service-account key)
    """
    import os

    from serf.dspy.budget import TrackedLM, get_ledger

    if "gpt-oss" in model:
        import google.auth.transport.requests

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable required for "
                f"{model} (Vertex AI Model-as-a-Service; see "
                "docs/SERF_LONG_SHOT_PLAN.md Section 7.9). Also requires "
                "Application Default Credentials: `gcloud auth "
                "application-default login`, or a service-account key via "
                "GOOGLE_APPLICATION_CREDENTIALS."
            )
        region = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        return TrackedLM(
            model,
            ledger=get_ledger("gpt_oss_120b_maas"),
            api_base=(
                f"https://{region}-aiplatform.googleapis.com/v1/projects/"
                f"{project_id}/locations/{region}/endpoints/openapi"
            ),
            api_key=creds.token,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return TrackedLM(
        model,
        ledger=get_ledger("gemini"),
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def evaluate_program(
    program: Callable[..., dspy.Prediction], examples: list[dspy.Example]
) -> float:
    """Average pairwise F1 of a program over a set of examples.

    Parameters
    ----------
    program : Callable[..., dspy.Prediction]
        A BlockMatch-shaped callable (e.g. dspy.Predict(BlockMatch) or GEPA output)
    examples : list[dspy.Example]
        Examples with gold `resolution` labels

    Returns
    -------
    float
        Mean pairwise F1 across examples (0.0 for examples where the call fails)
    """
    scores = []
    for ex in examples:
        try:
            pred = program(
                block_records=ex.block_records,
                schema_info=ex.schema_info,
                few_shot_examples=ex.few_shot_examples,
            )
            scores.append(er_metric(ex, pred).score)
        except Exception as e:
            logger.warning(f"Evaluation call failed: {e}")
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def run_gepa_optimization(
    dataset_name: str,
    task_model: str,
    reflection_model: str,
    sample_size: int | None = 60,
    target_block_size: int = 30,
    auto: str | None = "light",
    max_metric_calls: int | None = None,
    require_true_pair: bool = True,
    seed: int = 0,
) -> dict[str, object]:
    """End-to-end: build labeled blocks, split, optimize BlockMatch with GEPA,
    and evaluate the optimized program against the hand-written baseline on
    a sealed test split.

    Parameters
    ----------
    dataset_name : str
        Benchmark dataset name (e.g. "dblp-acm")
    task_model : str
        Student/task LM, e.g. "gemini/gemini-3.5-flash-lite"
    reflection_model : str
        GEPA's reflection_lm, e.g. "gemini/gemini-3.7-flash"
    sample_size : int | None
        Number of ground-truth pairs' entities to sample before blocking.
        None blocks the full dataset, for the largest possible training set.
    target_block_size : int
        Target entities per block
    auto : str | None
        GEPA's auto budget: "light", "medium", or "heavy". Mutually
        exclusive with max_metric_calls; set this to None when passing
        max_metric_calls explicitly.
    require_true_pair : bool
        If True (default), only train on blocks with >=1 true pair. Set
        False for a substantially larger, more production-realistic
        training set (see build_candidate_blocks).
    max_metric_calls : int | None
        Explicit cap on the number of metric calls GEPA may make, for
        predictable runtime independent of trainset size. Takes precedence
        over `auto` when both would otherwise be considered.
    seed : int
        Random seed for the train/val/test split

    Returns
    -------
    dict[str, object]
        baseline_f1, optimized_f1, n_train, n_val, n_test, optimized_program
    """
    dataset = BenchmarkDataset.download(dataset_name)
    blocks = build_candidate_blocks(
        dataset,
        target_block_size=target_block_size,
        sample_size=sample_size,
        seed=seed,
        require_true_pair=require_true_pair,
    )
    logger.info(f"Built {len(blocks)} candidate blocks from {dataset_name}")

    examples = build_gepa_examples(blocks, dataset.ground_truth)
    rng = random.Random(seed)
    rng.shuffle(examples)
    n = len(examples)
    n_train = max(1, int(n * 0.5))
    n_val = max(1, int(n * 0.25))
    trainset = examples[:n_train]
    valset = examples[n_train : n_train + n_val]
    testset = examples[n_train + n_val :]
    logger.info(f"Split: {len(trainset)} train, {len(valset)} val, {len(testset)} test (sealed)")

    # Task LM stays at temperature=0, matching production (EntityMatcher):
    # an optimized prompt is only useful if it was tuned under the same
    # conditions it will actually run under. The reflection_lm uses 1.0
    # (DSPy's GEPA convention for creative instruction proposals, Section
    # 7.7 of the plan doc; also required for reliable behavior specifically
    # on Gemini 3.x models, per LiteLLM's own provider guidance).
    from serf.config import config as serf_config

    max_output_tokens = serf_config.get("er.matching.max_output_tokens", 65536)
    task_lm = _build_tracked_lm(task_model, temperature=0.0, max_tokens=max_output_tokens)
    reflection_lm = _build_tracked_lm(
        reflection_model, temperature=1.0, max_tokens=max_output_tokens
    )
    dspy.configure(lm=task_lm, adapter=RobustXMLAdapter())

    student = cast(dspy.Module, dspy.Predict(BlockMatch))
    baseline_f1 = evaluate_program(student, testset)
    logger.info(f"Baseline (hand-written) test F1: {baseline_f1:.4f}")

    optimizer = dspy.GEPA(
        metric=cast(Any, er_metric),
        reflection_lm=reflection_lm,
        auto=None if max_metric_calls else cast(Any, auto),
        max_metric_calls=max_metric_calls,
        num_threads=4,
        track_stats=True,
    )
    optimized = optimizer.compile(student, trainset=trainset, valset=valset)

    optimized_f1 = evaluate_program(optimized, testset)
    logger.info(f"Optimized (GEPA) test F1: {optimized_f1:.4f}")

    return {
        "baseline_f1": baseline_f1,
        "optimized_f1": optimized_f1,
        "n_train": len(trainset),
        "n_val": len(valset),
        "n_test": len(testset),
        "optimized_program": optimized,
    }
