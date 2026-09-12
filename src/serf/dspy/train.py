"""Train a benchmark dataset's matching prompt with GEPA.

``serf optimize`` tunes the shared ``BlockMatch`` signature. This module tunes
the signature the benchmark actually runs: the per-dataset one in
``serf.dspy.dataset_signatures``, whose docstring carries the domain knowledge
for that match task and is therefore the text worth rewriting.

Three things make this different from optimizing the generic signature.

The examples are typed. A per-dataset signature takes each source as its own
list of typed records and returns typed candidate pairs, so training examples
have to be built the same way ``DatasetMatcher`` builds a live call, or the
optimized instructions would be tuned against a prompt shape that never ships.

The metric enumerates. GEPA's advantage over a scalar reward is that the
reflection model reads *why* an example scored what it did, so the feedback
names the pairs that were missed and the pairs that were invented, and prints
the records behind them. A bare F1 would throw that away.

The result is loadable. Training writes a DSPy program next to the dataset name,
and ``DatasetMatcher`` reads it back, so a trained prompt reaches the benchmark
without anyone editing a docstring.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from serf.block.pipeline import SemanticBlockingPipeline
from serf.config import config
from serf.dspy.dataset_signatures import DatasetSignatureSpec, get_dataset_spec
from serf.dspy.optimize import optimize_module
from serf.dspy.trained import predictor_instructions, trained_program_path
from serf.dspy.types import Entity, EntityBlock
from serf.eval.benchmarks import BenchmarkDataset
from serf.eval.metrics import f1_score
from serf.eval.splits import SplitSizes, count_gold_pairs, get_split_sizes, sample_random_splits
from serf.logs import get_logger
from serf.match.dataset_matcher import typed_sides

logger = get_logger(__name__)

GOLD_PAIRS_FIELD = "gold_pairs"

# How many errors of each kind the feedback names before it stops. The reflection
# model needs concrete examples, not the whole confusion set, and an unbounded
# list would crowd out the instructions in its context window.
MAX_ENUMERATED_ERRORS = 6

# How much of one record the feedback prints. Enough to see the title or the
# model code that decided the pair, short enough that six of them still fit.
MAX_RECORD_CHARACTERS = 200


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one GEPA training run.

    Parameters
    ----------
    dataset : str
        Benchmark dataset trained
    signature_name : str
        Signature class whose instructions were rewritten
    program_path : str
        Where the trained DSPy program was written
    train_examples : int
        Training blocks GEPA rolled out on
    val_examples : int
        Validation blocks GEPA selected candidates against
    baseline_score : float | None
        Validation score of the seed candidate, the prompt as shipped
    best_score : float | None
        Validation score of the candidate GEPA kept
    instructions_before : str
        Instructions the run started from
    instructions_after : str
        Instructions the run ended with
    """

    dataset: str
    signature_name: str
    program_path: str
    train_examples: int
    val_examples: int
    baseline_score: float | None
    best_score: float | None
    instructions_before: str
    instructions_after: str

    @property
    def improved(self) -> bool:
        """Whether GEPA found a candidate that beat the shipped prompt.

        Returns
        -------
        bool
            True when both scores are known and the best one is higher
        """
        if self.baseline_score is None or self.best_score is None:
            return False
        return self.best_score > self.baseline_score


def gold_pairs_in_block(block: EntityBlock, ground_truth: set[tuple[int, int]]) -> list[list[int]]:
    """Return the ground-truth pairs that both fall inside one block.

    Parameters
    ----------
    block : EntityBlock
        Block of entities from both sources
    ground_truth : set[tuple[int, int]]
        True matching pairs over record ids

    Returns
    -------
    list[list[int]]
        Pairs present in the block, smaller id first, sorted
    """
    ids = {entity.id for entity in block.entities}
    pairs = [
        [min(left, right), max(left, right)]
        for left, right in ground_truth
        if left in ids and right in ids
    ]
    return sorted(pairs)


def blocks_to_dataset_examples(
    blocks: list[EntityBlock],
    ground_truth: set[tuple[int, int]],
    spec: DatasetSignatureSpec,
) -> list[dspy.Example]:
    """Build typed training examples from blocks, the way the matcher builds calls.

    A block is only usable when it holds both sources and at least one gold pair.
    Single-source blocks never reach the LM in production, and a block with no
    gold pair inside it carries no signal about how to match: the best possible
    answer is the empty list, which every candidate prompt already returns.

    Parameters
    ----------
    blocks : list[EntityBlock]
        Blocks to convert
    ground_truth : set[tuple[int, int]]
        True matching pairs over record ids
    spec : DatasetSignatureSpec
        Typed matching contract for the dataset

    Returns
    -------
    list[dspy.Example]
        Examples with both typed sides as inputs and the gold pairs as the label
    """
    examples: list[dspy.Example] = []
    for block in blocks:
        left, right = typed_sides(block, spec)
        if not left or not right:
            continue
        pairs = gold_pairs_in_block(block, ground_truth)
        if not pairs:
            continue
        examples.append(
            dspy.Example(
                **{
                    spec.left_field: left,
                    spec.right_field: right,
                    GOLD_PAIRS_FIELD: pairs,
                }
            ).with_inputs(spec.left_field, spec.right_field)
        )
    return examples


def _normalize(pairs: Any) -> set[tuple[int, int]]:
    """Coerce a pair collection into a set of ordered id pairs.

    Parameters
    ----------
    pairs : Any
        Iterable of two-element sequences, or None

    Returns
    -------
    set[tuple[int, int]]
        Pairs with the smaller id first
    """
    out: set[tuple[int, int]] = set()
    for pair in pairs or []:
        left, right = int(pair[0]), int(pair[1])
        if left != right:
            out.add((min(left, right), max(left, right)))
    return out


def candidate_pairs(prediction: Any, candidates_field: str = "candidates") -> set[tuple[int, int]]:
    """Extract the matched pairs from a per-dataset prediction.

    Parameters
    ----------
    prediction : Any
        Prediction carrying a list of typed candidates
    candidates_field : str
        Output field holding the candidates

    Returns
    -------
    set[tuple[int, int]]
        Record id pairs the model marked as matches
    """
    pairs: set[tuple[int, int]] = set()
    for candidate in getattr(prediction, candidates_field, None) or []:
        if not getattr(candidate, "is_match", False):
            continue
        left = getattr(getattr(candidate, "left", None), "record_id", None)
        right = getattr(getattr(candidate, "right", None), "record_id", None)
        if left is None or right is None or left == right:
            continue
        pairs.add((min(int(left), int(right)), max(int(left), int(right))))
    return pairs


def _record_index(example: dspy.Example, spec: DatasetSignatureSpec) -> dict[int, str]:
    """Map each record id in an example to a short rendering of that record.

    Parameters
    ----------
    example : dspy.Example
        Training example holding both typed sides
    spec : DatasetSignatureSpec
        Typed matching contract for the dataset

    Returns
    -------
    dict[int, str]
        Record id to a compact one-line description
    """
    index: dict[int, str] = {}
    for field in (spec.left_field, spec.right_field):
        for record in getattr(example, field, None) or []:
            values = [
                f"{name}={getattr(record, name)!r}"
                for name in type(record).source_columns()
                if str(getattr(record, name, "") or "")
            ]
            text = ", ".join(values)
            if len(text) > MAX_RECORD_CHARACTERS:
                text = text[:MAX_RECORD_CHARACTERS] + "..."
            index[int(record.record_id)] = text
    return index


def _describe_pairs(pairs: set[tuple[int, int]], index: dict[int, str]) -> str:
    """Render a set of pairs with the records behind them.

    Parameters
    ----------
    pairs : set[tuple[int, int]]
        Pairs to describe
    index : dict[int, str]
        Record id to a compact description

    Returns
    -------
    str
        One indented line per pair, truncated to ``MAX_ENUMERATED_ERRORS``
    """
    lines: list[str] = []
    for left, right in sorted(pairs)[:MAX_ENUMERATED_ERRORS]:
        lines.append(f"  ({left}, {right}):")
        lines.append(f"    {left}: {index.get(left, 'record not in this block')}")
        lines.append(f"    {right}: {index.get(right, 'record not in this block')}")
    remaining = len(pairs) - MAX_ENUMERATED_ERRORS
    if remaining > 0:
        lines.append(f"  ...and {remaining} more")
    return "\n".join(lines)


def make_dataset_metric(spec: DatasetSignatureSpec) -> Callable[..., ScoreWithFeedback]:
    """Build the GEPA feedback metric for one dataset's typed signature.

    The returned function follows GEPA's five-argument protocol and returns a
    score with natural-language feedback. The feedback names every pair that was
    missed and every pair that was invented and prints the two records behind
    each one, because the reflection model can only write a rule about an error
    it can see.

    Parameters
    ----------
    spec : DatasetSignatureSpec
        Typed matching contract for the dataset

    Returns
    -------
    Callable[..., ScoreWithFeedback]
        Metric GEPA can both score and reflect with
    """

    def metric(
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> ScoreWithFeedback:
        """Score one block against its gold pairs and explain the errors.

        Parameters
        ----------
        gold : dspy.Example
            Example carrying the gold pairs for this block
        pred : dspy.Prediction
            Model prediction carrying typed candidates
        trace : Any
            DSPy trace (unused)
        pred_name : str | None
            Predictor name (unused)
        pred_trace : Any
            Predictor trace (unused)

        Returns
        -------
        ScoreWithFeedback
            F1 over match pairs, and feedback enumerating the errors
        """
        gold_set = _normalize(getattr(gold, GOLD_PAIRS_FIELD, None))
        pred_set = candidate_pairs(pred, spec.candidates_field)
        missed = gold_set - pred_set
        extra = pred_set - gold_set

        if not gold_set and not pred_set:
            # F1 is undefined on two empty sets, and returning 0 would punish the
            # only correct answer a block with no true pair admits.
            return ScoreWithFeedback(
                score=1.0,
                feedback="Correct: this block contains no matching pair and none was emitted.",
            )
        score = f1_score(pred_set, gold_set)
        if not missed and not extra:
            return ScoreWithFeedback(
                score=score,
                feedback=(
                    f"Correct: all {len(gold_set)} matching pair(s) in this block were found "
                    "and nothing else was emitted."
                ),
            )

        index = _record_index(gold, spec)
        parts = [
            f"F1={score:.3f} on this block ({len(gold_set)} true pair(s), {len(pred_set)} emitted)."
        ]
        if missed:
            parts.append(
                f"Missed {len(missed)} true pair(s). These records ARE the same entity and "
                f"should have been emitted:\n{_describe_pairs(missed, index)}"
            )
        if extra:
            parts.append(
                f"Emitted {len(extra)} pair(s) that are NOT matches. These records are "
                f"different entities:\n{_describe_pairs(extra, index)}"
            )
        return ScoreWithFeedback(score=score, feedback="\n".join(parts))

    return metric


def train_dataset(
    dataset: str,
    *,
    sizes: SplitSizes | None = None,
    seed: int | None = None,
    train_blocks: int | None = None,
    val_blocks: int | None = None,
    student_model: str | None = None,
    teacher_model: str | None = None,
    auto: str | None = None,
    log_dir: str | None = None,
    output_dir: str | None = None,
    data_dir: str | None = None,
) -> TrainResult:
    """Optimize one benchmark dataset's matching prompt with GEPA.

    Samples disjoint train and validation records, blocks each split on its own
    so no record crosses the boundary, builds typed examples from the blocks that
    contain a gold pair, runs GEPA with the student LM executing and the teacher
    LM reflecting, and writes the winning program where the matcher can load it.

    Parameters
    ----------
    dataset : str
        Benchmark dataset with a per-dataset signature
    sizes : SplitSizes | None
        Record budgets per split. Defaults to this dataset's configured budgets.
    seed : int | None
        Sampling seed. Defaults to config ``optimize.seed``.
    train_blocks : int | None
        Cap on training examples. Defaults to config ``optimize.train_blocks``.
    val_blocks : int | None
        Cap on validation examples. Defaults to config ``optimize.val_blocks``.
    student_model : str | None
        Task LM. Defaults to config ``models.student``.
    teacher_model : str | None
        Reflection LM. Defaults to config ``models.teacher``.
    auto : str | None
        GEPA budget preset. Defaults to config ``optimize.auto``.
    log_dir : str | None
        GEPA checkpoint directory. Defaults to config ``optimize.log_dir``.
    output_dir : str | None
        Where to write the trained program. Defaults to config
        ``optimize.trained_dir``.
    data_dir : str | None
        Benchmark download directory

    Returns
    -------
    TrainResult
        Where the program landed, and the scores and instructions on both sides
        of the run
    """
    spec = get_dataset_spec(dataset)
    seed = seed if seed is not None else int(config.get("optimize.seed", 42))
    sizes = sizes or get_split_sizes(dataset)
    train_cap = train_blocks or int(config.get("optimize.train_blocks", 40))
    val_cap = val_blocks or int(config.get("optimize.val_blocks", 20))
    instructions_before = spec.signature.instructions

    benchmark = BenchmarkDataset.download(dataset, data_dir)
    left_entities, right_entities = benchmark.to_entities()
    entities: list[Entity] = left_entities + right_entities
    splits = sample_random_splits(
        entities,
        benchmark.ground_truth,
        train_records=sizes.train_records,
        val_records=sizes.val_records,
        holdout_records=sizes.holdout_records,
        seed=seed,
    )
    blocker = SemanticBlockingPipeline(
        target_block_size=int(config.get("er.blocking.target_block_size", 30)),
        max_block_size=int(config.get("er.blocking.max_block_size", 100)),
        auto_scale=False,
    )
    train_all, _ = blocker.run(splits.train_records)
    val_all, _ = blocker.run(splits.val_records)
    trainset = blocks_to_dataset_examples(train_all, benchmark.ground_truth, spec)[:train_cap]
    valset = blocks_to_dataset_examples(val_all, benchmark.ground_truth, spec)[:val_cap]
    logger.info(
        f"Training {dataset} on {spec.signature.__name__}: "
        f"{len(splits.train_records)} train records in {len(train_all)} blocks "
        f"({count_gold_pairs(splits.train_records, benchmark.ground_truth)} gold pairs) "
        f"yielded {len(trainset)} usable examples; "
        f"{len(splits.val_records)} val records in {len(val_all)} blocks "
        f"yielded {len(valset)} usable examples"
    )
    if not trainset:
        raise ValueError(
            f"No trainable blocks for {dataset}: blocking put no gold pair inside a block that "
            "holds both sources. Raise benchmarks.train_records or the blocking block size."
        )

    module = cast(dspy.Module, dspy.Predict(spec.signature))
    optimized = optimize_module(
        module,
        trainset=trainset,
        valset=valset or None,
        metric=make_dataset_metric(spec),
        student_model=student_model,
        teacher_model=teacher_model,
        auto=auto,
        log_dir=log_dir,
    )

    path = trained_program_path(dataset, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(path))
    baseline_score, best_score = _validation_scores(optimized)
    instructions_after = (
        predictor_instructions(cast(dspy.Predict, optimized)) or instructions_before
    )
    logger.info(
        f"Trained {dataset}: baseline={baseline_score} best={best_score} "
        f"instructions {len(instructions_before)} -> {len(instructions_after)} characters, "
        f"saved to {path}"
    )
    return TrainResult(
        dataset=dataset,
        signature_name=spec.signature.__name__,
        program_path=str(path),
        train_examples=len(trainset),
        val_examples=len(valset),
        baseline_score=baseline_score,
        best_score=best_score,
        instructions_before=instructions_before,
        instructions_after=instructions_after,
    )


def _validation_scores(optimized: dspy.Module) -> tuple[float | None, float | None]:
    """Pull the seed and winning validation scores out of a GEPA result.

    GEPA scores the seed candidate first, so index zero is the prompt as
    shipped and ``best_idx`` is what the run chose over it.

    Parameters
    ----------
    optimized : dspy.Module
        Program returned by ``GEPA.compile`` with ``track_stats=True``

    Returns
    -------
    tuple[float | None, float | None]
        Baseline and best validation scores, each None when unavailable
    """
    results = getattr(optimized, "detailed_results", None)
    scores = getattr(results, "val_aggregate_scores", None)
    if not scores:
        return None, None
    best_idx = getattr(results, "best_idx", 0)
    return float(scores[0]), float(scores[best_idx])
