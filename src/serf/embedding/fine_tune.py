"""Contrastive fine-tuning for sentence transformers on entity resolution datasets.

Improves entity representation learning using contrastive learning on labeled
entity pairs (matches vs non-matches), inspired by Eridu and sentence-transformers.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.evaluation import BinaryClassificationEvaluator
from sentence_transformers.sentence_transformer.losses import (
    ContrastiveLoss,
    CoSENTLoss,
    MultipleNegativesRankingLoss,
    OnlineContrastiveLoss,
)
from sentence_transformers.sentence_transformer.model_card import SentenceTransformerModelCardData
from sentence_transformers.sentence_transformer.training_args import (
    SentenceTransformerTrainingArguments,
)

from serf.block.embeddings import get_torch_device
from serf.config import config
from serf.dspy.types import Entity
from serf.eval.benchmarks import RIGHT_ID_OFFSET, BenchmarkDataset
from serf.eval.blocking_sweep import evaluate_blocking
from serf.eval.splits import get_split_sizes, sample_random_splits
from serf.logs import get_logger

logger = get_logger(__name__)

# Silence noisy external downloaders and loggers during training
for _noisy in [
    "httpx",
    "httpcore",
    "huggingface_hub",
    "urllib3",
    "filelock",
    "fsspec",
]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)


@dataclass
class LabeledPair:
    """One labeled pair for contrastive fine-tuning."""

    sentence1: str
    sentence2: str
    label: float  # 1.0 for match, 0.0 for non-match
    id1: int
    id2: int


@dataclass
class FineTuneResult:
    """Results from contrastive embedding fine-tuning."""

    dataset: str
    base_model: str
    output_dir: str
    loss_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    train_pairs: int
    val_pairs: int
    holdout_pairs: int
    raw_eval_metrics: dict[str, Any]
    tuned_eval_metrics: dict[str, Any]
    raw_blocking_recall: float
    tuned_blocking_recall: float


def entity_to_text(entity: Entity, strategy: str = "name") -> str:
    """Render an entity as text for embedding.

    Parameters
    ----------
    entity : Entity
        The entity to render
    strategy : str
        "name" for entity name, "json" for attributes JSON representation, or
        "combined" for name + description

    Returns
    -------
    str
        Text representation
    """
    if strategy == "json":
        return entity.json_for_embedding()
    if strategy == "combined":
        parts = [entity.text_for_embedding()]
        if entity.description:
            parts.append(entity.description)
        return " ".join(parts).strip()
    return entity.text_for_embedding()


def generate_labeled_pairs(
    entities: list[Entity],
    ground_truth: set[tuple[int, int]],
    *,
    negative_ratio: float = 3.0,
    strategy: str = "name",
    seed: int = 42,
    max_negatives: int | None = None,
) -> list[LabeledPair]:
    """Generate positive and hard/random negative pairs from entities and ground truth.

    Parameters
    ----------
    entities : list[Entity]
        Entity records available in this split
    ground_truth : set[tuple[int, int]]
        Gold matching pairs (left_id, right_id)
    negative_ratio : float
        Ratio of negative pairs to positive pairs (e.g. 3.0 means 3 negatives per positive)
    strategy : str
        Entity text rendering strategy ("name", "json", "combined")
    seed : int
        RNG seed for negative sampling
    max_negatives : int | None
        Optional maximum cap on negative pairs

    Returns
    -------
    list[LabeledPair]
        Shuffled list of positive and negative pairs
    """
    rng = random.Random(seed)
    by_id = {e.id: e for e in entities}
    left_entities = [e for e in entities if e.id < RIGHT_ID_OFFSET]
    right_entities = [e for e in entities if e.id >= RIGHT_ID_OFFSET]

    if not left_entities or not right_entities:
        return []

    # 1. Collect valid positive pairs in this entity set
    positives: list[LabeledPair] = []
    gold_set: set[tuple[int, int]] = set()
    for l_id, r_id in ground_truth:
        if l_id in by_id and r_id in by_id:
            gold_set.add((l_id, r_id))
            e1 = by_id[l_id]
            e2 = by_id[r_id]
            t1 = entity_to_text(e1, strategy)
            t2 = entity_to_text(e2, strategy)
            if t1 and t2:
                positives.append(
                    LabeledPair(sentence1=t1, sentence2=t2, label=1.0, id1=l_id, id2=r_id)
                )

    if not positives:
        return []

    num_negatives = int(len(positives) * negative_ratio)
    if max_negatives is not None:
        num_negatives = min(num_negatives, max_negatives)

    # 2. Mine hard/lexical negatives (random cross-source pairs not in ground truth)
    negatives: list[LabeledPair] = []
    seen_neg_pairs: set[tuple[int, int]] = set()
    max_attempts = num_negatives * 10
    attempts = 0

    while len(negatives) < num_negatives and attempts < max_attempts:
        attempts += 1
        l_ent = rng.choice(left_entities)
        r_ent = rng.choice(right_entities)
        pair_key = (l_ent.id, r_ent.id)
        if pair_key in gold_set or pair_key in seen_neg_pairs:
            continue
        seen_neg_pairs.add(pair_key)
        t1 = entity_to_text(l_ent, strategy)
        t2 = entity_to_text(r_ent, strategy)
        if t1 and t2:
            negatives.append(
                LabeledPair(
                    sentence1=t1,
                    sentence2=t2,
                    label=0.0,
                    id1=l_ent.id,
                    id2=r_ent.id,
                )
            )

    all_pairs = positives + negatives
    rng.shuffle(all_pairs)
    return all_pairs


def pairs_to_dataset(pairs: list[LabeledPair]) -> Dataset:
    """Convert a list of LabeledPair into a Hugging Face Dataset.

    Parameters
    ----------
    pairs : list[LabeledPair]
        Labeled pairs

    Returns
    -------
    Dataset
        Hugging Face Dataset with keys sentence1, sentence2, label
    """
    return Dataset.from_dict(
        {
            "sentence1": [p.sentence1 for p in pairs],
            "sentence2": [p.sentence2 for p in pairs],
            "label": [float(p.label) for p in pairs],
        }
    )


def run_fine_tune(
    dataset_name: str,
    *,
    model_name: str | None = None,
    output_dir: str | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    margin: float = 0.5,
    loss_type: str = "contrastive",
    negative_ratio: float = 3.0,
    strategy: str = "name",
    seed: int = 42,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    device: str | None = None,
    eval_steps: int = 50,
    train_records: int | None = None,
    val_records: int | None = None,
    holdout_records: int | None = None,
) -> FineTuneResult:
    """Fine-tune a sentence-transformer model on ER benchmark data using contrastive learning.

    Parameters
    ----------
    dataset_name : str
        Benchmark dataset name (e.g. "dblp-acm", "abt-buy")
    model_name : str | None
        Base sentence transformer model (defaults to config models.embedding or BAAI/bge-small-en-v1.5)
    output_dir : str | None
        Directory to save fine-tuned model and evaluation artifacts
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size per device
    learning_rate : float
        Optimizer learning rate
    margin : float
        Contrastive loss margin
    loss_type : str
        "contrastive", "online_contrastive", "mnrl", or "cosent"
    negative_ratio : float
        Ratio of negative pairs to positive pairs in training/val
    strategy : str
        Text rendering strategy ("name", "json", "combined")
    seed : int
        RNG seed for reproducibility
    warmup_ratio : float
        Learning rate warmup ratio
    weight_decay : float
        L2 regularization weight decay
    device : str | None
        Torch device ("cuda", "mps", "cpu")
    eval_steps : int
        Number of steps between evaluation and checkpointing
    train_records : int | None
        Optional override for train split record count
    val_records : int | None
        Optional override for val split record count
    holdout_records : int | None
        Optional override for holdout split record count

    Returns
    -------
    FineTuneResult
        Complete summary of the fine-tuning run
    """
    device = device or get_torch_device()
    model_name = model_name or str(config.get("models.embedding", "BAAI/bge-small-en-v1.5"))

    if output_dir is None:
        clean_name = model_name.replace("/", "-")
        output_dir = f"data/models/fine-tuned-{dataset_name}-{clean_name}"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark dataset
    benchmark_data = BenchmarkDataset.download(dataset_name)
    left_entities, right_entities = benchmark_data.to_entities()
    all_entities = left_entities + right_entities

    # 2. Split dataset cleanly by match groups
    try:
        configured_sizes = get_split_sizes(dataset_name)
    except Exception:
        configured_sizes = None

    effective_train_records = train_records
    effective_val_records = val_records
    effective_holdout_records = holdout_records

    if configured_sizes is not None:
        if effective_train_records is None:
            effective_train_records = configured_sizes.train_records
        if effective_val_records is None:
            effective_val_records = configured_sizes.val_records
        if effective_holdout_records is None:
            effective_holdout_records = configured_sizes.holdout_records

    splits = sample_random_splits(
        all_entities,
        benchmark_data.ground_truth,
        train_records=effective_train_records,
        val_records=effective_val_records,
        holdout_records=effective_holdout_records,
        seed=seed,
    )

    logger.info(
        f"Splits for {dataset_name}: train={len(splits.train_records)}, "
        f"val={len(splits.val_records)}, holdout={len(splits.holdout_records)}"
    )

    # 3. Generate labeled pairs for train, val, and holdout
    train_pairs = generate_labeled_pairs(
        splits.train_records,
        benchmark_data.ground_truth,
        negative_ratio=negative_ratio,
        strategy=strategy,
        seed=seed,
    )
    val_pairs = generate_labeled_pairs(
        splits.val_records,
        benchmark_data.ground_truth,
        negative_ratio=negative_ratio,
        strategy=strategy,
        seed=seed + 1,
    )
    holdout_pairs = generate_labeled_pairs(
        splits.holdout_records,
        benchmark_data.ground_truth,
        negative_ratio=negative_ratio,
        strategy=strategy,
        seed=seed + 2,
    )

    if not train_pairs:
        raise ValueError(f"No positive pairs found in training split for dataset {dataset_name}")

    logger.info(
        f"Generated pairs: train={len(train_pairs)}, val={len(val_pairs)}, "
        f"holdout={len(holdout_pairs)}"
    )

    train_ds = pairs_to_dataset(train_pairs)
    val_ds = pairs_to_dataset(val_pairs) if val_pairs else train_ds
    holdout_ds = pairs_to_dataset(holdout_pairs) if holdout_pairs else val_ds

    prompt = str(config.get("models.embedding_prompt", ""))

    # 4. Load base model
    logger.info(f"Loading base model {model_name} on {device}")
    model = SentenceTransformer(
        model_name,
        device=device,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{model_name}-finetuned-{dataset_name}",
        ),
    )

    # 5. Build holdout evaluator for pre- and post-training measurement
    # Prefix prompt if present to stay aligned with retrieval/blocking embeddings
    holdout_s1 = (
        [(prompt + s if prompt else s) for s in holdout_ds["sentence1"]]
        if len(holdout_ds) > 0
        else []
    )
    holdout_s2 = (
        [(prompt + s if prompt else s) for s in holdout_ds["sentence2"]]
        if len(holdout_ds) > 0
        else []
    )
    holdout_labels = holdout_ds["label"] if len(holdout_ds) > 0 else []

    holdout_evaluator = (
        BinaryClassificationEvaluator(
            sentences1=holdout_s1,
            sentences2=holdout_s2,
            labels=holdout_labels,
            name=f"{dataset_name}-holdout",
        )
        if len(holdout_labels) > 0
        else None
    )

    # Build validation evaluator for trainer (checkpoint selection on validation, NEVER holdout)
    val_s1 = [(prompt + s if prompt else s) for s in val_ds["sentence1"]] if len(val_ds) > 0 else []
    val_s2 = [(prompt + s if prompt else s) for s in val_ds["sentence2"]] if len(val_ds) > 0 else []
    val_labels = val_ds["label"] if len(val_ds) > 0 else []

    val_evaluator = (
        BinaryClassificationEvaluator(
            sentences1=val_s1,
            sentences2=val_s2,
            labels=val_labels,
            name=f"{dataset_name}-val",
        )
        if len(val_labels) > 0
        else None
    )

    logger.info("Evaluating raw baseline model on holdout split...")
    raw_eval: dict[str, Any] = {}
    if holdout_evaluator is not None:
        raw_eval = holdout_evaluator(model)
        logger.info(f"Raw model holdout evaluation: {raw_eval}")

    # Also evaluate blocking recall baseline on holdout (restricted to holdout records)
    holdout_ids = {e.id for e in splits.holdout_records}
    holdout_ground_truth = {
        (left_id, right_id)
        for left_id, right_id in benchmark_data.ground_truth
        if left_id in holdout_ids and right_id in holdout_ids
    }

    target_block = int(config.get("er.blocking.target_block_size", 30))
    max_block = int(config.get("er.blocking.max_block_size", 100))
    raw_blocking = evaluate_blocking(
        splits.holdout_records,
        holdout_ground_truth,
        dataset=dataset_name,
        model_name=model_name,
        prompt=prompt,
        target_block_size=target_block,
        max_block_size=max_block,
        strategy=strategy,
    )
    raw_recall = raw_blocking.blocking_recall
    logger.info(f"Raw model holdout blocking recall: {raw_recall:.4f}")

    # 6. Configure loss
    if loss_type == "mnrl":
        pos_train_pairs = [p for p in train_pairs if p.label == 1.0]
        train_ds = pairs_to_dataset(pos_train_pairs)
        train_loss = MultipleNegativesRankingLoss(model)
    elif loss_type == "online_contrastive":
        train_loss = OnlineContrastiveLoss(model=model, margin=margin)
    elif loss_type == "cosent":
        train_loss = CoSENTLoss(model=model)
    else:
        train_loss = ContrastiveLoss(model=model, margin=margin)

    # 7. Configure TrainingArguments
    use_fp16 = torch.cuda.is_available()
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_ratio,  # Transformers 5+ float warmup ratio
        weight_decay=weight_decay,
        fp16=use_fp16,
        eval_strategy="steps" if len(val_ds) > 0 else "no",
        eval_steps=eval_steps,
        save_strategy="steps" if len(val_ds) > 0 else "no",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=len(val_ds) > 0,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=max(1, eval_steps // 2),
        report_to="none",
        prompts=prompt if prompt else None,
    )

    # 8. Train (evaluated on val_ds, NEVER on holdout_ds)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        loss=train_loss,
        evaluator=val_evaluator if len(val_ds) > 0 else None,
    )

    logger.info(f"Starting fine-tuning for {epochs} epochs with {loss_type} loss...")
    trainer.train()

    # 9. Save final fine-tuned model
    model.save_pretrained(str(out_path))
    logger.info(f"Saved fine-tuned model to {out_path}")

    # 10. Final evaluation on holdout
    tuned_eval: dict[str, Any] = {}
    if holdout_evaluator is not None:
        tuned_eval = holdout_evaluator(model)
        logger.info(f"Fine-tuned model holdout evaluation: {tuned_eval}")

    # Evaluate blocking recall with fine-tuned model
    tuned_blocking = evaluate_blocking(
        splits.holdout_records,
        holdout_ground_truth,
        dataset=dataset_name,
        model_name=str(out_path),
        prompt=prompt,
        target_block_size=target_block,
        max_block_size=max_block,
        strategy=strategy,
    )
    tuned_recall = tuned_blocking.blocking_recall
    logger.info(f"Fine-tuned model holdout blocking recall: {tuned_recall:.4f}")

    # Metric evaluation & verdict line
    score = float(tuned_eval.get(f"{dataset_name}-holdout_cosine_f1", 0.0))
    baseline_eval = float(raw_eval.get(f"{dataset_name}-holdout_cosine_f1", 0.0))
    delta = score - baseline_eval
    verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
    logger.info(
        f"VERDICT: {verdict} | score={score:.4f} | baseline={baseline_eval:.4f} | delta={delta:+.4f} "
        f"| holdout_blocking_recall={tuned_recall:.4f} (raw={raw_recall:.4f}, delta={tuned_recall - raw_recall:+.4f})"
    )

    return FineTuneResult(
        dataset=dataset_name,
        base_model=model_name,
        output_dir=str(out_path),
        loss_name=loss_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_pairs=len(train_pairs),
        val_pairs=len(val_pairs),
        holdout_pairs=len(holdout_pairs),
        raw_eval_metrics=raw_eval,
        tuned_eval_metrics=tuned_eval,
        raw_blocking_recall=raw_recall,
        tuned_blocking_recall=tuned_recall,
    )
