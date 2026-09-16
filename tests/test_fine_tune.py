"""Tests for embedding fine-tuning dataset generation, loss, and training pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from datasets import Dataset

from serf.dspy.types import Entity
from serf.embedding.fine_tune import (
    FineTuneResult,
    LabeledPair,
    entity_to_text,
    generate_labeled_pairs,
    pairs_to_dataset,
    run_fine_tune,
)


def _entity(
    entity_id: int, name: str, desc: str = "", attrs: dict[str, str] | None = None
) -> Entity:
    return Entity(
        id=entity_id,
        name=name,
        description=desc,
        entity_type="product",
        attributes=attrs or {"name": name, "desc": desc},
    )


def test_entity_to_text() -> None:
    e = _entity(1, "ThinkPad X1", "Laptop 16GB", {"name": "ThinkPad X1", "price": "1200"})
    assert entity_to_text(e, "name") == "ThinkPad X1"
    assert entity_to_text(e, "combined") == "ThinkPad X1 Laptop 16GB"
    json_text = entity_to_text(e, "json")
    assert '"name": "ThinkPad X1"' in json_text
    assert '"price": "1200"' in json_text


def test_generate_labeled_pairs_with_positives_and_negatives() -> None:
    # 2 left entities, 2 right entities (right offset is 100000)
    left1 = _entity(0, "Apple iPhone 13")
    left2 = _entity(1, "Samsung Galaxy S22")
    right1 = _entity(100000, "Apple iPhone 13 128GB")
    right2 = _entity(100001, "Google Pixel 6")

    entities = [left1, left2, right1, right2]
    # left 0 matches right 100000
    ground_truth = {(0, 100000)}

    pairs = generate_labeled_pairs(
        entities, ground_truth, negative_ratio=2.0, strategy="name", seed=42
    )

    positives = [p for p in pairs if p.label == 1.0]
    negatives = [p for p in pairs if p.label == 0.0]

    assert len(positives) == 1
    assert positives[0].sentence1 == "Apple iPhone 13"
    assert positives[0].sentence2 == "Apple iPhone 13 128GB"
    assert positives[0].id1 == 0
    assert positives[0].id2 == 100000

    assert len(negatives) >= 1
    for neg in negatives:
        assert (neg.id1, neg.id2) != (0, 100000)


def test_generate_labeled_pairs_empty_when_no_entities_or_positives() -> None:
    pairs = generate_labeled_pairs([], set())
    assert pairs == []

    left = [_entity(0, "Apple iPhone 13")]
    pairs_no_right = generate_labeled_pairs(left, set())
    assert pairs_no_right == []


def test_pairs_to_dataset() -> None:
    pairs = [
        LabeledPair(sentence1="a", sentence2="b", label=1.0, id1=1, id2=2),
        LabeledPair(sentence1="c", sentence2="d", label=0.0, id1=3, id2=4),
    ]
    ds = pairs_to_dataset(pairs)
    assert isinstance(ds, Dataset)
    assert len(ds) == 2
    assert ds["sentence1"] == ["a", "c"]
    assert ds["sentence2"] == ["b", "d"]
    assert ds["label"] == [1.0, 0.0]


@patch("serf.embedding.fine_tune.evaluate_blocking")
@patch("serf.embedding.fine_tune.SentenceTransformerTrainer")
@patch("serf.embedding.fine_tune.SentenceTransformer")
@patch("serf.embedding.fine_tune.BenchmarkDataset.download")
def test_run_fine_tune_flow(
    mock_download: MagicMock,
    mock_sbert_cls: MagicMock,
    mock_trainer_cls: MagicMock,
    mock_eval_blocking: MagicMock,
    tmp_path: Path,
) -> None:
    # Set up mock benchmark dataset
    mock_ds = MagicMock()
    left = [_entity(i, f"item left {i}") for i in range(10)]
    right = [_entity(100000 + i, f"item right {i}") for i in range(10)]
    mock_ds.to_entities.return_value = (left, right)
    mock_ds.ground_truth = {(i, 100000 + i) for i in range(10)}
    mock_download.return_value = mock_ds

    # Set up mock SBERT model and evaluator
    mock_model = MagicMock()
    mock_model.similarity_fn_name = "cosine"
    mock_model.encode.side_effect = lambda texts, **kwargs: np.array(
        [[0.1, 0.2]] * len(texts), dtype=np.float32
    )
    mock_sbert_cls.return_value = mock_model

    # Set up mock blocking evaluation
    raw_blocking = MagicMock()
    raw_blocking.blocking_recall = 0.80
    tuned_blocking = MagicMock()
    tuned_blocking.blocking_recall = 0.88
    mock_eval_blocking.side_effect = [raw_blocking, tuned_blocking]

    # Set up mock trainer
    mock_trainer = MagicMock()
    mock_trainer_cls.return_value = mock_trainer

    result = run_fine_tune(
        "dblp-acm",
        model_name="BAAI/bge-small-en-v1.5",
        output_dir=str(tmp_path / "model_out"),
        epochs=1,
        batch_size=8,
        learning_rate=1e-5,
        margin=0.5,
        loss_type="contrastive",
        negative_ratio=2.0,
        strategy="name",
        seed=42,
        train_records=10,
        val_records=5,
        holdout_records=5,
    )

    assert isinstance(result, FineTuneResult)
    assert result.dataset == "dblp-acm"
    assert result.base_model == "BAAI/bge-small-en-v1.5"
    assert result.loss_name == "contrastive"
    assert result.raw_blocking_recall == 0.80
    assert result.tuned_blocking_recall == 0.88
    mock_trainer.train.assert_called_once()
    mock_model.save_pretrained.assert_called_once()
