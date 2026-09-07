# Fine-Tuning Embedding Models for Entity Resolution

Lessons learned from the [Eridu](https://github.com/Graphlet-AI/eridu) project — an open-source deep fuzzy matching system for multilingual person and company name resolution using representation learning.

## Overview

SERF uses pre-trained sentence-transformer embeddings (`intfloat/multilingual-e5-large`) for semantic blocking. While the pre-trained model works well out of the box, fine-tuning on domain-specific labeled pairs can significantly improve blocking quality — putting more true matches in the same blocks.

Eridu demonstrates the full fine-tuning pipeline: from data preparation through contrastive learning to threshold optimization. The lessons below are directly applicable to SERF's blocking embeddings.

## Key Lessons from Eridu

### 1. Contrastive Learning Is the Right Loss Function

Eridu uses **ContrastiveLoss** from sentence-transformers to fine-tune embeddings. This loss function:

- Pulls matching pairs closer together in embedding space
- Pushes non-matching pairs apart (beyond a configurable margin)
- Works directly with binary labeled pairs (match/no-match)

```python
from sentence_transformers.losses import ContrastiveLoss

loss = ContrastiveLoss(model=model, margin=0.5)
```

**Why not other losses?** Eridu tested `MultipleNegativesRankingLoss` and found it didn't work for name matching. Contrastive loss is more appropriate when you have explicit positive and negative pairs, which is exactly what ER ground truth provides.

### 2. Data Quality Matters More Than Quantity

Eridu trains on **2+ million labeled pairs** from Open Sanctions data, but key findings:

- **Negative pairs are just as important as positive pairs.** The model needs to learn what ISN'T a match to push non-matches apart in embedding space.
- **Group-aware splitting is critical.** Use `GroupShuffleSplit` (not random splitting) to ensure the same base entity name doesn't appear in both train and eval sets. Without this, the model memorizes specific names rather than learning general matching patterns.
- **Resampling helps with large datasets.** When using a fraction of the data (`sample_fraction < 1.0`), resample each epoch to expose the model to different examples.

```python
from sklearn.model_selection import GroupShuffleSplit

# Split by source group to prevent data leakage
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(data, groups=data["source"]))
```

### 3. Corporate Endings Are a Known Hard Problem

Eridu found that fine-tuned models struggle with **corporate suffixes**: "Inc.", "LLC", "GmbH", "Ltd.", etc. Two companies with the same base name but different corporate endings (e.g., "Alpha Capital LLC" vs "Alpha Capital Partners LLC") can be either the same entity or different entities.

**SERF addresses this** with the `cleanco` library in `serf.block.normalize` for stripping corporate suffixes before embedding, but a fine-tuned model that understands these distinctions would be better.

**Recommended approach:** Fine-tune with the [CorpWatch subsidiary dataset](https://www.opensanctions.org/datasets/us_corpwatch/) which contains labeled parent/subsidiary relationships where corporate endings matter.

### 4. Base Model Selection

Eridu's evolution of base models:

| Model                                   | Parameters | Dimensions | Status                                           |
| --------------------------------------- | ---------- | ---------- | ------------------------------------------------ |
| `paraphrase-multilingual-MiniLM-L12-v2` | 118M       | 384        | Original — now obsolete                          |
| `intfloat/multilingual-e5-large`        | 560M       | 1024       | Current — good ROC curve, semantic understanding |
| `Qwen/Qwen3-Embedding-4B`               | 4B         | 2048       | Testing — MTEB #2, needs 16GB GPU                |

**SERF's default** is `intfloat/multilingual-e5-large` — the same model Eridu found works best after fine-tuning. For SERF blocking, the pre-trained version is sufficient; fine-tuning is an optimization.

### 5. Training Configuration That Works

From Eridu's production runs:

```python
# Hyperparameters that work well for name matching
BATCH_SIZE = 1024        # Large batches for stable gradients
EPOCHS = 4-6             # More epochs overfit; early stopping helps
LEARNING_RATE = 3e-5     # Standard for fine-tuning transformers
WEIGHT_DECAY = 0.01      # L2 regularization
WARMUP_RATIO = 0.1       # 10% warmup for learning rate
PATIENCE = 2             # Early stopping after 2 epochs without improvement
MARGIN = 0.5             # Contrastive loss margin
OPTIMIZER = "adafactor"  # Memory-efficient optimizer
```

**Key insights:**

- **FP16 training** reduces memory usage ~2x with minimal quality loss
- **Gradient checkpointing** saves more memory but is broken on Apple MPS
- **Gradient accumulation** (`steps=4`) simulates larger batches on limited GPU memory
- **Early stopping** with `patience=2` prevents overfitting on the relatively small positive pair set

### 6. Evaluation: ROC Curve and Optimal Threshold

After fine-tuning, Eridu:

1. Computes similarity scores on a held-out test set
2. Generates a precision-recall curve across all thresholds
3. Selects the threshold that maximizes F1 score
4. Reports AUC-ROC for overall model quality

```python
from sklearn.metrics import precision_recall_curve, f1_score

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = [f1_score(y_true, y_scores >= t) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
```

**For SERF:** The optimal threshold from fine-tuning could be used as the `similarity_threshold` in the `ERConfig.blocking` section, though SERF's current approach (FAISS IVF clustering) doesn't use a threshold — it assigns every entity to a centroid.

### 7. Weights & Biases Integration

Eridu uses W&B for experiment tracking:

- Loss curves per epoch
- Binary classification metrics (accuracy, F1, precision, recall, AP)
- ROC and PR curves
- Hyperparameter logging
- Test result artifacts

This is valuable for comparing fine-tuning runs across different base models and hyperparameter settings.

## Fine-Tuning for SERF Blocking

### When to Fine-Tune

Fine-tuning the blocking embedding is worthwhile when:

1. **Domain-specific vocabulary**: Your entities use terminology the pre-trained model hasn't seen (medical codes, financial instruments, industry jargon)
2. **Low blocking recall**: Many true matches are landing in different blocks (the pre-trained model doesn't cluster them together)
3. **Multilingual matching**: Entities in different languages/scripts need to cluster together
4. **Corporate endings matter**: You need the model to understand that "Acme Corp" and "Acme Corporation" are likely the same but "Acme Corp" and "Acme Tools Inc" are not

### How to Fine-Tune for SERF

1. **Collect labeled pairs** from your ER ground truth or manual labeling
2. **Format as sentence pairs**: `(entity_name_a, entity_name_b, is_match)`
3. **Fine-tune using Eridu's approach**:

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import ContrastiveLoss

model = SentenceTransformer("intfloat/multilingual-e5-large")
loss = ContrastiveLoss(model=model, margin=0.5)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)
trainer.train()
model.save_pretrained("models/serf-blocking-finetuned")
```

4. **Update SERF config** to use the fine-tuned model:

```yaml
models:
  embedding: "models/serf-blocking-finetuned"
```

### Data Sources for Training Pairs

| Source                     | Type                   | Pairs  | Notes                 |
| -------------------------- | ---------------------- | ------ | --------------------- |
| **Open Sanctions**         | Person + company names | 2M+    | Multilingual, curated |
| **CorpWatch subsidiaries** | Company names          | ~100K  | Corporate endings     |
| **Your ER ground truth**   | Domain-specific        | Varies | Best for your domain  |
| **DBLP-ACM / Abt-Buy**     | Benchmark pairs        | ~5K    | Good for testing      |

### Expected Impact

Based on Eridu's results:

- **Pre-trained model**: ~85% blocking recall (true matches in same block)
- **Fine-tuned model**: ~95%+ blocking recall with tighter blocks
- **Training time**: 1-4 hours on a single GPU for 2M pairs
- **Inference**: No change — same model, same speed

## References

1. [Eridu Repository](https://github.com/Graphlet-AI/eridu) — Full fine-tuning pipeline
2. [Eridu HuggingFace Model](https://huggingface.co/Graphlet-AI/eridu) — Pre-trained model card
3. [Sentence Transformers Training](https://www.sbert.net/docs/training/overview.html) — Framework documentation
4. [ContrastiveLoss](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#contrastiveloss) — Loss function details
5. [Open Sanctions](https://www.opensanctions.org/) — Training data source
