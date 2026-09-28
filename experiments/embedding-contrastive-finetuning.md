# Contrastive Embedding Fine-Tuning for Entity Resolution

## Motivation & Architecture

In entity resolution, semantic blocking recall establishes the ceiling on end-to-end recall:
any genuine matching pair that FAISS clustering fails to group into a shared block is permanently
lost to the downstream LLM matcher. Profiling across the Leipzig and DeepMatcher benchmarks revealed
that missed pairs during blocking accounted for over 59% of all missed pairs.

Following patterns from [Graphlet-AI/eridu](https://github.com/Graphlet-AI/eridu), SERF applies
contrastive fine-tuning to adapt sentence-transformer representations to entity matching without
introducing custom classifier heads or multi-stage neural architectures. By keeping the interface
as a standard `SentenceTransformer`, the fine-tuned model directly slots into SERF's FAISS
blocking pipeline (`SemanticBlockingPipeline`).

### Training Pipeline (`serf fine-tune`)

1. **Split Isolation (Rule 1 Enforced)**:
   - Disjoint match-group partition via `sample_random_splits`.
   - In-trainer evaluation and checkpoint selection (`load_best_model_at_end`) strictly evaluate on
     the `val` split.
   - Holdout evaluation is executed strictly before training (baseline) and after training (final).
2. **Negative Mining**:
   - Generates positive pairs (`label=1.0`) from ground truth within each split.
   - Mines cross-source negative pairs (`label=0.0`) between left and right entities not in ground
     truth at configurable `negative_ratio` (default `3.0`).
3. **Loss Functions**:
   - `contrastive`: `ContrastiveLoss(model, margin=0.5)`
   - `online_contrastive`: `OnlineContrastiveLoss(model, margin=0.5)`
   - `mnrl`: `MultipleNegativesRankingLoss(model)`
   - `cosent`: `CoSENTLoss(model)`
4. **Instruction Prompts**:
   - Base model: `BAAI/bge-small-en-v1.5` (33M params).
   - `models.embedding_prompt` (`Represent this sentence for searching relevant passages: `) is
     consistently passed during training, evaluation, and blocking inference.

---

## Benchmark Results on Holdout Split

### DBLP-ACM (Bibliographic Benchmark)

- **Command**: `serf fine-tune dblp-acm --epochs 2 --batch-size 32 --output-dir data/models/fine-tuned-dblp-acm-bge-small-contrastive --seed 42`
- **Artifact Log**: `/opt/cursor/artifacts/fine_tune_dblp_acm.log`
- **Data Splits**: 4,324 train pairs (1,081 gold), 2,080 val pairs (520 gold), 2,068 holdout pairs (517 gold)
- **Training Wall Clock**: 250.3 seconds on 4-core CPU

| Metric | Raw Base (`bge-small-en-v1.5`) | Fine-Tuned (`ContrastiveLoss`) | Delta |
|---|---|---|---|
| Holdout Cosine Accuracy | 0.9990 | **1.0000** | +0.0010 |
| Holdout Cosine F1 | 0.9981 | **1.0000** | +0.0019 |
| Holdout Cosine Average Precision | 0.9999 | **1.0000** | +0.0001 |
| **Holdout Blocking Recall** | **0.9942** (514/517) | **0.9981** (516/517) | **+0.0039** |
| Missed Gold Pairs in Blocking | 3 | **1** | **-66.7%** |
| Matcher Pairs to Compare | 26,735 | **19,403** | **-27.4%** |
| Max Block Size | 100 | **62** | **-38.0%** |
| Reduction Ratio | 0.9558 | **0.9679** | +0.0121 |

**Analysis**:
On DBLP-ACM, contrastive fine-tuning reduces missed gold pairs by two-thirds (from 3 down to 1),
reaching 0.9981 blocking recall on holdout. Furthermore, because entity representations are drawn
closer to their true counterparts and further from negatives, clusters become significantly tighter:
the maximum block size drops from 100 to 62, and the total pairwise comparisons fed to the matcher
drops by 27.4% (from 26,735 to 19,403), directly saving downstream LLM inference spend.

---

### Abt-Buy (Product Catalog Benchmark)

- **Command**: `serf fine-tune abt-buy --epochs 2 --batch-size 32 --output-dir data/models/fine-tuned-abt-buy-bge-small-contrastive --seed 42`
- **Artifact Log**: `/opt/cursor/artifacts/fine_tune_abt_buy.log`
- **Data Splits**: 1,348 train pairs (337 gold), 1,520 val pairs (380 gold), 1,520 holdout pairs (380 gold)
- **Training Wall Clock**: 106.6 seconds on 4-core CPU

| Metric | Raw Base (`bge-small-en-v1.5`) | Fine-Tuned (`ContrastiveLoss`) | Delta |
|---|---|---|---|
| Holdout Cosine Accuracy | 0.9895 | **0.9908** | +0.0013 |
| Holdout Cosine F1 | 0.9801 | **0.9819** | +0.0018 |
| Holdout Cosine Recall | 0.9921 | **0.9974** | +0.0053 |
| Holdout Cosine Average Precision | 0.9958 | **0.9959** | +0.0001 |
| **Holdout Blocking Recall** | **0.8947** (340/380) | **0.9105** (346/380) | **+0.0158** |
| Missed Gold Pairs in Blocking | 40 | **34** | **-15.0%** |
| Matcher Pairs to Compare | 13,875 | 14,094 | +1.5% |
| Reduction Ratio | 0.9506 | 0.9498 | -0.0008 |

**Analysis**:
Abt-Buy represents a challenging product deduplication task with varied brand naming, model numbers,
and truncated descriptions. Fine-tuning lifts holdout blocking recall from 0.8947 to 0.9105
(+1.58 percentage points), recovering 6 previously unreachable gold pairs in just 2 epochs of
CPU training.
