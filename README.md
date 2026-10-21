# SERF: Semantic Entity Resolution Framework

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)

SERF is an open-source Python framework for **semantic entity resolution** (deduplication and record linkage). It combines vector embeddings and FAISS for fast candidate blocking with DSPy-orchestrated Large Language Models (LLMs) for entity matching, schema alignment, and canonical record merging.

<div align="center">
  <img src="assets/entity_resolution.png" alt="Stages of entity resolution: blocking, matching, merging" width="600px" />
  <p><em>Source: <a href="https://medium.com/data-science/entity-resolution-identifying-real-world-entities-in-noisy-data-3e8c59f4f41c">Entity Resolution: Identifying Real-World Entities in Noisy Data</a></em></p>
</div>

---

## Why SERF?

Entity resolution across heterogeneous data sources is notoriously difficult:

- **Quadratic complexity ($O(N^2)$)** makes naive pairwise comparisons intractable at scale.
- **Rule-based heuristics and fuzzy joins** are brittle and require endless manual tuning.
- **Pure embedding similarity** works for coarse semantic grouping but fails to distinguish fine-grained differences (e.g., product model numbers, storage capacities, edition years).

SERF solves this by combining the speed of dense vector search with the reasoning capabilities of LLMs in a modular, multi-iteration pipeline:

1. **Semantic Blocking**: Dense embeddings (`BAAI/bge-small-en-v1.5`) and FAISS IVF index cluster records into small candidate blocks, filtering out 99%+ of non-matches without quadratic blowup.
2. **Whole-Block Partition Matching**: Rather than evaluating pairwise links independently, the LLM views all records in a candidate block simultaneously and outputs entity partitions (clusters). This prevents contradictory pairwise decisions (e.g., $A=B$, $B=C$, but $A \neq C$).
3. **Auditable Canonical Records**: Merged entities receive newly minted UUIDs while preserving full transitive lineage (`source_uuids`). Multi-valued attributes are deduplicated using type-aware semantic policies.
4. **Multi-Round Iterative Resolution**: Resolved entities are re-blocked and matched over multiple iterations. Records separated in the first round of coarse blocking get additional opportunities to merge.
5. **Prompt Optimization with GEPA**: Self-improving matching signatures powered by DSPy's Generative Prompt Optimization (GEPA), with strict train/validation/holdout data separation.

---

## Architecture & Data Flow

```text
Raw Records
    │
    ▼
[ 1. Semantic Blocking ] ──> Dense vector embeddings + FAISS IVF
    │                        Groups records into manageable blocks (e.g., 30–50 records)
    ▼
[ 2. Partition Matching ] ──> DSPy + LLM (XMLAdapter)
    │                         Partitions each block into resolved entity clusters
    ▼
[ 3. Canonical Merging ]  ──> Mints new UUIDs, records complete source lineage,
    │                         and deduplicates attributes by field semantics
    ▼
[ 4. Multi-Round Loop ]   ──> Re-blocks surviving canonical records for N iterations
    │
    ▼
Resolved Knowledge Graph
```

| Component           | Technology                                                                             |
| ------------------- | -------------------------------------------------------------------------------------- |
| **Package Manager** | [uv](https://docs.astral.sh/uv/)                                                       |
| **Data Processing** | [PySpark 4.x](https://spark.apache.org/) (optional distributed ETL & graph operations) |
| **LLM Framework**   | [DSPy](https://dspy.ai/) with `XMLAdapter` for structured XML/Pydantic outputs         |
| **Embeddings**      | `BAAI/bge-small-en-v1.5` via [Sentence-Transformers](https://sbert.net/)               |
| **Vector Indexing** | [FAISS](https://github.com/facebookresearch/faiss) (`IndexIVFFlat`)                    |
| **Prompt Tuning**   | DSPy GEPA (student/teacher reflection optimization)                                    |
| **Code Quality**    | Ruff (linting & formatting), zuban / mypy (static type checking)                       |

---

## The Resolved Record Data Model

Entity resolution merges multiple input records into unified canonical entities. To ensure data integrity, auditability, and no information loss, SERF enforces strict record conventions:

1. **UUID-Native Identity**: Every input entity has a UUID. When records merge, SERF mints a new UUID and records all constituent record UUIDs—including transitively merged ones—in `source_uuids`. Unmatched records preserve their original identity.
2. **Multi-Valued Fields as Lists**: In a canonical record, non-identifier fields hold lists ordered by completeness (fullest value first). If two sources supply valid information (e.g., different phone numbers or addresses), both are retained rather than forcing an arbitrary lossy selection.
3. **Type-Aware Deduplication**: Field values are merged based on semantic field types:

| Field Type                                       | Deduplication Strategy             | Description                                                                             |
| ------------------------------------------------ | ---------------------------------- | --------------------------------------------------------------------------------------- |
| `name`, `address`                                | Fuzzy similarity (`dedupe: fuzzy`) | Near-duplicate spellings collapse to the most complete variant.                         |
| `identifier`, `email`, `phone`, `url`, `numeric` | Normalization (`dedupe: exact`)    | Values normalize (e.g., lowercasing, stripping punctuation) before exact deduplication. |
| `text`                                           | Preserved (`dedupe: none`)         | Freeform text (e.g., long descriptions) is never collapsed.                             |

### Example: Before and After Merging

```python
# Input records
[
    {"uuid": "11111111-...", "name": "Russell Jurney", "state": "CA"},
    {"uuid": "22222222-...", "name": "Russell H. Jurney", "state": "CA", "source_uuids": ["33333333-..."]},
    {"uuid": "44444444-...", "name": "Bob Dorf", "state": "NY"}
]

# Output canonical records
[
    {
        "uuid": "99999999-...",  # Newly minted UUID
        "name": ["Russell H. Jurney"],  # Fuller variant retained
        "state": ["CA"],  # Duplicate values deduplicated
        "source_uuids": ["11111111-...", "22222222-...", "33333333-..."]  # Full lineage preserved
    },
    {
        "uuid": "44444444-...",  # Unmatched record retained unchanged
        "name": ["Bob Dorf"],
        "state": ["NY"],
        "source_uuids": []
    }
]
```

### Declaring Schemas

You can define explicit entity schemas and merge policies in YAML:

```yaml
entity: Person
description: Cleaned person entity with contact details.
fields:
  - name: name
    type: name
    description: Full person name, given name first.
    required: true
  - name: state
    type: address
    description: Two-letter state abbreviation.
    merge:
      dedupe: exact # Exact matching prevents merging "CA" and "GA"
  - name: email
    type: email
    description: Primary email address.
  - name: bio
    type: text
    description: Freeform biography, preserved as array.
```

Inspect schemas and test merge policies directly via the CLI:

```bash
# View resolved field policies and generated Pydantic models
serf schema schemas/person.yml

# Dry-run merging records against a schema
serf schema schemas/person.yml --merge records.json
```

---

## Quick Start

### Installation

Prerequisites:

- **Python 3.12+**
- **Java 11, 17, or 21** (required for PySpark operations)

Install dependencies using `uv`:

```bash
git clone https://github.com/Graphlet-AI/serf.git
cd serf
uv sync --extra dev
```

### Configure API Keys

Set your LLM credentials in a `.env` file or export them in your shell:

```bash
# Vertex AI / Google Cloud
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
export VERTEX_AI_TOKEN="your-vertex-token"

# Or Gemini API
export GEMINI_API_KEY="your-gemini-api-key"
```

### Quick Python Example

```python
import asyncio
from serf.block.pipeline import SemanticBlockingPipeline
from serf.match.matcher import EntityMatcher
from serf.merge.canonical import canonicalize_groups

async def main():
    records = [
        {"id": 1, "name": "Apple iPhone 14 Pro 128GB Space Black", "price": "999"},
        {"id": 2, "name": "iPhone 14 Pro, 128 GB, Black", "price": "999.00"},
        {"id": 3, "name": "Samsung Galaxy S23 Ultra 256GB", "price": "1199"}
    ]

    # 1. Block records semantically
    blocking = SemanticBlockingPipeline(target_block_size=50)
    blocks, metrics = blocking.run(records)

    # 2. Match entities in blocks using LLM
    matcher = EntityMatcher()
    resolutions = await matcher.resolve_blocks(blocks)

    # 3. Canonicalize matches into unified records
    canonical_entities = canonicalize_groups(resolutions.groups, records)
    print(canonical_entities)

asyncio.run(main())
```

---

## CLI Reference

SERF provides a comprehensive Click CLI for running pipelines, evaluating benchmarks, and tuning prompts.

### 1. Core Entity Resolution Pipeline

```bash
# Run the complete end-to-end ER pipeline on your data
serf resolve --input data/entities.csv --output data/resolved/ --iteration 3

# Run individual pipeline stages
serf block --input data/entities.csv --output data/blocks/ --method semantic
serf match --input data/blocks/ --output data/matches/
serf eval --input data/matches/
serf edges --input data/graph_edges.parquet --output data/resolved_edges/
```

### 2. Benchmarks & Evaluation

SERF includes built-in loaders and evaluation runners for standard Leipzig entity resolution benchmarks (`dblp-acm`, `abt-buy`, `walmart-amazon`, `amazon-google`, `dblp-scholar`):

```bash
# Download a benchmark dataset
serf download --dataset dblp-acm

# Run end-to-end benchmark evaluation
serf benchmark --dataset dblp-acm --output data/results/

# Run benchmark with multi-iteration re-blocking (default is 3 iterations)
serf benchmark --dataset dblp-acm --max-iterations 3

# Benchmark with union blocking (name + JSON representation for high product recall)
serf benchmark --dataset amazon-google --blocking-strategy union

# Profile dataset attributes and match discriminativeness with Spark SQL
serf profile-benchmark --dataset abt-buy --output data/abt_buy_profile.md
```

### 3. Prompt Optimization with DSPy GEPA

SERF uses GEPA to iteratively optimize DSPy matching signatures, using a student LM (`openai/gpt-oss-120b-maas`) guided by a reflection teacher LM (`gemini/gemini-4-argon` at `medium` reasoning effort):

```bash
# Inspect generated DSPy signatures and rendered prompts
serf prompts --dataset abt-buy

# Train a benchmark dataset's prompt with GEPA and a contrastive fine-tuned blocker
serf train --dataset dblp-acm \
  --blocking-model data/models/fine-tuned-dblp-acm-bge-small-contrastive \
  --num-threads 16

# Evaluate the trained prompt on the disjoint holdout split
serf benchmark --dataset dblp-acm --signature-mode per-dataset \
  --trained-prompts --eval-split holdout --max-iterations 3
```

### 4. Blocking Analysis, Embedding Sweeps & Contrastive Fine-Tuning

Evaluate and compare embedding models purely on candidate blocking recall without spending LLM tokens, or fine-tune embedding representations using contrastive learning:

```bash
# Sweep default candidate embeddings on blocking recall
serf blocking-sweep --dataset dblp-acm --output data/blocking_sweep.json

# Rank candidate embeddings against published MTEB benchmark categories
serf mteb-rank --candidate-set all --sweep data/blocking_sweep.json

# Contrastive embedding fine-tuning on labeled benchmark datasets (representation learning)
serf fine-tune dblp-acm --model BAAI/bge-small-en-v1.5 --epochs 2 --batch-size 32 \
  --output-dir data/models/fine-tuned-dblp-acm-bge-small-contrastive --seed 42
```

---

## The Experiments: Where SERF Stands Today

Most entity resolution papers do not test entity resolution. They test **pair classification**: an oracle hands the model a pre-selected list of candidate pairs, and the model answers yes or no to each pair in isolation. If a blocker would have missed a pair in real life, pair classification never finds out. If merging $A=B$ and $B=C$ would collapse a cluster into a hairball across three rounds, pair classification never sees that either.

SERF runs **match and merge**—blocking, whole-block resolution, and golden-record canonicalization—on raw tables. We take two unlinked tables, embed them with `BAAI/bge-small-en-v1.5` (33M parameters on CPU), cluster them with FAISS into blocks of ~30 records, hand each whole block to `openai/gpt-oss-120b-maas`, and merge the surviving entities across three iterations. Along the way, SERF enforces **100% identifier conservation**: every input record ID must come out the other side either as a canonical record or inside one's `source_uuids` lineage list. An LLM that silently drops rows is not an entity resolution system.

To know whether this actually works, we have to keep three different evaluation protocols straight and never mix their numbers.

### 1. Full-Table End-to-End Resolution (Untrained Baseline vs. Published SOTA)

First, we ran SERF across the **complete tables** (`--eval-split all --max-iterations 3`) with our hand-written per-dataset DSPy signatures and off-the-shelf `BAAI/bge-small-en-v1.5`—no prompt training, no embedding fine-tuning.

Why run the full tables instead of a 1,000-record sample? Because earlier 1,000-record samples fooled us. When you sample a benchmark by pulling gold match groups, you keep the true matches together and throw away 80% to 97% of the unrelated records in the catalog. Those unrelated records are the **hard negatives**—the near-miss cables, software editions, and printer cartridges that look almost identical to the real thing. When we moved from 1,000-record samples to the full tables (`24,628` records on Walmart-Amazon and `66,879` on DBLP-Scholar), F1 barely moved on clean bibliographic data (`-1.4` points on DBLP-ACM), but dropped `17.6` and `19.4` points on Walmart-Amazon and Amazon-Google purely from false-positive merges.

Here is where untrained SERF sits on the complete tables against **SC-Block** (Brinkmann et al., CIKM 2023—the only published end-to-end paper on these benchmarks) and the best published **Pair Classification (`PC-full`)** scores on the full DeepMatcher test split:

```bash
uv run serf benchmark --dataset <dataset> --signature-mode per-dataset \
  --eval-split all --max-iterations 3
```

| Benchmark Dataset  | Full-Table Records | Gold Pairs | SERF Precision | SERF Recall | SERF Full-Table `E2E` F1 | Published `E2E` SOTA (SC-Block) | Published `PC-full` SOTA (Pair Classification) | Records Conserved |
| :----------------- | -----------------: | ---------: | -------------: | ----------: | -----------------------: | ------------------------------: | ---------------------------------------------: | ----------------: |
| **DBLP-ACM**       |              4,910 |      2,224 |         97.12% |      98.61% |               **97.86%** |                   not published |                           99.32% (EM-Join '25) |         100.0000% |
| **Abt-Buy**        |              2,173 |      1,097 |         97.49% |      88.61% |               **92.84%** |                      **92.90%** |                    95.15% (Qwen3-8B LoRA '26) |         100.0000% |
| **DBLP-Scholar**   |             66,879 |      5,347 |         83.90% |      94.56% |               **88.91%** |                   not published |                     98.51% (Jellyfish-13B '24) |         100.0000% |
| **Walmart-Amazon** |             24,628 |        962 |         70.98% |      84.41% |               **77.11%** |                      **86.00%** |                    91.62% (Qwen3-4B LoRA '26) |         100.0000% |
| **Amazon-Google**  |              4,589 |      1,300 |         56.11% |      75.92% |               **64.53%** |                      **80.30%** |                      81.69% (Jellyfish-7B '24) |         100.0000% |
| **5-Dataset Mean** |                    |            |         81.12% |      88.42% |               **84.25%** |                               — |                                         93.26% |     **100.0000%** |

**What the full-table baseline tells us:**

- **We tie the only published end-to-end SOTA on Abt-Buy** (`92.84%` vs. SC-Block's `92.90%`), and we sit `1.46` points below the pair-classification ceiling on DBLP-ACM (`97.86%` vs. `99.32%`).
- **We are not state of the art yet on Walmart-Amazon (`77.11%`) or Amazon-Google (`64.53%`).** Look at the precision and recall columns: recall is `84.41%` and `75.92%`, while precision is `70.98%` and `56.11%`. Blocking is putting the true pairs together, and then the matcher is merging too many near-misses.
- **Identifier conservation held at `100.0000%` across all five datasets.** On the full tables, the LLM omitted or double-claimed records in `3` to `18` blocks per dataset (`50` to `90` records per block). Under a pairwise contract those rows would have vanished silently; SERF's partition recovery caught and repaired every one.

### 2. Contrastive Blocking + GEPA Prompt Optimization (Scored on Held-Out Splits)

Writing prompts by hand gets you only so far, and if you read the errors on a test set to tweak your prompt, you are training on the test set. Instead, we partitioned each dataset into three strictly disjoint match-group splits (`train`, `val`, and `holdout` at `seed=42`), reserving the `val` and `holdout` budgets first so training never starves evaluation, and trained two components:

1. **Contrastive fine-tuning of the blocker (`serf fine-tune`)**: We fine-tuned `BAAI/bge-small-en-v1.5` for 2 epochs on CPU using `ContrastiveLoss` over the training split (`1:3` positive-to-negative cross-source pairs). On the unseen `holdout` splits, contrastive tuning lifted blocking recall from `67.05%` to **`90.62%`** (`+23.58` points) on Amazon-Google and from `99.42%` to **`99.81%`** on DBLP-ACM, while shrinking the candidate pairs fed to the LLM by `13%` to `28%`.
2. **Reflective prompt optimization with DSPy GEPA (`serf train`)**: Using `openai/gpt-oss-120b-maas` as the student matcher and `gemini/gemini-4-argon` (`reasoning_effort="medium"`) as the reflection teacher, GEPA ran student rollouts on training blocks, read the exact false-positive and false-negative pairs, proposed new DSPy signature instructions, and selected the winning prompt on a **~35-block validation split**. Only after GEPA finished did we score the baseline prompt and the winning prompt—once—on the **disjoint holdout split**.

Why 35 validation blocks? In our first run, we budgeted 200 validation records, which yielded only 4 to 6 scoreable blocks. Four of the six blocks scored `1.0` immediately, the Pareto frontier saturated, and a `+3.79` point validation gain on Amazon-Google collapsed to `+0.16` points (pure noise) on holdout. Sizing validation to ~35 blocks (`1,100` to `2,100` records, matching DSPy's full-eval threshold) gave GEPA enough signal that validation gains actually transferred to the holdout split:

```bash
uv run serf train --dataset <dataset> \
  --blocking-model data/models/fine-tuned-<dataset>-bge-small-contrastive \
  --num-threads 16
```

| Dataset (Contrastive Blocker + GEPA) | Train Blocks | Val Blocks | Holdout Blocks | Validation F1 (Written → Trained) | Disjoint Holdout F1 (Written → Trained) | Holdout Gain |
| :----------------------------------- | -----------: | ---------: | -------------: | :-------------------------------: | :-------------------------------------: | -----------: |
| **DBLP-ACM**                         |           90 |         35 |             36 |         98.42% → **99.15%**       |           98.03% → **98.49%**           |   **+0.46%** |
| **Abt-Buy**                          |           22 |         25 |             25 |         87.91% → **97.61%**       |           91.52% → **95.17%**           |   **+3.65%** |
| **Amazon-Google**                    |           66 |         35 |             35 |         82.53% → **92.36%**       |           82.13% → **89.82%**           |   **+7.69%** |
| **Walmart-Amazon**                   |          140 |         35 |             42 |         87.45% → **94.71%**       |           87.85% → **95.33%**           |   **+7.48%** |
| **DBLP-Scholar**                     |          119 |         35 |             44 |         60.16% → **96.85%**       |           66.61% → **95.77%**           |  **+29.16%** |
| **5-Dataset Mean**                   |              |            |                |       **83.29% → 96.13%**         |         **85.23% → 94.92%**             |   **+9.69%** |

**What Gemini 4 Argon actually discovered during reflection:**

- **Retailer SKU stripping and price fallback on Abt-Buy (`+3.65%` holdout F1)**: The teacher noticed that `abt.name` often lists an internal store SKU while hiding the manufacturer model code inside `abt.description`, and that retailers append kit suffixes (`BD`, `KIT`, `2PK`, `B2`). It rewrote the instructions to strip kit suffixes, extract model numbers from descriptions, and use within-25% price agreement to confirm fallback matches when one side omits the model code entirely.
- **Fixing 1-to-N bipartite grouping on Amazon-Google (`+7.69%`), Walmart-Amazon (`+7.48%`), and DBLP-Scholar (`+29.16%`)**: In those three datasets, a single left-side entity frequently matches 2 to 6 duplicate right-side records inside the same block. When the baseline prompt placed `[left_1, right_1, right_2]` into a single 3-element group, the block-level scorer (`predicted_pairs`) expanded all pairwise combinations—including the same-source pair `(right_1, right_2)`—and penalized it as a false positive against the strictly bipartite ground truth. Reading the error feedback (`"Emitted N pair(s) that are NOT matches: (100xxx, 100yyy)"`), Gemini 4 Argon rewrote the output rules to emit separate 2-element `[left_id, right_id]` groups for each cross-source match, eliminating same-source false positives on the first reflection iteration.

### 3. Why `94.92%` Holdout Block F1 Does Not Mean We Beat Full-Table SOTA Yet

It is tempting to look at the `94.92%` holdout mean (`89.82%` to `98.49%` across the five datasets) and declare victory over the `80.30%` to `92.90%` full-table leaderboard. **Don't do that.** Those two tables measure different tasks:

1. **The holdout split is a match-group sample (`750` to `2,101` records)**: Just like our earlier 1,000-record samples, `sample_random_splits` pulls whole gold match groups so true pairs stay together in their split. On Walmart-Amazon and DBLP-Scholar, a ~2,000-record holdout split leaves out over 90% of the unlinked catalog records that cause false positives in a full-table run.
2. **Single-pass scoreable blocks vs. 3-iteration full-table closure**: `serf train` evaluates one matching pass over the holdout blocks that contain at least one gold pair and macro-averages F1 across those blocks. `serf benchmark --max-iterations 3` runs three rounds of blocking, matching, merging, and connected-component closure across every block (including blocks with zero gold pairs and any gold pairs that blocking failed to group together).
3. **Training touched 65% to 80% of each table**: Because our per-dataset contrastive blocker and GEPA prompt consumed the `train` and `val` splits of each dataset, scoring them on `--eval-split all` would evaluate on records the model was trained on.

**What comes next to prove state of the art:**

- Run 3-iteration end-to-end benchmarks (`serf benchmark --eval-split holdout --max-iterations 3 --trained-prompts`) on the disjoint holdout splits to measure the full pipeline gain after 3-round transitive closure.
- Train leave-one-dataset-out blockers and prompts (`cross-dataset` protocol) so we can score the **complete 100% tables** (`--eval-split all`) on Walmart-Amazon and Amazon-Google with zero training overlap against SC-Block.

For the complete lab notebooks, commands, seeds, and logs behind every number above:

- [experiments/gepa-tuning-results.md](experiments/gepa-tuning-results.md): GEPA prompt optimization with Gemini 4 Argon and contrastive blocking on disjoint holdout splits.
- [experiments/embedding-contrastive-finetuning.md](experiments/embedding-contrastive-finetuning.md): Contrastive fine-tuning of `BAAI/bge-small-en-v1.5` (`+23.6%` blocking recall on Amazon-Google).
- [experiments/split-sizing.md](experiments/split-sizing.md): Why 6-block validation sets saturate and how splits were resized to ~35 validation blocks.
- [experiments/full-scale-benchmark.md](experiments/full-scale-benchmark.md): Full-table 3-iteration end-to-end benchmark and why match-group samples overstate precision.
- [experiments/state-of-the-art.md](experiments/state-of-the-art.md): Literature audit separating `PC-full`, `PC-down`, and `E2E` protocols.
- [BENCHMARKS.md](BENCHMARKS.md): Attribute discriminativeness and dataset profiles.

---

## Configuration

All configuration is centralized in `config.yml` and accessed through `serf.config.config`:

```yaml
models:
  student: "openai/gpt-oss-120b-maas" # Fast open-weight task model
  teacher: "gemini/gemini-4-argon" # Reflection model for GEPA
  teacher_reasoning_effort: "medium"
  llm: "${models.student}"
  embedding: "BAAI/bge-small-en-v1.5" # Vector blocking model
  request_timeout_seconds: 90

optimize:
  auto: "light"
  num_threads: 16
  val_blocks: 35

er:
  max_iterations: 3 # Multi-round resolution rounds
  blocking:
    target_block_size: 30
    min_block_size: 2
    max_block_size: 100

merge:
  semantics:
    name:
      dedupe: fuzzy
      threshold: 0.80
    address:
      dedupe: fuzzy
      threshold: 0.80
    identifier:
      dedupe: exact
```

Access configuration in Python:

```python
from serf.config import config

model_name = config.get("models.llm")
max_iterations = config.get("er.max_iterations", 3)
```

---

## Project Structure

```text
src/serf/
├── cli/             # Click CLI commands and entry points
├── block/           # Vector embeddings, FAISS clustering, and blocking pipeline
├── match/           # LLM matching, UUID mapping, and partition building
├── merge/           # Canonical record creation, type-aware semantics, and lineage
├── schema/          # Schema parsing, YAML specification, and Pydantic model generation
├── edge/            # Knowledge graph edge resolution and deduplication
├── dspy/            # DSPy signatures, XML adapters, prompt optimization (GEPA)
├── eval/            # Evaluation metrics, dataset loaders, and benchmark splits
├── analyze/         # Spark SQL attribute profiling and field type detection
├── config.py        # Centralized YAML configuration loader
└── logs.py          # Structured logging
```

---

## Documentation

| Document                                             | What it covers                                                                |
| ---------------------------------------------------- | ----------------------------------------------------------------------------- |
| [docs/MISSION.md](docs/MISSION.md)                   | What SERF is proving, and what counts as state of the art                     |
| [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md) | Simple, minimal, readable code — start here before contributing               |
| [docs/ID_INVARIANTS.md](docs/ID_INVARIANTS.md)       | The identifier-conservation contract for match/merge, and how to test it      |
| [docs/RESEARCH_LOOP.md](docs/RESEARCH_LOOP.md)       | Experiment registry, metrics, evaluation protocols, and reproducibility rules |
| [docs/QUICKSTART.md](docs/QUICKSTART.md)             | Getting started guide and practical walkthrough                               |
| [docs/SCALABILITY.md](docs/SCALABILITY.md)           | Vector engine options for blocking beyond RAM                                 |
| [docs/FINE_TUNING.md](docs/FINE_TUNING.md)           | Embedding fine-tuning with contrastive learning                               |

---

## Development

```bash
# Install dependencies
uv sync --extra dev

# Run the test suite
uv run pytest tests/

# Run linter and code formatter
uv run ruff check --fix src tests
uv run ruff format src tests

# Static type checking
uv run zuban check src tests

# Pre-commit hooks
pre-commit run --all-files
```

---

## References

1. Jurney, R. (2024). "[The Rise of Semantic Entity Resolution](https://towardsdatascience.com/the-rise-of-semantic-entity-resolution/)." _Towards Data Science_.
2. Khattab, O. et al. (2024). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." _ICLR 2024_.
3. Li, Y. et al. (2021). "Ditto: A Simple and Efficient Entity Matching Framework." _VLDB 2021_.
4. Mudgal, S. et al. (2018). "Deep Learning for Entity Matching: A Design Space Exploration." _SIGMOD 2018_.
5. Papadakis, G. et al. (2020). "Blocking and Filtering Techniques for Entity Resolution: A Survey." _ACM Computing Surveys_.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
