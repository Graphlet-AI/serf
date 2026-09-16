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

| Component | Technology |
|---|---|
| **Package Manager** | [uv](https://docs.astral.sh/uv/) |
| **Data Processing** | [PySpark 4.x](https://spark.apache.org/) (optional distributed ETL & graph operations) |
| **LLM Framework** | [DSPy](https://dspy.ai/) with `XMLAdapter` for structured XML/Pydantic outputs |
| **Embeddings** | `BAAI/bge-small-en-v1.5` via [Sentence-Transformers](https://sbert.net/) |
| **Vector Indexing** | [FAISS](https://github.com/facebookresearch/faiss) (`IndexIVFFlat`) |
| **Prompt Tuning** | DSPy GEPA (student/teacher reflection optimization) |
| **Code Quality** | Ruff (linting & formatting), zuban / mypy (static type checking) |

---

## The Resolved Record Data Model

Entity resolution merges multiple input records into unified canonical entities. To ensure data integrity, auditability, and no information loss, SERF enforces strict record conventions:

1. **UUID-Native Identity**: Every input entity has a UUID. When records merge, SERF mints a new UUID and records all constituent record UUIDs—including transitively merged ones—in `source_uuids`. Unmatched records preserve their original identity.
2. **Multi-Valued Fields as Lists**: In a canonical record, non-identifier fields hold lists ordered by completeness (fullest value first). If two sources supply valid information (e.g., different phone numbers or addresses), both are retained rather than forcing an arbitrary lossy selection.
3. **Type-Aware Deduplication**: Field values are merged based on semantic field types:

| Field Type | Deduplication Strategy | Description |
|---|---|---|
| `name`, `address` | Fuzzy similarity (`dedupe: fuzzy`) | Near-duplicate spellings collapse to the most complete variant. |
| `identifier`, `email`, `phone`, `url`, `numeric` | Normalization (`dedupe: exact`) | Values normalize (e.g., lowercasing, stripping punctuation) before exact deduplication. |
| `text` | Preserved (`dedupe: none`) | Freeform text (e.g., long descriptions) is never collapsed. |

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
      dedupe: exact  # Exact matching prevents merging "CA" and "GA"
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

SERF uses GEPA to iteratively optimize DSPy matching signatures, using a student LM (e.g., `openai/gpt-oss-120b-maas`) guided by a reflection teacher LM (e.g., `gemini/gemini-3.8-flash`):

```bash
# Inspect generated DSPy signatures and rendered prompts
serf prompts --dataset abt-buy

# Train a benchmark dataset's prompt with GEPA
serf train --dataset dblp-acm --auto light

# Evaluate the trained prompt on the held-out evaluation split
serf benchmark --dataset dblp-acm --signature-mode per-dataset --trained-prompts
```

### 4. Blocking Analysis & Embedding Sweeps

Evaluate and compare embedding models purely on candidate blocking recall without spending LLM tokens:

```bash
# Sweep default candidate embeddings on blocking recall
serf blocking-sweep --dataset dblp-acm --output data/blocking_sweep.json

# Rank candidate embeddings against published MTEB benchmark categories
serf mteb-rank --candidate-set all --sweep data/blocking_sweep.json
```

---

## Benchmark Performance

We evaluate SERF on the standard [Leipzig Entity Resolution Benchmarks](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).

All figures below reflect **genuine full-table end-to-end evaluation**: the framework starts with the raw unlinked tables, runs semantic vector blocking, prompts the LLM for block partitions, merges entities across 3 iterations, and evaluates against ground-truth pairs.

| Benchmark Dataset | Domain | Left Records | Right Records | Gold Pairs | SERF End-to-End F1 | SOTA Reference |
|---|---|---|---|---|---|---|
| **DBLP-ACM** | Citations / Publications | 2,616 | 2,294 | 2,224 | **97.86%** | 99.32% (EM-Join)* |
| **Abt-Buy** | E-commerce Products | 1,076 | 1,076 | 1,097 | **92.84%** | 92.90% (SC-Block E2E) |
| **DBLP-Scholar** | Citations / Publications | 2,616 | 64,263 | 5,347 | **88.91%** | 98.51% (Jellyfish-13B)* |
| **Walmart-Amazon** | Electronics & Office | 2,554 | 22,074 | 1,154 | **77.11%** | 86.00% (SC-Block E2E) |
| **Amazon-Google** | Software & Electronics | 1,363 | 3,226 | 1,167 | **64.53%** | 80.30% (SC-Block E2E) |

*\* Note on published comparisons: Most published academic results (marked with \*) evaluate **pair classification** on pre-filtered candidate pairs rather than full-table end-to-end entity resolution. In full end-to-end resolution, any true match missed during blocking can never be recovered by the matcher. Across end-to-end systems, SERF achieves competitive results on product matching and matches published state-of-the-art on Abt-Buy.*

For detailed experimental reports, error breakdowns, and ablation studies:
- [BENCHMARKS.md](BENCHMARKS.md): Attribute discriminativeness and dataset profiles.
- [experiments/full-scale-benchmark.md](experiments/full-scale-benchmark.md): Full-scale multi-iteration benchmark analysis.
- [experiments/state-of-the-art.md](experiments/state-of-the-art.md): Evaluation protocol comparison (pair classification vs. end-to-end).
- [experiments/embedding-blocking-sweep.md](experiments/embedding-blocking-sweep.md): Embedding model blocking recall comparisons.

---

## Configuration

All configuration is centralized in `config.yml` and accessed through `serf.config.config`:

```yaml
models:
  llm: "openai/gpt-oss-120b-maas"        # Student / task model
  teacher: "gemini/gemini-3.8-flash"     # Reflection model for GEPA
  embedding: "BAAI/bge-small-en-v1.5"    # Vector blocking model
  request_timeout_seconds: 180

er:
  max_iterations: 3                      # Multi-round resolution rounds
  blocking:
    target_block_size: 50
    min_block_size: 5
    max_block_size: 100

merge:
  semantics:
    name:
      dedupe: fuzzy
      threshold: 0.85
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

1. Jurney, R. (2024). "[The Rise of Semantic Entity Resolution](https://towardsdatascience.com/the-rise-of-semantic-entity-resolution/)." *Towards Data Science*.
2. Khattab, O. et al. (2024). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." *ICLR 2024*.
3. Li, Y. et al. (2021). "Ditto: A Simple and Efficient Entity Matching Framework." *VLDB 2021*.
4. Mudgal, S. et al. (2018). "Deep Learning for Entity Matching: A Design Space Exploration." *SIGMOD 2018*.
5. Papadakis, G. et al. (2020). "Blocking and Filtering Techniques for Entity Resolution: A Survey." *ACM Computing Surveys*.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
