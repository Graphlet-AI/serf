# SERF: Agentic Semantic Entity Resolution Framework

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)

SERF is an open-source framework for **semantic entity resolution** — identifying when two or more records refer to the same real-world entity using large language models, sentence embeddings, and agentic AI.

SERF runs multiple rounds of entity resolution until the dataset converges to a stable state, with DSPy agents controlling all phases dynamically.

<div align="center">
    <img src="assets/entity_resolution.png" alt="Stages of entity resolution: blocking, matching, merging" width="600px" />
    <p><em>Source: <a href="https://medium.com/data-science/entity-resolution-identifying-real-world-entities-in-noisy-data-3e8c59f4f41c">Entity Resolution: Identifying Real-World Entities in Noisy Data</a></em></p>
</div>

## Features

### Phase 0 — Agentic Control

DSPy ReAct agents dynamically orchestrate the entire pipeline, adjusting blocking parameters, selecting matching strategies, and deciding when convergence is reached.

### Phase 1 — Semantic Blocking

Clusters records using **bge-small-en-v1.5 sentence embeddings** and **FAISS IVF** to create efficient blocks for comparison. Auto-scales block size across iterations. A larger `bge-large-en-v1.5` tier is available with `--embedding-tier high`, and records can be embedded as JSON with all field names inline instead of by name with `--blocking-strategy json`.

### Phase 2 — Schema Alignment, Matching and Merging

All three operations in a single LLM prompt via **DSPy signatures** with **`dspy.XMLAdapter`** for structured output formatting. Block-level matching lets the LLM see all records simultaneously for holistic decisions.

### Phase 3 — Edge Resolution

For knowledge graphs: deduplicate edges that result from merging nodes using LLM-guided intelligent merging.

## Architecture

| Component          | Technology                                         |
| ------------------ | -------------------------------------------------- |
| Package Manager    | **uv**                                             |
| Data Processing    | **PySpark 4.x**                                    |
| LLM Framework      | **DSPy 3.x** with `XMLAdapter`                     |
| Embeddings         | **bge-small-en-v1.5** (LOW) / **bge-large-en-v1.5** (HIGH) via sentence-transformers |
| Vector Search      | **FAISS IndexIVFFlat**                             |
| Linting/Formatting | **Ruff**                                           |
| Type Checking      | **zuban** (mypy-compatible)                        |

## Quick Start

### Installation

```bash
git clone https://github.com/Graphlet-AI/serf.git
cd serf
uv sync --extra dev
```

### Docker

```bash
# Build
docker compose build

# Run any serf command
docker compose run serf benchmark --dataset dblp-acm

# Run benchmarks
docker compose --profile benchmark up

# Run tests
docker compose --profile test up

# Analyze a dataset (put your file in data/)
docker compose run serf analyze --input data/input.csv --output data/er_config.yml
```

Set your API keys in a `.env` file or export them:

```bash
echo "GEMINI_API_KEY=your-key" > .env
echo "VERTEX_AI_TOKEN=your-vertex-token" >> .env
echo "GOOGLE_CLOUD_PROJECT=your-gcp-project" >> .env
```

### System Requirements

- Python 3.12+
- Java 11/17/21 (for PySpark)
- 4GB+ RAM recommended

### CLI Usage

```bash
# Profile a dataset
serf analyze --input data/companies.parquet

# Run the full ER pipeline
serf resolve --input data/entities.csv --output data/resolved/ --iteration 1

# Run individual phases
serf block --input data/entities.csv --output data/blocks/ --method semantic
serf match --input data/blocks/ --output data/matches/ --iteration 1
serf eval --input data/matches/

# Benchmark against standard datasets
# (walmart-amazon, abt-buy, amazon-google, dblp-acm, dblp-scholar)
serf download --dataset dblp-acm
serf benchmark --dataset dblp-acm --output data/results/

# Each run does er.max_iterations rounds (3 by default): every round re-blocks
# the entities merged by the previous one, so records the first round of
# blocking kept apart get another chance to meet. Override per run:
serf benchmark --dataset dblp-acm --max-iterations 1 --output data/results/

# Match with the typed signature written for one dataset instead of the shared
# BlockMatch signature, on a 1000-record sample drawn by ground-truth match group
serf benchmark --dataset dblp-acm --signature-mode per-dataset \
  --sample-records 1000 --seed 42 --output data/results/

# Compare embedding models on blocking recall alone, no LLM calls and no cost.
# Candidates default to benchmarks.embedding_candidates in config.yml
serf blocking-sweep --dataset dblp-acm --output data/blocking_sweep.json

# Sweep the large candidates instead, scoring name-only against JSON blocking
serf blocking-sweep --candidate-set large --blocking-strategy both --sample 2000 \
  --output data/blocking_sweep_large.json

# Block with the large embedding and embed every field as JSON, not just the name
serf benchmark --dataset amazon-google --embedding-tier high --blocking-strategy json \
  --sample-records 1000 --output data/results/

# Optimize ER signatures with GEPA (GPT OSS 120b student, Gemini 3.5 Flash-Lite teacher)
# Randomly samples 2000 train / 1000 val / 1000 holdout records, keeping
# ground-truth match groups whole so gold pairs survive, then blocks within
# each split to build the BlockMatch examples. Val is filled first.
serf optimize --dataset dblp-acm --signature block-match
```

### Python API

```python
from serf.block.pipeline import SemanticBlockingPipeline
from serf.match.matcher import EntityMatcher
from serf.eval.metrics import evaluate_resolution

# Block
pipeline = SemanticBlockingPipeline(target_block_size=50)
blocks, metrics = pipeline.run(entities)

# Match
matcher = EntityMatcher(model="openai/gpt-oss-120b-maas")
resolutions = await matcher.resolve_blocks(blocks)

# Evaluate
metrics = evaluate_resolution(predicted_pairs, ground_truth_pairs)
```

### DSPy Interface

```python
import dspy
from serf.dspy.signatures import BlockMatch

from serf.dspy.lm import create_lm

lm = create_lm(role="student")  # openai/gpt-oss-120b-maas via VERTEX_AI_TOKEN
dspy.configure(lm=lm, adapter=dspy.XMLAdapter())

matcher = dspy.ChainOfThought(BlockMatch)
result = matcher(block_records=block_json, schema_info=schema, few_shot_examples=examples)
```

### Per-Dataset Typed Signatures

`BlockMatch` describes one anonymous `Entity` type and is shared by every dataset.
Each benchmark match task also has its own signature with two typed side models,
one per source, whose field descriptions come from the entity resolution
literature for that task. Select them with `--signature-mode per-dataset`.

```python
from serf.dspy.dataset_signatures import get_dataset_spec
from serf.match.dataset_matcher import DatasetMatcher

spec = get_dataset_spec("walmart-amazon")  # WalmartProduct / AmazonElectronicsProduct
resolutions = await DatasetMatcher("walmart-amazon").resolve_blocks(blocks)
```

## Benchmark Results

Performance on standard ER benchmarks from the [Leipzig Database Group](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution). Matching uses GPT OSS 120b (Vertex AI MaaS) as the student/task LM via DSPy BlockMatch, with Gemini 3.5 Flash-Lite as the teacher/reflection LM for GEPA.

These rows were measured with the former multilingual-e5-base default, before the FAISS cluster-count fix and before the XML adapter fix, so they understate what the current config reaches — see [Blocking Recall](#blocking-recall) for the gap and `experiments/embedding-blocking-sweep.md` for the measurements.

| Dataset      | Domain        | Left  | Right | Matches | Precision | Recall | F1         |
| ------------ | ------------- | ----- | ----- | ------- | --------- | ------ | ---------- |
| **DBLP-ACM** | Bibliographic | 2,616 | 2,294 | 2,224   | 0.8849    | 0.5809 | **0.7014** |

Blocking uses name-only embeddings for tighter semantic clusters. All matching decisions are made by the LLM — no embedding similarity thresholds.

Every number below the full-table row is post-fix. Until recently `dspy.XMLAdapter` rejected 27 of every
33 blocks over unescaped ampersands and XML's missing null literal, which sent them through DSPy's JSON
fallback at double the inference cost and lost the ones the fallback also failed. The investigation and
the before/after are in [experiments/xml-adapter-block-loss.md](experiments/xml-adapter-block-loss.md).

### Against the Public Leaderboard

The standard entity resolution scoreboard is the Papers With Code [Entity Resolution](https://paperswithcode.com/task/entity-resolution)
task. Papers With Code was sunset in 2025 and now redirects to Hugging Face, so the live mirror of
those boards is [OpenCodePapers](https://opencodepapers-b7572d.gitlab.io/benchmarks/entity-resolution-on-abt-buy.html).
SERF is a 1,000-record sample at seed 42, three ER iterations, `gpt-oss-120b` matching, no training.

| Model                                | Abt-Buy F1 | Task                     | Trained on the benchmark |
| ------------------------------------ | ---------- | ------------------------ | ------------------------ |
| gpt4-0613 zero-shot                  | 95.78      | pair classification      | no                       |
| RoBERTa-SupCon                       | 94.29      | pair classification      | yes                      |
| gpt-4o-mini fine-tuned               | 94.09      | pair classification      | yes                      |
| gpt-4o-2024-08-06                    | 92.20      | pair classification      | no                       |
| RobEM                                | 90.90      | pair classification      | yes                      |
| **SERF gpt-oss-120b, LOW**           | **90.38**  | **end-to-end resolution**| **no**                   |
| HierGAT                              | 89.80      | pair classification      | yes                      |
| **SERF gpt-oss-120b, HIGH**          | **89.63**  | **end-to-end resolution**| **no**                   |
| Ditto                                | 89.33      | pair classification      | yes                      |
| gpt-4o-mini                          | 87.68      | pair classification      | no                       |
| Llama-3.1-70B                        | 79.12      | pair classification      | no                       |

The comparison is indicative, not like-for-like. Every leaderboard entry scores pair classification:
the candidate pairs are handed to the model and it labels each one. SERF does the whole task, so its
recall carries the pairs blocking never proposed — an Abt-Buy blocking ceiling of 0.8912 on this
sample — which a pair classifier never pays for. Read the row as "end-to-end, untrained, around
fine-tuned Ditto", not as a rank.

### Blocking Recall

Blocking recall is the share of ground-truth pairs whose two records land in the same block. Matching
never sees the rest, so this is a hard ceiling on end-to-end recall. Measured with `serf blocking-sweep`
on the full tables, one pass, `target_block_size: 30`. Full results and the two blocking bugs this
uncovered are in [experiments/embedding-blocking-sweep.md](experiments/embedding-blocking-sweep.md).

| Embedding                         | DBLP-ACM | DBLP-Scholar | Abt-Buy | Amazon-Google | Walmart-Amazon | Mean       | Embed secs |
| --------------------------------- | -------- | ------------ | ------- | ------------- | -------------- | ---------- | ---------- |
| **bge-small-en-v1.5** *(default)* | 0.9654   | 0.9048       | 0.8952  | 0.6512        | 0.8545         | **0.8542** | **273**    |
| gte-base                          | 0.9708   | 0.9211       | 0.8724  | 0.6821        | 0.8514         | 0.8595     | 661        |
| gte-small                         | 0.9604   | 0.9233       | 0.8824  | 0.6410        | 0.8410         | 0.8496     | 228        |
| bge-base-en-v1.5                  | 0.9856   | 0.9020       | 0.8569  | 0.6461        | 0.8545         | 0.8490     | 898        |
| all-MiniLM-L6-v2                  | 0.9717   | 0.9205       | 0.8551  | 0.6435        | 0.7775         | 0.8337     | 133        |
| all-mpnet-base-v2                 | 0.9793   | 0.9278       | 0.8323  | 0.6590        | 0.7391         | 0.8275     | 697        |
| multilingual-e5-base *(former)*   | 0.9735   | 0.8898       | 0.8724  | 0.5047        | 0.8025         | 0.8086     | 718        |

`gte-base` edges out the default by 0.005 mean recall for 2.4x the embedding time. Blocking re-embeds
every record on every ER round, so the smaller model is the better default; set `models.embedding` to
trade back.

### LOW vs HIGH Embeddings

The default is the LOW tier. HIGH is the large alternative, chosen from MTEB(eng, v2) *clustering* —
the task category blocking actually performs — among models within 3B parameters and 1,024 dimensions.
Measured on 2,000-record samples per dataset, name-only blocking, one pass. Full protocol, the JSON
results and the four large models that will not run on transformers 5.16 are in
[experiments/low-high-embeddings-and-json-blocking.md](experiments/low-high-embeddings-and-json-blocking.md).

| Embedding                                | DBLP-ACM   | DBLP-Scholar | Abt-Buy    | Walmart-Amazon | Amazon-Google | Mean       | Embed secs |
| ---------------------------------------- | ---------- | ------------ | ---------- | -------------- | ------------- | ---------- | ---------- |
| mxbai-embed-large-v1                     | 0.9904     | **0.9744**   | **0.9001** | **0.9236**     | 0.6323        | **0.8842** | 291        |
| **bge-small-en-v1.5** *(LOW, default)*   | 0.9745     | 0.9659       | 0.8912     | 0.8750         | **0.6756**    | 0.8765     | **47**     |
| multilingual-e5-large-instruct           | **0.9947** | 0.9455       | 0.8952     | 0.9028         | 0.6143        | 0.8705     | 334        |
| **bge-large-en-v1.5** *(HIGH)*           | 0.9936     | 0.9489       | 0.8813     | 0.8750         | 0.6338        | 0.8665     | 324        |
| F2LLM-0.6B                               | 0.9172     | 0.8944       | 0.7636     | 0.7708         | 0.5979        | 0.7888     | 183        |
| Qwen3-Embedding-0.6B                     | 0.9299     | 0.8739       | 0.7280     | 0.7847         | 0.6099        | 0.7853     | 186        |

Going big does not buy blocking recall. Only one large model beats the 33M default, by 0.0077 for six
times the CPU, and two finish below it. MTEB clustering rank is no guide either: `F2LLM-0.6B` has the
best clustering score of the six and the worst blocking recall, while `bge-small-en-v1.5` has the
second-worst clustering score and the second-best blocking recall. HIGH is worth reaching for on the
bibliographic datasets, where it leads by 0.019 on DBLP-ACM.

### Name vs JSON Blocking

`--blocking-strategy json` embeds every populated field as a JSON object with the field names inline
instead of embedding the name alone. It loses on 27 of 30 model-dataset cells, by 0.13 to 0.34 on the
mean, worst on DBLP-ACM where venue, year and authors are shared by thousands of papers and drown the
title. The exception is Amazon-Google with a large model, where it produces the best blocking recall
this project has reached:

| Strategy                                 | Amazon-Google |
| ---------------------------------------- | ------------- |
| **json** + multilingual-e5-large-instruct | **0.7429**    |
| json + mxbai-embed-large-v1               | 0.7190        |
| name + bge-small-en-v1.5 *(default)*      | 0.6756        |
| json + bge-small-en-v1.5                  | 0.4753        |

Short, abbreviated product names that share a manufacturer token leave the name alone ambiguous, and
the side fields carry signal it does not. Only models large enough to read structure out of a JSON blob
benefit; the smaller ones lose on Amazon-Google too. JSON blocking stays off by default.

End to end on Abt-Buy, 1,000-record sample at seed 42, three ER iterations, the ordering holds:

| Tier     | Embedding            | Precision  | Recall     | F1         | Seconds |
| -------- | -------------------- | ---------- | ---------- | ---------- | ------- |
| **LOW**  | bge-small-en-v1.5    | 0.9204     | **0.8878** | **0.9038** | 410     |
| HIGH     | bge-large-en-v1.5    | **0.9265** | 0.8681     | 0.8963     | 508     |

Matching cannot recover a pair blocking never proposed, so the tier that blocks better finishes better.

### Generic vs Per-Dataset Signatures

Measured on 1,000-record samples per dataset drawn by ground-truth match group (seed 42), one ER
iteration, identical blocking in both arms. These are sample runs, so they are not comparable to the
full-table row above. Full protocol and cost in
[experiments/per-dataset-signature-baseline.md](experiments/per-dataset-signature-baseline.md).

| Dataset            | F1 `generic` | F1 `per-dataset` | Delta      |
| ------------------ | ------------ | ---------------- | ---------- |
| **DBLP-ACM**       | 0.9077       | **0.9742**       | +0.0666    |
| **DBLP-Scholar**   | 0.7491       | **0.8713**       | +0.1222    |
| **Abt-Buy**        | 0.7574       | **0.8402**       | +0.0827    |
| **Amazon-Google**  | 0.5000       | **0.6654**       | +0.1654    |
| **Walmart-Amazon** | 0.7634       | **0.8905**       | +0.1272    |

The typed signatures also improved precision on all five datasets and used less than half the tokens,
because their output is the list of matched pairs rather than an echo of every entity in the block.

## Project Structure

```
src/serf/
├── cli/             # Click CLI commands
├── dspy/            # DSPy types, signatures, per-dataset schemas, agents
├── block/           # Semantic blocking (embeddings, FAISS, normalization)
├── match/           # UUID mapping, LLM matching, few-shot examples
├── merge/           # Field-level entity merging
├── edge/            # Edge resolution for knowledge graphs
├── eval/            # Metrics, benchmark datasets
├── analyze/         # Dataset profiling, field detection
├── spark/           # PySpark schemas, utils, Iceberg, graph components
├── config.py        # Configuration management
└── logs.py          # Logging
```

## Configuration

All configuration is centralized in `config.yml`:

```python
from serf.config import config
model = config.get("models.llm")  # "openai/gpt-oss-120b-maas"
teacher = config.get("models.teacher")  # "gemini/gemini-3.5-flash-lite"
block_size = config.get("er.blocking.target_block_size")  # 50
```

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/

# Lint and format
uv run ruff check --fix src tests
uv run ruff format src tests

# Type check
uv run zuban check src tests

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## References

1. Jurney, R. (2024). "[The Rise of Semantic Entity Resolution](https://towardsdatascience.com/the-rise-of-semantic-entity-resolution/)." _Towards Data Science_.
2. Khattab, O. et al. (2024). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." _ICLR 2024_.
3. Li, Y. et al. (2021). "Ditto: A Simple and Efficient Entity Matching Framework." _VLDB 2021_.
4. Mudgal, S. et al. (2018). "Deep Learning for Entity Matching: A Design Space Exploration." _SIGMOD 2018_.
5. Papadakis, G. et al. (2020). "Blocking and Filtering Techniques for Entity Resolution: A Survey." _ACM Computing Surveys_.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
