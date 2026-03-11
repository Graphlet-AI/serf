# SERF: Agentic Semantic Entity Resolution Framework -- Build Plan

This document is a comprehensive implementation plan for building **SERF** (Semantic Entity Resolution Framework) -- an open-source, agentic system for semantic entity resolution. It is designed for an AI coding agent (Cursor Agent grind mode) to execute overnight (8-12 hours) and produce a working, tested framework.

**Repository:** [github.com/Graphlet-AI/serf](https://github.com/Graphlet-AI/serf)

> **Key Rule:** Embeddings are for BLOCKING only (FAISS clustering). ALL matching is done by LLM via DSPy signatures. Never use embedding cosine similarity for match decisions.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Historical Context](#2-historical-context)
3. [Abzu Code Guide](#3-abzu-code-guide)
4. [Architecture and Technology Stack](#4-architecture-and-technology-stack)
5. [Data Model and Type System](#5-data-model-and-type-system)
6. [User Interfaces](#6-user-interfaces)
7. [Agentic Semantic Entity Resolution](#7-agentic-semantic-entity-resolution)
8. [Standard ER Benchmark Datasets](#8-standard-er-benchmark-datasets)
9. [Scaling Analysis](#9-scaling-analysis)
10. [Implementation Plan](#10-implementation-plan)
11. [Testing Strategy](#11-testing-strategy)
12. [References](#12-references)

---

## 1. Introduction and Motivation

Entity resolution (ER) -- the process of determining when two or more records refer to the same real-world entity -- is among the oldest and most important problems in data management. The Fellegi-Sunter model (1969) formalized probabilistic record linkage, and decades of research have produced systems based on string similarity metrics (Jaccard, Jaro-Winkler, Levenshtein), rule-based blocking, and supervised classifiers. These traditional systems, exemplified by production platforms like Senzing and Informatica MDM, are fast and auditable but fundamentally brittle: they require hand-crafted rules per domain, break on schema variations, and cannot capture the deep semantic understanding needed when data sources use different terminology for the same concepts.

The arrival of transformer-based language models created a paradigm shift. The BERT era (2019-2023) demonstrated that pre-trained language model embeddings could dramatically improve blocking (DeepBlocker, Thirumuruganathan et al. 2021) and matching (Ditto, Li et al. 2021). But these approaches still required fine-tuning per domain and explicit pairwise comparison logic.

**Semantic entity resolution** represents the next evolution: using large language models to perform schema alignment, matching, and merging in a single prompt, guided by structured output specifications. As argued in [The Rise of Semantic Entity Resolution](https://towardsdatascience.com/the-rise-of-semantic-entity-resolution/) (Jurney, 2024), LLMs bring "world knowledge" to ER -- they understand that "IBM" and "International Business Machines" are the same entity, that "TSLA" is Tesla's ticker symbol, and that "Cupertino, CA" and "Apple Park" both refer to Apple's headquarters, without any explicit rules.

SERF takes this further with **agentic** semantic ER: DSPy agents (ReAcT pattern) dynamically control all phases of the pipeline -- adjusting blocking parameters, selecting matching strategies, evaluating results, and deciding when convergence is reached. The system runs multiple rounds of entity resolution until the dataset converges to a stable state, with each round's results informing the next.

### Key Innovation: Block-Level Matching

Rather than the traditional pairwise comparison approach, SERF matches entire blocks of records at once. The LLM sees all records in a block simultaneously and makes holistic matching decisions, leveraging cross-record context that pairwise approaches miss. This is both more accurate (the LLM can triangulate: if A matches B and B matches C, then A should match C) and more cost-efficient (shared context reduces token usage by 20-40x compared to pairwise calls).

### Counter-Arguments to LLM-ER Skepticism

Senzing's Jeff Jonas has argued that LLMs are [too slow, too expensive, and too prone to hallucination](https://senzing.com/entity-resolution-generative-ai/) for production ER. SERF addresses each concern:

| Concern           | SERF's Counter                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| Too slow          | Semantic blocking reduces LLM calls to O(blocks); PySpark parallelizes across clusters          |
| Too expensive     | Gemini 2.0 Flash at $0.10/1M tokens; block-level matching reduces calls 50x or more vs pairwise |
| Hallucinations    | BAML structured outputs + DSPy optimization + multi-round convergence                           |
| Not deterministic | Temperature=0, multiple convergence rounds, confidence scoring                                  |
| Not explainable   | Chain-of-thought reasoning, match rationale in structured output fields                         |
| Not scalable      | PySpark for ETL, embedding models for blocking, LLMs only for semantic decisions                |

The fundamental insight is that SERF uses LLMs **where they add unique value** (semantic understanding, schema alignment, complex matching decisions) and traditional scalable tools (embeddings, FAISS, PySpark, Iceberg) for everything else.

---

## 2. Historical Context

### Deep Discovery and Graphlet AI

SERF is the culmination of a multi-year research trajectory. Russell Jurney founded **Deep Discovery**, a startup focused on entity resolution and knowledge graph construction. The thesis was that ER is a foundational problem that could be dramatically improved with modern NLP and embedding techniques. Deep Discovery aimed to build tooling and services around automated ER pipelines for business intelligence, data integration, and graph analytics.

This work evolved into **Graphlet AI** ([github.com/graphlet-ai/graphlet](https://github.com/graphlet-ai/graphlet)), an open-source framework that used BERT-era sentence-transformer embeddings for semantic blocking and PySpark for scalable ETL. The [Graphlet AI issues](https://github.com/graphlet-ai/graphlet/issues) laid out an ambitious roadmap for configurable blocking strategies, evaluation metrics, graph database integration, and benchmark evaluation. A key presentation ([YouTube talk](https://www.youtube.com/watch?v=GVFiUjERxhk), [slides](https://docs.google.com/presentation/u/0/d/1ihMyfjbhltzfeSmv12jlQ81yJ27bwwG8e-_rBZt5fHk/mobilepresent)) described the vision of using transformer models for ER in knowledge graphs.

The concept of **Knowledge Graph Factories** ([blog post](https://blog.graphlet.ai/knowledge-graph-factories-f50466fb7512)) extended this vision: automated pipelines that continuously ingest, extract, resolve, and integrate entities into knowledge graphs. ER is the critical bottleneck in such factories -- without it, graphs fill with duplicate nodes that degrade query quality and analytics.

### From Graphlet AI to Abzu to SERF

**Abzu** ([github.com/Graphlet-AI/abzu](https://github.com/Graphlet-AI/abzu)) was the production implementation that proved these concepts work. It implements a complete, multi-iteration ER pipeline for company entities extracted from SEC 10-K filings and industry news. Abzu demonstrated that semantic blocking with FAISS + LLM-based matching/merging could achieve significant data reduction across multiple rounds. SERF generalizes Abzu's proven patterns into a domain-agnostic, open-source framework.

| Era       | System                    | Blocking                          | Matching                           | Merging                     |
| --------- | ------------------------- | --------------------------------- | ---------------------------------- | --------------------------- |
| 2019-2022 | Deep Discovery / Graphlet | Sentence-transformers             | Embedding similarity + classifiers | Rule-based field resolution |
| 2023-2025 | Abzu                      | FAISS IVF + sentence-transformers | Gemini via BAML (block-level)      | LLM-guided via BAML         |
| 2026+     | **SERF**                  | FAISS IVF + Qwen3 embeddings      | DSPy agents + Gemini/Claude        | DSPy-optimized LLM merging  |

---

## 3. Abzu Code Guide

Abzu's codebase at `/Users/rjurney/Software/weave` contains the proven ER patterns that SERF should adopt and generalize. The agent building SERF should study these files carefully:

### 3.1 BAML Templates (LLM Prompt Engineering)

- **`baml_src/multi_er.baml`** -- The core multi-entity resolution prompt. Defines `MultiEntityResolution` and `FewShotMultiEntityResolution` functions using `Gemini20Flash`. Key types: `CompanyList` (block_key, block_key_type, block_size, companies), `MergeCompany` (id, source_ids only -- no name field). The LLM receives an entire block of companies and returns merged results with comprehensive ID tracking (lowest input ID becomes master, ALL OTHER IDs go to `source_ids`). Company class includes `match_skip_history` field for tracking which iterations skipped each company. Test cases cover two-company merges, UUID tracking, and multiple source UUID accumulation. **This pattern must be preserved in SERF using DSPy signatures instead of BAML.**

- **`baml_src/article.baml`** -- Base schema definitions. `Company` class with fields: id, uuid, name, cik, ticker, description, website_url, headquarters_location, jurisdiction, revenue_usd, employees, founded_year, ceo, linkedin_url, source_ids, source_uuids, match_skip, match_skip_reason, match_skip_history. Also defines `Ticker` (id, uuid, symbol, exchange), `Exchange` enum (50+ Bloomberg codes), and `Relationship` types.

- **`baml_src/edge_er.baml`** -- Edge resolution prompt. Defines comprehensive types: `EdgeRelationshipInput` (type, description, amount, currency, date, percentage, quarter, url), `EdgeBlock` (src_name, dst_name, relationships, block_size), `MergedEdgeRelationship`, and `EdgeResolutionResult` (merged_relationships, was_resolved, original_count, resolved_count). The `ResolveEdgeBlock` function uses `Gemini20Flash` with rules for merging same-deal relationships across different types (e.g., "Supplier" + "SupplyAgreement"). Test cases cover duplicate investments and distinct relationships.

- **`baml_src/clients.baml`** -- LLM client configurations. Active clients: `Gemini20Flash` (60s/120s timeouts), `Gemini20FlashLong` (180s/180s timeouts, same model), `Gemini25FlashLite` (60s/120s timeouts). All use `temperature 0.0` and exponential retry policy (3 retries, 300ms base delay, 1.5x multiplier, 10s max). Commented out: O3Mini, GPT4o, Gemini25Flash, Gemini25Pro, fallback strategies.

- **`baml_src/company_er.baml`** -- Pairwise company matching for cases where blocking isn't sufficient.

- **`baml_src/final_er.baml`** -- Final deduplication for companies sharing the same UUID across different blocking iterations.

### 3.2 Python ER Modules

- **`abzu/er/uuid.py`** (574 lines) -- **Critical pattern.** `UUIDMapper` maps UUIDs to consecutive integers for LLM processing (LLMs work better with small integers than UUIDs). `process_block_with_uuid_mapping()` is the core async function: (1) maps UUIDs to ints, (2) strips source_uuids before sending to LLM (avoids context bloat), (3) calls BAML, (4) restores source_uuids from cached data, (5) recovers missing companies via two-phase recovery (Step 1: add back missing UUIDs to existing output companies; Step 2: recover entire missing companies with `match_skip_reason = "missing_in_match_output"`). Ticker normalization handles str, list, Row, dict formats with Exchange enum validation. **This UUID-to-integer mapping pattern is essential for SERF.**

- **`abzu/er/match.py`** (537 lines) -- Main matching orchestration. `match_entities()` loads blocks from Parquet (Arrow optimization disabled to preserve Python lists), validates block schema via `validate_block_schema()`, separates singleton/multi-company blocks, processes multi-company blocks via `process_blocks_async()` with semaphore-based rate limiting. Includes `backup_file()` for file/directory backup before overwriting (keeps only most recent backup). Comprehensive error recovery: marks error-recovered companies with `match_skip_reason = "error_recovery"`, updates `match_skip_history` with current iteration number, and logs detailed recovery statistics. Always generates new UUIDs for resolved companies.

- **`abzu/er/embeddings.py`** (83 lines) -- `CompanyEmbedder` class using `SentenceTransformer` (default: `intfloat/multilingual-e5-base`, 768-dim), auto-detects device (CUDA/MPS/CPU) via `get_torch_device()`, normalizes embeddings.

- **`abzu/er/faiss.py`** (142 lines) -- `FAISSBlocker` class using `IndexIVFFlat` with inner product metric. Calculates `nlist` from `target_block_size`, caps at `sqrt(n)`. Optional `max_distance` filtering. Returns `{block_key: [uuid1, uuid2, ...]}`.

- **`abzu/er/edge_match.py`** (176 lines) -- Async edge resolution: `process_edge_block()`, `resolve_edge_blocks()`, `run_edge_resolution()`. Uses `EdgeBlock`, `EdgeRelationshipInput` BAML types with semaphore rate limiting. Error recovery returns original relationships unchanged.

- **`abzu/er/few_shot.py`** (80 lines) -- Few-shot example generation for merge prompts. `company_dicts_to_baml()` converts dicts to `MergeCompanyExampleSet`. Hardcoded `company_id_tracking_dicts` example demonstrating merge of companies with ids 1 (source_ids [3,7]) and 22 (source_ids [2,4]) into master id=1, source_ids=[22,3,7,2,4].

- **`abzu/er/metrics.py`** (161 lines) -- `BlockInfo`, `BlockingMetrics`, `IterationMetrics` TypedDicts. `IterationMetrics` includes `overall_reduction_pct` for cumulative tracking from original baseline. `get_all_iteration_metrics()` computes both per-round and overall reduction by tracking original company count from iteration 1. Also provides `get_blocking_metrics()`, `get_matching_metrics()`, and `get_evaluation_metrics()`.

- **`abzu/er/acronyms.py`** (88 lines) -- **Company name normalization.** `get_basename()` uses `cleanco` library for corporate suffix removal (Inc., LLC, etc.). `get_corporate_ending()` extracts the corporate ending. `get_acronyms()` generates acronyms from cleaned company names, filtering multilingual stop words via `get_multilingual_stop_words()`. Used by name-based blocking for key generation.

### 3.3 PySpark ER Modules

- **`abzu/spark/er_block_semantic.py`** (863 lines) -- Full semantic blocking pipeline. Uses subprocess isolation via inline Python scripts (`EMBED_SCRIPT`, `FAISS_SCRIPT`) to avoid PyTorch/FAISS memory conflicts on macOS. Auto-scales target block size by iteration: `effective_target = max(10, target_block_size // iteration)` (tighter clusters in later rounds). Embeds company names, clusters via FAISS, returns detailed per-block cosine distance stats and quality metrics. Generates a 7-section analysis report: (1) Input Data, (2) Clustering Parameters, (3) Cosine Distance Distribution (with percentiles), (4) Block Size Distribution (pre/post split), (5) Levenshtein Distance Analysis via `_compute_block_levenshtein_stats()` and `_compute_normalized_levenshtein()`, (6) Sample Blocks, (7) Recommendations. Builds Spark DataFrame with company structs, splits oversized blocks via `create_split_large_blocks_udtf()` factory from utils. Uses `normalize_company_dataframe()` from schemas to ensure consistent field order.

- **`abzu/spark/er_eval.py`** (601 lines) -- Post-match evaluation with iteration-aware validation. `evaluate_er_matches()` explodes resolved companies, deduplicates exact copies, splits into BAML-processed vs skipped companies. Validates source_uuids against ORIGINAL raw companies data (always iteration 0). For iteration 2+, loads previous iteration output for comparison and validates against ALL historical UUIDs from all previous iterations. Comprehensive metrics: `iteration_input_companies`, `total_original_companies`, `companies_that_went_into_matching`, `skipped_records`, `unique_baml_processed`, `reduction_from_matching`, `ids_dropped_by_baml`. Match skip reason analysis: `singleton_block_count`, `error_recovery_count`, `missing_uuid_recovery_count`, `missing_primary_uuid_count`, `missing_source_uuid_count`. Overall status assessment with PASS/FAIL checks (coverage >= 99.99%, error < 0.01%, overlap < 1.0%). Saves `er_evaluation_metrics.parquet` with all metrics.

- **`abzu/spark/er_block.py`** (848 lines) -- Name-based (heuristic) blocking. Three strategies combined via union: first-word extraction (`get_first_word()`), acronym extraction (`get_acronym()` via `abzu.er.acronyms`), and domain suffix removal (`remove_domain_suffix()` with 50+ TLD suffixes). `build_blocks()` is the main entry point. Uses `normalize_company_dataframe()` for consistent schema and `create_split_large_blocks_udtf()` for oversized block splitting.

- **`abzu/spark/refine_kg.py`** -- Maps relationships to resolved company nodes, optional edge ER.

- **`abzu/spark/schemas.py`** (280 lines) -- SparkDantic bridge between Pydantic/BAML models and Spark StructType schemas. `SparkTicker` and `SparkCompany` extend BAML types with `sparkdantic.SparkModel` for automatic Spark schema generation. Key functions:

  - `get_company_spark_schema()`: Generates Spark StructType from Pydantic, recursively converts IntegerType to LongType via `convert_ints_to_longs()`
  - `normalize_company_dataframe()`: Ensures consistent field order/types across iterations, adds missing fields as null with correct type
  - `validate_block_schema()`: Validates required `BLOCK_FIELDS = ["block_key", "block_key_type", "companies", "block_size"]`
  - `get_matches_schema()`: Returns StructType for reading matches.jsonl with correct BAML types (block_key, resolved_companies array, etc.)
  - `build_udtf_return_type()`: Dynamically builds UDTF return type string from Company schema
  - `get_company_fields_without_blocks()`: Returns Company fields excluding block metadata

- **`abzu/spark/utils.py`** (262 lines) -- Shared PySpark utilities. `create_split_large_blocks_udtf()`: Factory function that creates PySpark UDTF classes dynamically for splitting oversized blocks into chunks. `select_most_common_property()`: Window function-based selection of most common value per group (with longest-string tiebreaker). Also `create_uuid_schema()` and `validate_referential_integrity()` for article processing.

### 3.4 CLI Orchestration

- **`abzu/cli/process/er/all.py`** (353 lines) -- Click command orchestrating block -> match -> eval cycle as a 3-step pipeline with per-stage timing. Options: `--iteration`, `--blocking-method` (name/embed), `--max-block-size`, `--batch-size`, `--local-mode`, `--debug`. Each stage displays metrics on completion: blocking shows input companies, blocks created, largest block, and throughput (companies/sec); matching shows blocks processed, companies in output, skipped/singletons, and throughput (blocks/sec); evaluation shows original vs final companies and total reduction. Prints comprehensive cycle summary with stage breakdown and performance percentages. **Cross-iteration summary table**: `get_all_iteration_metrics()` generates a formatted table showing Iteration, Blocks, Companies In, Companies Out, per-round Reduction %, and cumulative Overall % across all iterations.

### 3.5 Patterns to Preserve in SERF

1. **UUID-to-integer ID mapping** -- Reduces context size, caches mapping locally, reconstructs after LLM returns. Two-phase missing recovery ensures no companies are silently dropped.
2. **Async concurrent processing** -- `asyncio.Semaphore` for rate limiting, `tqdm` progress bars
3. **Few-shot learning** -- Provides examples for complex merge logic (ID tracking, source_ids accumulation)
4. **Subprocess isolation** -- PyTorch + FAISS in subprocesses via inline Python scripts to avoid memory conflicts on macOS
5. **Parquet for Spark, JSONL for streaming** -- Columnar for structured data, line-delimited for LLM results
6. **Configuration-driven paths** -- Centralized config.yml with `{iteration}` placeholders
7. **Comprehensive source tracking** -- `source_uuids` field enables multi-round ER with complete merge lineage
8. **Iterative convergence** -- Per-iteration reduction tracking, cumulative metrics from baseline
9. **match_skip_history tracking** -- `match_skip_history: list[int]` records which iterations skipped each company, enabling skip frequency analysis across rounds
10. **match_skip_reason categorization** -- Categorized skip reasons: `singleton_block` (only company in block), `error_recovery` (LLM call failed), `missing_in_match_output` (LLM dropped the company, recovered via UUID tracking), `missing_primary_uuid`, `missing_source_uuid`
11. **Backup file management** -- `backup_file()` creates timestamped backups before overwriting, keeping only most recent backup for both files and directories (Parquet)
12. **Block schema validation** -- `validate_block_schema()` catches schema drift early by validating required BLOCK_FIELDS before processing
13. **Schema normalization** -- `normalize_company_dataframe()` ensures consistent field order/types across iterations, adds missing fields as null with correct BAML-derived types
14. **Cross-iteration UUID validation** -- Evaluation validates source_uuids against ALL historical UUIDs from all previous iterations, not just the current round
15. **Auto-scaling target block size** -- `effective_target = max(10, target_block_size // iteration)` creates tighter clusters in later rounds when remaining duplicates are harder to find
16. **UDTF factory pattern** -- `create_split_large_blocks_udtf()` dynamically creates PySpark UDTF classes with configurable return types and max block sizes
17. **SparkDantic schema bridge** -- `SparkModel` subclasses of Pydantic types automatically generate Spark schemas with IntegerType-to-LongType conversion. In SERF, this works bidirectionally: Pydantic-to-Spark for writing, and Spark-to-Pydantic for auto-generating entity types from input DataFrames
18. **Comprehensive analysis reports** -- 7-section blocking reports with input data, clustering params, distance distribution (percentiles), block size stats, Levenshtein analysis, sample blocks, and recommendations
19. **Per-stage timing and throughput** -- CLI tracks wall-clock time and throughput (companies/sec, blocks/sec) per pipeline stage, plus cross-iteration summary table
20. **Company name normalization** -- `cleanco` library for corporate suffix removal, multilingual stop word filtering for acronym generation, domain suffix removal for blocking keys

### 3.6 Patterns to Evolve in SERF

1. **BAML types -> fresh DSPy Pydantic types** -- Do NOT reuse Abzu's BAML-generated types (`abzu.baml_client.types`). Those types were auto-generated by BAML for a company-specific domain. SERF needs fresh, domain-agnostic Pydantic classes designed for DSPy signatures. Study Abzu's types for field patterns and ER metadata (source_ids, source_uuids, match_skip, match_skip_history), but build new classes from scratch.
2. **BAML templates -> DSPy signatures** -- Replace BAML templates with DSPy `Signature` classes + `BAMLAdapter` for output formatting. The signatures should use the new Pydantic types as input/output fields.
3. **Auto-generate entity types from DataFrames** -- When a user provides a PySpark DataFrame (or Parquet/CSV file), SERF should automatically infer Pydantic entity types from the DataFrame schema. This means inspecting `df.schema` (StructType), mapping Spark types to Python types, and generating a Pydantic class with the appropriate fields. The profiler (Section 5.3) identifies which fields are names, identifiers, dates, etc. -- this metadata enriches the generated type with DSPy field descriptions. Use this approach when it simplifies the user experience (e.g., `serf resolve --input data.parquet` should work without the user defining any types). For advanced use cases, users can define their own Pydantic entity types.
4. **Poetry -> uv** -- Replace Poetry with the faster, standards-compliant uv package manager
5. **PySpark 3.5 -> 4.1** -- Leverage VARIANT type, Spark Connect, Python Data Source API
6. **Parquet -> Iceberg** -- Add ACID transactions, time travel, schema evolution for iterative ER
7. **Company-only -> domain-agnostic** -- Generalize entity types beyond companies
8. **Manual orchestration -> agentic** -- DSPy ReAcT agents control the pipeline dynamically

---

## 4. Architecture and Technology Stack

### 4.1 Core Technologies

| Component              | Technology                                                                                              | Rationale                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Package Manager**    | **uv** (replacing Poetry)                                                                               | 10-100x faster, PEP 621 compliant, built-in Python version management   |
| **Data Processing**    | **PySpark 4.1**                                                                                         | Python-first, Spark Connect, VARIANT type, Arrow UDFs                   |
| **Table Format**       | **Apache Iceberg**                                                                                      | ACID transactions, time travel for iteration tracking, schema evolution |
| **LLM Framework**      | **DSPy 3.x** with `BAMLAdapter`                                                                         | Programming-not-prompting, automatic optimization, Pydantic integration |
| **Embeddings**         | **Qwen3-Embedding** via sentence-transformers                                                           | Top MTEB leaderboard, multilingual support                              |
| **Vector Search**      | **FAISS IndexIVFFlat**                                                                                  | Fast approximate nearest neighbor for semantic blocking                 |
| **Graph Processing**   | **GraphFrames**                                                                                         | Connected components for transitive closure of match decisions          |
| **LLM Models**         | **Gemini 2.0 Flash** (production), **Gemini 2.5 Flash Lite** (lightweight), **Claude Opus 4** (quality) | Flash for cost efficiency at temp=0; Opus for difficult cases           |
| **CLI**                | **Click**                                                                                               | Existing SERF pattern, `show_default=True`                              |
| **Type Checking**      | **zuban** (mypy-compatible)                                                                             | Existing SERF pattern                                                   |
| **Linting/Formatting** | **Ruff** (replacing black/isort/flake8)                                                                 | Single tool, 10-100x faster                                             |

### 4.2 Migration from Poetry to uv

Convert `pyproject.toml` from `[tool.poetry]` to PEP 621 `[project]` format:

```toml
[project]
name = "serf"
version = "0.1.0"
description = "SERF: Agentic Semantic Entity Resolution Framework"
readme = "README.md"
requires-python = ">=3.12"
license = {text = "Apache-2.0"}
authors = [{name = "Russell Jurney", email = "rjurney@graphlet.ai"}]
dependencies = [
    "dspy-ai>=3.0.3",
    "click>=8.1",
    "pyyaml>=6.0",
    "pyspark>=4.0,<5.0",
    "sentence-transformers>=5.1",
    "faiss-cpu>=1.9",
    "graphframes>=0.8",
    "pyiceberg>=0.8",
]

[project.scripts]
serf = "serf.cli.main:cli"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-asyncio>=1.0",
    "ruff>=0.11",
    "zuban>=0.0.23",
    "pre-commit>=4.0",
]
```

Replace `poetry install` with `uv sync`, `poetry run` with `uv run`, `poetry add` with `uv add`.

### 4.3 PySpark 4.1 Features to Leverage

- **VARIANT type**: Handle heterogeneous entity schemas from different sources without rigid struct definitions
- **Spark Connect**: Decouple SERF CLI (lightweight Python) from compute cluster
- **Arrow-optimized UDFs**: Efficient embedding computation via `pandas_udf`
- **`mapInPandas`**: Process blocks independently with full Python library access (DSPy, FAISS)
- **Python Data Source API**: Custom data sources written in pure Python

### 4.4 Apache Iceberg Integration

Iceberg provides critical capabilities for iterative ER:

```
local.serf.raw_entities        -- Raw ingested entities from all sources
local.serf.blocked_entities    -- Entities with block assignments (per iteration)
local.serf.match_decisions     -- Match decisions with scores and reasoning
local.serf.resolved_entities   -- Entities after merging (per iteration)
local.serf.edges               -- Relationships between entities
local.serf.resolved_edges      -- Edges after edge resolution
```

- **Time travel**: Compare results across iterations (`SELECT * FROM table VERSION AS OF <snapshot>`)
- **MERGE INTO**: Atomic updates of canonical IDs after merge decisions
- **Schema evolution**: Add fields as new data sources are integrated without rewriting data
- **Incremental processing**: Process only new/modified records via snapshot diffing

### 4.5 Project Structure

```
serf/
  src/
    serf/
      __init__.py
      config.py                    # Config management (exists)
      logs.py                      # Logging (exists)
      cli/
        __init__.py
        main.py                    # CLI entry point (exists, extend)
      dspy/
        __init__.py
        types.py                   # Fresh Pydantic entity types for DSPy (exists, rewrite)
        type_generator.py          # Auto-generate entity types from DataFrame schemas (NEW)
        baml_adapter.py            # BAMLAdapter for DSPy (exists)
        signatures.py              # DSPy signatures for ER (NEW)
        agents.py                  # DSPy ReAcT agents (NEW)
      block/
        __init__.py
        embeddings.py              # Sentence-transformer embedder (NEW)
        faiss_blocker.py           # FAISS IVF blocking (NEW)
        normalize.py               # Company name normalization: cleanco, acronyms (NEW)
        pipeline.py                # PySpark blocking pipeline (NEW)
      match/
        __init__.py
        uuid_mapper.py             # UUID-to-integer mapping (NEW)
        matcher.py                 # Block-level LLM matching (NEW)
        few_shot.py                # Few-shot example generation (NEW)
      merge/
        __init__.py
        merger.py                  # Field-level merge logic (NEW)
      edge/
        __init__.py
        resolver.py                # Edge deduplication (NEW)
      eval/
        __init__.py
        metrics.py                 # ER quality metrics (NEW)
        benchmarks.py              # Benchmark dataset loading (NEW)
      analyze/
        __init__.py
        profiler.py                # Dataset profiling/analysis (NEW)
        field_detection.py         # Auto-detect common field types (NEW)
      spark/
        __init__.py
        schemas.py                 # SparkDantic bridge, schema validation/normalization (NEW)
        utils.py                   # UDTF factory, common Spark utilities (NEW)
        iceberg.py                 # Iceberg table management (NEW)
        graph.py                   # GraphFrames connected components (NEW)
  tests/
    test_config.py
    test_types.py
    test_type_generator.py
    test_baml_adapter.py
    test_embeddings.py
    test_faiss_blocker.py
    test_uuid_mapper.py
    test_matcher.py
    test_merger.py
    test_edge_resolver.py
    test_metrics.py
    test_profiler.py
    test_field_detection.py
    test_schemas.py
    test_iceberg.py
    test_graph.py
    test_benchmarks.py
    test_signatures.py
    test_agents.py
    test_cli.py
    test_pipeline_integration.py
  docs/
    SERF_LONG_SHOT_PLAN.md         # This document
  config.yml
  pyproject.toml
  .pre-commit-config.yaml
  README.md
  CLAUDE.md
```

---

## 5. Data Model and Type System

**Important: SERF does NOT reuse Abzu's BAML-generated types.** Abzu's types were auto-generated by BAML for a company-specific SEC filing domain. SERF builds fresh Pydantic types designed for DSPy, preserving only the proven ER metadata patterns (source_ids, source_uuids, match_skip, match_skip_history).

### 5.1 Domain-Agnostic Entity Types

SERF should generalize beyond Abzu's Company-only model. The base entity type is inspired by [schema.org](https://schema.org/Person) property conventions:

```python
class Entity(BaseModel):
    """Base entity type for all resolvable entities.

    Domain-specific fields live in `attributes` or in subclasses.
    ER metadata fields (id, uuid, source_ids, etc.) are fixed across all domains.
    """
    id: int
    uuid: Optional[str] = None
    name: str
    description: str
    entity_type: str  # "company", "person", "product", etc.
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_ids: Optional[list[int]] = None
    source_uuids: Optional[list[str]] = None
    match_skip: Optional[bool] = None
    match_skip_reason: Optional[str] = None
    match_skip_history: Optional[list[int]] = None  # iterations that skipped this entity
```

Specialized entity types extend this base. These are **examples** -- users can define their own or let SERF auto-generate them from DataFrame schemas:

```python
class Company(Entity):
    """Company entity with business-specific fields."""
    entity_type: str = "company"
    cik: Optional[str] = None
    ticker: Optional[Ticker] = None
    website_url: Optional[str] = None
    headquarters_location: Optional[str] = None
    jurisdiction: Optional[str] = None
    revenue_usd: Optional[int] = None
    employees: Optional[int] = None
    founded_year: Optional[int] = None
    ceo: Optional[str] = None

class Person(Entity):
    """Person entity inspired by schema.org/Person."""
    entity_type: str = "person"
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    email: Optional[str] = None
    job_title: Optional[str] = None
    works_for: Optional[str] = None
    birth_date: Optional[str] = None
    nationality: Optional[str] = None
```

### 5.1.1 Auto-Generating Entity Types from DataFrames

When the user provides a PySpark DataFrame (or Parquet/CSV file) without a custom entity type, SERF should auto-generate a Pydantic entity class from the DataFrame schema:

```python
def entity_type_from_spark_schema(
    schema: StructType,
    profile: DatasetProfile,
    entity_type_name: str = "AutoEntity",
) -> type[Entity]:
    """Generate a Pydantic Entity subclass from a Spark StructType schema.

    Uses the DatasetProfile to enrich fields with DSPy descriptions
    (e.g., marking a field as "name", "identifier", "date").

    Parameters
    ----------
    schema : StructType
        The Spark schema to convert
    profile : DatasetProfile
        Profiling results identifying field types and roles
    entity_type_name : str
        Name for the generated class

    Returns
    -------
    type[Entity]
        A dynamically created Pydantic subclass of Entity
    """
    ...
```

The type mapping from Spark to Python is straightforward:

| Spark Type            | Python Type           | Notes                       |
| --------------------- | --------------------- | --------------------------- |
| StringType            | str                   | Default for most fields     |
| LongType/IntegerType  | int                   | Numeric identifiers, counts |
| DoubleType/FloatType  | float                 | Revenue, percentages        |
| BooleanType           | bool                  | Flags                       |
| ArrayType(StringType) | list[str]             | Tags, categories            |
| StructType            | nested Pydantic model | Auto-generated recursively  |

ER metadata fields (id, uuid, source_ids, source_uuids, match_skip, match_skip_reason, match_skip_history) are automatically added by the framework -- the user's schema only needs to contain domain fields. The profiler's `FieldProfile` provides DSPy-compatible descriptions for each field.

### 5.2 Block and Match Types

```python
class EntityBlock(BaseModel):
    """A block of entities for matching."""
    block_key: str
    block_key_type: str  # "semantic", "name", "custom"
    block_size: int
    entities: list[Entity]

class MatchDecision(BaseModel):
    """A single match decision between entities."""
    entity_a_id: int
    entity_b_id: int
    is_match: bool
    confidence: float = Field(ge=0, le=1)
    reasoning: str

class BlockResolution(BaseModel):
    """Result of resolving all matches within a block."""
    matches: list[MatchDecision]
    resolved_entities: list[Entity]
    was_resolved: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
```

### 5.3 Dataset Profiling and Auto-Detection

Before running ER, SERF should profile the input dataset to auto-detect common field types. Inspired by schema.org property patterns:

```python
class FieldProfile(BaseModel):
    """Profile of a single field in the dataset."""
    name: str
    inferred_type: str  # "name", "email", "phone", "address", "url", "identifier", "text", "numeric", "date"
    completeness: float  # fraction of non-null values
    uniqueness: float  # fraction of unique values
    sample_values: list[str]
    is_blocking_candidate: bool
    is_matching_feature: bool

class DatasetProfile(BaseModel):
    """Profile of the entire input dataset."""
    record_count: int
    field_profiles: list[FieldProfile]
    recommended_blocking_fields: list[str]
    recommended_matching_fields: list[str]
    estimated_duplicate_rate: float
```

The profiler should auto-detect:

- **Name fields**: High cardinality, mixed case, common name patterns
- **Email/URL fields**: Regex pattern matching
- **Address fields**: Contains numbers + street/city/state patterns
- **Identifier fields**: High uniqueness, alphanumeric patterns (SSN, EIN, CIK, ticker)
- **Date fields**: ISO 8601 or common date patterns
- **Numeric fields**: Revenue, employee count, etc.

---

## 6. User Interfaces

SERF exposes three interfaces for different user personas:

### 6.1 CLI (Command-Line Interface)

For data engineers running ER pipelines. Extends the existing Click CLI:

```bash
# Profile a dataset before ER
serf analyze --input data/companies.parquet

# Run the full ER pipeline with iteration tracking
serf resolve --input data/companies.parquet --output data/resolved/ --iteration 1

# Run individual phases
serf block --input data/companies.parquet --output data/blocks/ --method semantic --target-block-size 50
serf match --input data/blocks/ --output data/matches/ --iteration 1 --batch-size 10
serf eval --input data/matches/ --output data/resolved/ --iteration 1

# Run edge resolution for knowledge graphs
serf edges --input data/resolved/ --output data/edges/

# Load and evaluate against a benchmark dataset
serf benchmark --dataset walmart-amazon --output data/benchmark_results/

# Download benchmark datasets
serf download --dataset walmart-amazon --output data/datasets/
```

Key CLI design principles (from CLAUDE.md):

- Use `@click.command(context_settings={"show_default": True})` -- never put defaults in help strings
- Separate logic under `serf.block.*`, `serf.match.*` etc. from CLI code in `serf.cli.*`
- All paths come from `config.yml` via `from serf.config import config`
- The `--iteration` argument drives iterative ER with `{iteration}` path templates

### 6.2 PySpark DataFrame API

**The primary interface is DataFrame-in, DataFrame-out.** Users pass any `pyspark.sql.DataFrame` and get a resolved DataFrame back. Pydantic types are an internal implementation detail -- the user never needs to define or see them.

The internal flow for any DataFrame:

1. **Profile**: Inspect `df.schema` (StructType) + sample data to build a `DatasetProfile` identifying field roles (name, identifier, date, etc.)
2. **Generate types**: Auto-generate a Pydantic `Entity` subclass from the schema + profile (Section 5.1.1). This gives the LLM structured field descriptions and validates output.
3. **Generate Spark schema**: Use SparkDantic to derive the output Spark schema from the generated Pydantic type (with ER metadata fields added: uuid, source_ids, source_uuids, match_skip, etc.)
4. **Block/Match/Merge**: All operations work on DataFrames. The Pydantic types are used internally for LLM serialization (block rows → JSON for DSPy) and deserialization (LLM output → validated Pydantic → DataFrame rows).
5. **Return DataFrame**: The resolved output is a standard `pyspark.sql.DataFrame` with the original columns plus ER metadata columns.

```python
from serf.block import SemanticBlocker
from serf.match import EntityMatcher
from serf.eval import evaluate_resolution

# Load ANY DataFrame -- no type definitions needed
companies = spark.read.parquet("data/companies.parquet")

# Block (works on raw DataFrame, embeds the "name" column by default)
blocker = SemanticBlocker(target_block_size=50)
blocks = blocker.transform(companies)  # returns DataFrame with block_key, entities array

# Match and merge (internally: profile → generate types → serialize → LLM → deserialize)
matcher = EntityMatcher(model="gemini/gemini-2.0-flash", batch_size=10)
resolved = matcher.resolve(blocks)  # returns DataFrame with original cols + ER metadata

# Evaluate
metrics = evaluate_resolution(resolved, companies)
print(f"Reduction: {metrics.reduction_pct:.1f}%")

# Write to Iceberg
resolved.writeTo("local.serf.resolved_entities").overwritePartitions()
```

For advanced users who want control over the entity type:

```python
from serf.dspy.types import Entity

class Product(Entity):
    """Custom entity type with domain-specific fields."""
    entity_type: str = "product"
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None

# Pass explicit type -- skips auto-generation
matcher = EntityMatcher(model="gemini/gemini-2.0-flash", entity_type=Product)
resolved = matcher.resolve(blocks)
```

### 6.3 DSPy Module Interface

For ML engineers building custom ER pipelines with DSPy optimization:

```python
import dspy
from serf.dspy.signatures import BlockMatch, EntityMerge
from serf.dspy.agents import ERAgent
from dspy.adapters.baml_adapter import BAMLAdapter

# Configure DSPy
lm = dspy.LM("gemini/gemini-2.0-flash", api_key=GEMINI_API_KEY)
dspy.configure(lm=lm, adapter=BAMLAdapter())

# Use individual signatures
matcher = dspy.ChainOfThought(BlockMatch)
result = matcher(block_records=block_json, schema_info=schema_desc)

# Use the agentic resolver
agent = ERAgent(tools=[blocker, matcher, evaluator], max_iterations=5)
resolved = agent(dataset=companies)

# Optimize with DSPy compilers
from dspy.teleprompt import MIPROv2
optimizer = MIPROv2(metric=f1_metric, auto="medium")
optimized_agent = optimizer.compile(agent, trainset=labeled_pairs)
```

---

## 7. Agentic Semantic Entity Resolution

### 7.1 Phase 0: Agentic Control with DSPy

The SERF README states: "DSPy agents control all phases dynamically." This is implemented via DSPy's `ReAct` pattern:

```python
class ERAgent(dspy.Module):
    """Agentic entity resolution controller.

    Uses ReAct (Reasoning + Acting) to dynamically orchestrate
    the blocking -> matching -> merging -> evaluation pipeline.
    """

    def __init__(self, tools, max_iterations=5, convergence_threshold=0.01):
        super().__init__()
        self.react = dspy.ReAct(
            ERControlSignature,
            tools=tools,
            max_iters=max_iterations * 3,  # allow multiple tool calls per iteration
        )
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def forward(self, dataset, entity_type="company"):
        return self.react(
            dataset_description=describe_dataset(dataset),
            entity_type=entity_type,
            max_iterations=self.max_iterations,
            convergence_threshold=self.convergence_threshold,
        )
```

The agent has access to tools for each pipeline phase:

- `profile_dataset(path)` -- Run dataset profiling
- `create_blocks(path, method, params)` -- Run semantic or name-based blocking
- `match_blocks(blocks_path, iteration)` -- Run LLM matching on blocks
- `evaluate_matches(matches_path, raw_path)` -- Compute quality metrics
- `check_convergence(metrics)` -- Decide if another round is needed

### 7.2 Phase 1: Semantic Blocking

```
Raw Entities -> Embed (Qwen3) -> FAISS IVF Cluster -> Blocks
```

- Embed entity records using sentence-transformers (Qwen3 or multilingual-e5-base)
- Cluster using FAISS `IndexIVFFlat` with configurable `target_block_size`
- Auto-scale target_block_size by iteration (tighter in later rounds)
- Subprocess isolation for PyTorch/FAISS to avoid memory conflicts
- Compute per-block Levenshtein distance statistics for quality monitoring
- Split oversized blocks into sub-blocks

### 7.3 Phase 2: Schema Alignment + Matching + Merging

All three operations in a single DSPy signature. The internal flow from DataFrame to LLM and back:

1. **DataFrame → Pydantic**: Each block's entity rows are converted to the auto-generated (or user-provided) Pydantic Entity subclass instances, then serialized to JSON
2. **Pydantic → DSPy**: The JSON block + auto-generated schema description are passed to the `BlockMatch` signature
3. **DSPy → LLM**: DSPy + BAMLAdapter formats the structured prompt and sends to Gemini
4. **LLM → Pydantic**: BAMLAdapter parses the structured output into `BlockResolution` Pydantic instances, validating all fields
5. **Pydantic → DataFrame**: Resolved entities are converted back to Spark rows using the SparkDantic-derived schema

```python
class BlockMatch(dspy.Signature):
    """Examine all entities in a block, identify duplicates, and merge them.

    For each group of matching entities:
    1. Select the entity with the LOWEST input id as the master record
    2. Collect ALL OTHER input ids into source_ids
    3. Merge all source_ids from inputs into a single output source_ids list
    4. Choose the most complete field values across all matched records
    5. Return ALL entities (merged + non-matched)
    """
    block_records: str = dspy.InputField(desc="JSON array of entity records in this block")
    schema_info: str = dspy.InputField(desc="Auto-generated description of entity fields and their roles from the Pydantic type and DatasetProfile")
    few_shot_examples: str = dspy.InputField(desc="Examples of correct merge behavior")
    resolution: BlockResolution = dspy.OutputField()
```

The `schema_info` is generated automatically from the Pydantic entity type (which itself may have been auto-generated from the input DataFrame schema via Section 5.1.1). This means `serf resolve --input data.parquet` works end-to-end without the user defining any types -- the DataFrame schema drives everything.

Key implementation details from Abzu:

- **UUID-to-integer mapping**: Map UUIDs to consecutive integers before LLM call, map back after
- **Source UUID caching**: Strip `source_uuids` before sending to LLM (can have 1000+ entries), restore from cache after
- **MDM-style master record**: Lowest input ID becomes master, all others become `source_ids`
- **Two-phase missing recovery**: (1) Add back missing UUIDs to existing output companies' source_uuids, (2) Recover entire missing companies with `match_skip_reason = "missing_in_match_output"`
- **New UUID generation**: Each resolved entity gets a new UUID to distinguish from inputs
- **match_skip_history tracking**: Record which iterations skipped each entity for cross-round analysis
- **Error recovery**: Mark failed blocks with `match_skip_reason = "error_recovery"` and pass through original companies

### 7.4 Phase 3: Edge Resolution (Knowledge Graphs)

For knowledge graphs, merging nodes creates duplicate edges:

```
Block edges by GROUP BY (src, dst, type)
-> Send edge blocks to LLM for intelligent merging
-> Output: deduplicated edges with merged attributes
```

### 7.5 Transitive Closure with GraphFrames

After pairwise match decisions, use GraphFrames connected components to find all transitively connected entities:

```python
from graphframes import GraphFrame

graph = GraphFrame(vertices_df, match_edges_df)
spark.sparkContext.setCheckpointDir("data/checkpoints")
components = graph.connectedComponents()
# component column = canonical entity ID
```

If GraphFrames is unavailable for PySpark 4.x, implement manual connected components via iterative min-label propagation (see Abzu code guide section).

### 7.6 Iterative Convergence

SERF runs multiple rounds until convergence:

1. **Round 1**: Initial blocking (broader similarity threshold). Merges high-confidence matches. Typical reduction: 20-40%.
2. **Round 2**: Re-embed merged records, re-block. New matches visible as merged records pull closer in embedding space. Additional 10-20%.
3. **Round 3+**: Diminishing returns. Each round resolves fewer new matches.
4. **Convergence**: Stop when reduction per round falls below threshold (e.g., 1%).

Per-iteration metrics tracked (from Abzu's `IterationMetrics` and `er_eval.py`):

- Companies in / companies out (both per-iteration and from original baseline)
- Per-round reduction percentage and cumulative overall reduction from iteration 1
- Block count and size distribution (pre/post split)
- UUID coverage validation against ALL historical UUIDs from all previous iterations
- Match skip reason distribution: singleton_block, error_recovery, missing_in_match_output
- Skip frequency analysis (how many iterations each entity has been skipped)
- Overall PASS/FAIL assessment (coverage >= 99.99%, error < 0.01%)
- Cross-iteration summary table with per-round and overall reduction %

### 7.7 DSPy Optimization

Use DSPy compilers to optimize ER prompts:

- **`BootstrapFewShot`**: Auto-select best few-shot examples from labeled training data
- **`MIPROv2`**: Jointly optimize instruction text and few-shot examples
- **`BootstrapFinetune`**: Use expensive model (Gemini Pro) traces to fine-tune cheaper model (Gemini Flash) for production deployment
- **Metric function**: F1 score on a validation set of labeled entity pairs

---

## 8. Standard ER Benchmark Datasets

SERF must be rigorously evaluated against standard benchmarks. The following datasets should be downloadable via `serf download` and evaluable via `serf benchmark`:

### 8.1 Primary Benchmarks

| Dataset            | Domain        | Left  | Right  | Matches | Difficulty | Best F1      | Source                                                                                           |
| ------------------ | ------------- | ----- | ------ | ------- | ---------- | ------------ | ------------------------------------------------------------------------------------------------ |
| **Walmart-Amazon** | Products      | 2,554 | 22,074 | 962     | Hard       | ~87% (Ditto) | [DeepMatcher](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)                |
| **Abt-Buy**        | Products      | 1,081 | 1,092  | 1,097   | Hard       | ~89% (Ditto) | [DeepMatcher](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)                |
| **Amazon-Google**  | Products      | 1,363 | 3,226  | 1,300   | Hard       | ~76% (Ditto) | [DeepMatcher](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)                |
| **DBLP-ACM**       | Bibliographic | 2,616 | 2,294  | 2,224   | Easy       | ~99% (Ditto) | [Leipzig](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution) |
| **DBLP-Scholar**   | Bibliographic | 2,616 | 64,263 | 5,347   | Medium     | ~96% (Ditto) | [Leipzig](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution) |

### 8.2 Additional Benchmarks

| Dataset          | Domain              | Size           | Source                                                                 |
| ---------------- | ------------------- | -------------- | ---------------------------------------------------------------------- |
| **WDC Products** | E-commerce products | 5K-300K pairs  | [Web Data Commons](http://webdatacommons.org/largescaleproductcorpus/) |
| **Cora**         | Citations           | 1,295 records  | McCallum et al. 2000                                                   |
| **FEBRL**        | Synthetic persons   | 5K-10K records | Christen 2008                                                          |

### 8.3 Benchmark Evaluation Metrics

- **Precision**: Fraction of predicted matches that are true matches
- **Recall**: Fraction of true matches that are found
- **F1 Score**: Harmonic mean of precision and recall
- **Pair Completeness (PC)**: Fraction of true matches retained after blocking
- **Reduction Ratio (RR)**: `1 - (pairs after blocking / total possible pairs)`
- **Cluster F1**: F1 computed at the cluster level (accounts for transitivity)

### 8.4 Benchmark Implementation

```python
class BenchmarkDataset:
    """Standard ER benchmark dataset."""
    name: str
    left_table: DataFrame
    right_table: DataFrame
    ground_truth: DataFrame  # (left_id, right_id) pairs
    metadata: dict

    @classmethod
    def download(cls, name: str, output_dir: str) -> "BenchmarkDataset":
        """Download and prepare a benchmark dataset."""
        ...

    def evaluate(self, predictions: DataFrame) -> dict[str, float]:
        """Evaluate predictions against ground truth."""
        ...
```

---

## 9. Scaling Analysis

### 9.1 The Quadratic Comparison Problem

| Records (N) | Naive Pairs | 99% Blocking RR | 99.9% Blocking RR |
| ----------- | ----------- | --------------- | ----------------- |
| 1,000       | 499,500     | 4,995           | 500               |
| 10,000      | ~50M        | ~500K           | ~50K              |
| 100,000     | ~5B         | ~50M            | ~5M               |
| 1,000,000   | ~500B       | ~5B             | ~500M             |

Blocking is essential. Without it, LLM-based ER is economically impossible beyond a few thousand records.

### 9.2 Cost Analysis

**Block-level matching** (SERF's approach, avg block size 20, ~500 blocks for 10K records):

- 500 blocks x 2,500 tokens avg = 1.25M tokens total

| Model            | Input Cost | Output Cost | Total (10K records) |
| ---------------- | ---------- | ----------- | ------------------- |
| Gemini 2.0 Flash | $0.10/1M   | $0.40/1M    | **~$0.63**          |
| Gemini 2.5 Pro   | $1.25/1M   | $10.00/1M   | **~$6.56**          |
| Claude Sonnet 4  | $3.00/1M   | $15.00/1M   | **~$11.25**         |
| Claude Opus 4    | $15.00/1M  | $75.00/1M   | **~$56.25**         |
| GPT-4o           | $2.50/1M   | $10.00/1M   | **~$8.13**          |

**Key insight**: Block-level matching reduces costs by 20-40x compared to pairwise LLM matching. At $0.63 per 10K records with Gemini Flash, SERF can process 1M records for ~$63.

### 9.3 Recommended Cascade Architecture

For production at scale, implement a cost-optimization cascade:

1. **Embedding similarity filter** (free): Skip blocks where all pairwise embedding similarities < threshold
2. **Gemini Flash** ($0.10/1M tokens): Process medium-difficulty blocks
3. **Gemini Pro / Claude Sonnet** ($3-10/1M tokens): Process the hardest blocks where Flash confidence is low

This cascade can reduce costs by 80%+ while maintaining F1 within 1-2% of full expensive-model coverage.

### 9.4 Latency and Parallelism

- LLM inference: ~1-10 seconds per block
- 500 blocks with 50 concurrent requests: ~10-100 seconds wall-clock time
- PySpark `mapInPandas` distributes across cluster workers
- `asyncio.Semaphore` within each worker controls API rate limits

### 9.5 Multi-Round Convergence Efficiency

After 3 rounds with merge factor 0.8 per round:

- Dataset shrinks to 0.8^3 = 51.2% of original
- Comparison pairs shrink to 0.8^6 = 26.2% of original
- Each round is cheaper than the last due to smaller dataset

### 9.6 Overnight Build Budget Constraint

**Hard budget: $100 total Gemini API spend for the overnight build.**

A `GEMINI_API_KEY` environment variable will be provided. The agent must stay within budget by following these rules:

1. **Use Gemini 2.0 Flash exclusively** for all ER pipeline operations (blocking analysis, matching, merging, edge resolution). At $0.10/$0.40 per 1M input/output tokens, this allows ~160M+ input tokens -- more than enough for iterative ER across all three benchmark datasets.

2. **Gemini 2.5 Pro is allowed ONLY for generating validation data** -- high-quality labeled match/non-match pairs and few-shot examples that will be used to evaluate and optimize the pipeline. Limit Gemini 2.5 Pro to **fewer than 2,000 API calls** total. At ~2,500 tokens per call with $1.25/$10.00 per 1M input/output tokens, 2K calls costs roughly $50 -- leaving ample headroom for Flash usage.

3. **Never use Claude, GPT-4o, or any non-Gemini model** for pipeline operations during the build. The DSPy signatures and pipeline code should be model-agnostic, but all actual LLM calls during this build session must go through Gemini.

4. **Track token usage** by logging input/output token counts from API responses. If cumulative spend approaches $80, stop making Gemini 2.5 Pro calls and finish remaining work with Flash only.

| Use Case                       | Model            | Max Calls                 | Est. Cost  |
| ------------------------------ | ---------------- | ------------------------- | ---------- |
| ER pipeline (match/merge/edge) | Gemini 2.0 Flash | Unlimited (within budget) | ~$10-30    |
| Validation data generation     | Gemini 2.5 Pro   | < 2,000                   | ~$50       |
| **Total**                      |                  |                           | **< $100** |

---

## 10. Implementation Plan

The following ordered steps should be executed by the Cursor Agent. Each step produces working, tested code.

### Step 1: Project Infrastructure (30 min)

1. Convert `pyproject.toml` from Poetry to uv (PEP 621 format)
2. Update `.pre-commit-config.yaml` to use Ruff instead of black/isort/flake8
3. Update `config.yml` with ER pipeline paths, iteration templates, model configs, benchmark dataset URLs
4. Create all module directories with `__init__.py` files
5. Update `CLAUDE.md` to reflect new tooling (uv, Ruff)

### Step 2: Core Type System (1.5 hr)

1. Rewrite `src/serf/dspy/types.py` from scratch with fresh, DSPy-appropriate Pydantic types -- do NOT copy Abzu's BAML-generated types. Build domain-agnostic `Entity` base class with ER metadata (id, uuid, source_ids, source_uuids, match_skip, match_skip_reason, match_skip_history) + example `Company`, `Person` specializations
2. Add `EntityBlock`, `MatchDecision`, `BlockResolution` types
3. Add `FieldProfile`, `DatasetProfile` types for dataset analysis
4. Add `IterationMetrics`, `BlockingMetrics` TypedDicts
5. Create `src/serf/dspy/type_generator.py` -- `entity_type_from_spark_schema()` that auto-generates a Pydantic Entity subclass from a Spark StructType + DatasetProfile. Maps Spark types to Python types, adds ER metadata fields automatically, generates DSPy field descriptions from profiling results
6. Write exhaustive unit tests: `tests/test_types.py`, `tests/test_type_generator.py`

### Step 3: DSPy Signatures and Adapter (1 hr)

1. Create `src/serf/dspy/signatures.py` with DSPy signatures:
   - `BlockMatch` -- match entire blocks
   - `EntityMerge` -- merge matched entities
   - `EdgeResolve` -- merge duplicate edges
   - `AnalyzeDataset` -- profile and recommend ER strategy
2. Verify `BAMLAdapter` works with new Pydantic types
3. Write unit tests: `tests/test_signatures.py`, `tests/test_baml_adapter.py`

### Step 4: Embeddings and Blocking (1.5 hr)

1. Create `src/serf/block/embeddings.py` -- `EntityEmbedder` class (generalized from Abzu's `CompanyEmbedder`)
2. Create `src/serf/block/faiss_blocker.py` -- `FAISSBlocker` class (ported from Abzu)
3. Create `src/serf/block/normalize.py` -- Entity name normalization (cleanco for company suffixes, acronym generation, domain suffix removal, multilingual stop word filtering)
4. Create `src/serf/block/pipeline.py` -- PySpark blocking pipeline with subprocess isolation, auto-scaling target block size by iteration, 7-section analysis report, UDTF-based block splitting
5. Write unit tests: `tests/test_embeddings.py`, `tests/test_faiss_blocker.py`, `tests/test_normalize.py`

### Step 5: UUID Mapping and Matching (2 hr)

1. Create `src/serf/match/uuid_mapper.py` -- `UUIDMapper` class (ported from Abzu)
2. Create `src/serf/match/matcher.py` -- `EntityMatcher` with async block processing, semaphore rate limiting
3. Create `src/serf/match/few_shot.py` -- Few-shot example generation
4. Write unit tests: `tests/test_uuid_mapper.py`, `tests/test_matcher.py`

### Step 6: Evaluation and Metrics (1 hr)

1. Create `src/serf/eval/metrics.py` -- Precision, recall, F1, pair completeness, reduction ratio
2. Create `src/serf/eval/benchmarks.py` -- Benchmark dataset download/loading/evaluation
3. Write unit tests: `tests/test_metrics.py`, `tests/test_benchmarks.py`

### Step 7: Dataset Analysis (1 hr)

1. Create `src/serf/analyze/profiler.py` -- Dataset profiling with PySpark
2. Create `src/serf/analyze/field_detection.py` -- Auto-detect field types using regex patterns and statistical analysis
3. Write unit tests: `tests/test_profiler.py`, `tests/test_field_detection.py`

### Step 8: Edge Resolution (45 min)

1. Create `src/serf/edge/resolver.py` -- Async edge resolution (ported from Abzu)
2. Write unit tests: `tests/test_edge_resolver.py`

### Step 9: Spark Integration (1.5 hr)

1. Create `src/serf/spark/schemas.py` -- SparkDantic bridge: SparkModel subclasses of Pydantic types, `get_entity_spark_schema()` with IntegerType-to-LongType conversion, `normalize_entity_dataframe()` for consistent field order/types, `validate_block_schema()` for early schema drift detection, `get_matches_schema()` for reading matches JSONL, `build_udtf_return_type()` for dynamic UDTF return types
2. Create `src/serf/spark/utils.py` -- `create_split_large_blocks_udtf()` factory for block splitting, `select_most_common_property()` window function utility
3. Create `src/serf/spark/iceberg.py` -- Iceberg table creation, time travel, merge-into operations
4. Create `src/serf/spark/graph.py` -- GraphFrames connected components + manual fallback
5. Write unit tests: `tests/test_schemas.py`, `tests/test_iceberg.py`, `tests/test_graph.py`

### Step 10: DSPy Agents (1 hr)

1. Create `src/serf/dspy/agents.py` -- `ERAgent` with ReAct pattern, tool definitions
2. Implement convergence checking, dynamic parameter adjustment
3. Write unit tests: `tests/test_agents.py`

### Step 11: CLI (1 hr)

1. Extend `src/serf/cli/main.py` with commands:
   - `serf analyze` -- Dataset profiling
   - `serf resolve` -- Full pipeline orchestration
   - `serf block` -- Blocking only
   - `serf match` -- Matching only
   - `serf eval` -- Evaluation only
   - `serf edges` -- Edge resolution
   - `serf benchmark` -- Run against benchmarks
   - `serf download` -- Download benchmark datasets
2. Write unit tests: `tests/test_cli.py`

### Step 12: Integration Testing (1.5 hr)

1. Create `tests/test_pipeline_integration.py` -- End-to-end pipeline test with small synthetic dataset
2. Test iterative convergence (3 rounds)
3. Test Iceberg time travel between iterations
4. Test benchmark evaluation on three datasets covering easy, medium, and hard difficulties:
   - **DBLP-ACM** (easy, bibliographic) -- Baseline sanity check, expect ~99% F1
   - **DBLP-Scholar** (medium, bibliographic) -- Tests scale (64K right-side records) and fuzzy matching
   - **Walmart-Amazon** (hard, products) -- Tests cross-schema matching with very different field formats and 22K right-side records

### Step 13: Documentation and Cleanup (30 min)

1. Update `README.md` with new CLI commands, API examples, benchmark results
2. Update `config.yml` with all new configuration keys
3. Run `ruff check --fix` and `ruff format` on all files
4. Run `zuban` type checking
5. Run `pre-commit run --all-files`
6. Final test run: `uv run pytest tests/`

---

## 11. Testing Strategy

### 11.1 Unit Tests

Every module gets exhaustive unit tests. Tests should:

- Use `pytest` style (no test classes)
- Include type hints on all functions
- Use fixtures for setup/teardown
- Use `pytest-asyncio` for async tests
- Mock LLM calls (don't make real API calls in unit tests)
- Test edge cases: empty blocks, single-entity blocks, oversized blocks
- Test UUID mapping roundtrip consistency
- Test metric calculations with known inputs/outputs

### 11.2 Integration Tests

- **Pipeline test**: Synthetic dataset through block -> match -> eval -> iterate
- **Benchmark tests**: Run against all three benchmark datasets from the start:
  - **DBLP-ACM** (easy) -- Verify basic pipeline correctness and metrics computation
  - **DBLP-Scholar** (medium) -- Verify handling of scale asymmetry (2.6K vs 64K records)
  - **Walmart-Amazon** (hard) -- Verify cross-schema matching with heterogeneous product data
- **Iceberg test**: Write/read/time-travel with local Iceberg catalog
- **GraphFrames test**: Connected components on known graph structure

### 11.3 Test Data

Create synthetic test datasets in `tests/fixtures/`:

- 100 synthetic company records with known duplicates
- Known ground truth match pairs
- Pre-computed embeddings for deterministic blocking tests
- Sample LLM responses for mock matching tests

---

## 12. References

### Core Blog Posts

1. Jurney, R. (2024). "The Rise of Semantic Entity Resolution." _Towards Data Science_. [link](https://towardsdatascience.com/the-rise-of-semantic-entity-resolution/)
2. Jurney, R. (2024). "Semantic Entity Resolution Demo." _Graphlet AI Blog_. [link](https://blog.graphlet.ai/semantic-entity-resolution-demo-9a08c851b45f)
3. Jurney, R. (2024). "Knowledge Graph Factories." _Graphlet AI Blog_. [link](https://blog.graphlet.ai/knowledge-graph-factories-f50466fb7512)
4. Jonas, J. (2024). "Entity Resolution and Generative AI." _Senzing_. [link](https://senzing.com/entity-resolution-generative-ai/)

### Foundational Papers

5. Fellegi, I.P. and Sunter, A.B. (1969). "A Theory for Record Linkage." _JASA_.
6. Christen, P. (2012). _Data Matching_. Springer.
7. Khattab, O. et al. (2024). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." _ICLR 2024_.

### LLMs for Entity Resolution

8. Peeters, R. & Bizer, C. (2024). "Entity Matching using Large Language Models." _EDBT 2024_. arXiv:2310.11244.
9. Peeters, R. & Bizer, C. (2024). "Match, Compare, or Select? An Investigation of LLMs for Entity Matching." _ISWC 2024_. arXiv:2405.16884.
10. Narayan, A. et al. (2022). "Can Foundation Models Wrangle Your Data?" _VLDB 2022_.
11. Li, Y. et al. (2021). "Ditto: A Simple and Efficient Entity Matching Framework." _VLDB 2021_.
12. Zhang, H. et al. (2023). "JELLYFISH: A Large Language Model for Data Preprocessing." arXiv:2312.01678.
13. Tu, J. et al. (2023). "Unicorn: A Unified Multi-Tasking Model for Entity Resolution." _VLDB 2023_.

### Blocking

14. Papadakis, G. et al. (2020). "Blocking and Filtering Techniques for Entity Resolution: A Survey." _ACM Computing Surveys_.
15. Thirumuruganathan, S. et al. (2021). "Deep Learning Blocking for Entity Matching." _VLDB 2021_.

### Benchmarks

16. Mudgal, S. et al. (2018). "Deep Learning for Entity Matching: A Design Space Exploration." _SIGMOD 2018_.
17. Kopcke, H. et al. (2010). "Evaluation of entity resolution approaches on real-world match problems." _PVLDB_.
18. Primpeli, A. et al. (2019). "The WDC Training Dataset and Gold Standard for Large-Scale Product Matching." _WWW 2019_.

### Iterative ER

19. Whang, S.E. et al. (2013). "Pay-As-You-Go Entity Resolution." _VLDB Journal_.

---

_This plan was prepared for Cursor Agent grind mode execution. Expected build time: 8-12 hours. The plan covers ~25 new source files, ~20 test files, and significant updates to configuration and documentation. All code should follow the conventions documented in CLAUDE.md and the PySpark Style Guide at assets/PYSPARK.md._
