# SERF Build Plan

## Executive Summary

Build the complete SERF (Semantic Entity Resolution Framework) as specified in `docs/SERF_LONG_SHOT_PLAN.md`:

1. Convert from Poetry to uv
2. Build all core modules (~25 source files)
3. Write comprehensive tests (~20 test files)
4. Run benchmarks on standard ER datasets for baseline scores
5. Prepare for PyPI publication

---

## Phase 1: Project Infrastructure (~30 min)

### Task 1.1: Convert Poetry → uv

**Files**: `pyproject.toml` (rewrite), `poetry.lock` (delete)

- Rewrite `pyproject.toml` to PEP 621 `[project]` format
- Dependencies: `dspy-ai>=3.1.0`, `click>=8.1`, `pyyaml>=6.0`, `pyspark>=4.0,<5.0`, `sentence-transformers>=5.1`, `faiss-cpu>=1.9`, `graphframes-py>=0.10`, `cleanco>=2.3`, `tqdm>=4.60`, `aiohttp>=3.9`
- Dev deps under `[dependency-groups]`: `pytest>=8.0`, `pytest-asyncio>=1.0`, `ruff>=0.11`, `zuban>=0.0.23`, `pre-commit>=4.0`, `types-pyyaml>=6.0`
- `[project.scripts]` serf = "serf.cli.main:cli"
- `[tool.ruff]` config (line-length=100, py312, isort)
- `[build-system]` with hatchling
- Run `uv sync`

### Task 1.2: Update .pre-commit-config.yaml

Replace black/isort/flake8 with Ruff hooks. Keep zuban, prettier.

### Task 1.3: Update config.yml

Add: `er.blocking.*`, `er.matching.*`, `er.eval.*`, `er.paths.*`, `benchmarks.datasets.*`, `models.*`

### Task 1.4: Create Module Directories

`__init__.py` for: `block/`, `match/`, `merge/`, `edge/`, `eval/`, `analyze/`, `spark/`

### Task 1.5: Update CLAUDE.md

Reflect uv/Ruff tooling.

**Acceptance**: `uv sync` succeeds, `uv run serf --help` works.

---

## Phase 2: Pipeline Types (~30 min)

### Task 2.1: Replace src/serf/dspy/types.py

**File**: `src/serf/dspy/types.py` — delete old contents, write only what the pipeline needs:

```python
class Entity(BaseModel):
    """Generic entity for ER. Domain fields live in `attributes`."""
    id: int
    uuid: Optional[str] = None
    name: str
    description: str = ""
    entity_type: str = "entity"
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_ids: Optional[list[int]] = None
    source_uuids: Optional[list[str]] = None
    match_skip: Optional[bool] = None
    match_skip_reason: Optional[str] = None
    match_skip_history: Optional[list[int]] = None

class EntityBlock(BaseModel):
    block_key: str
    block_key_type: str
    block_size: int
    entities: list[Entity]

class MatchDecision(BaseModel):
    entity_a_id: int
    entity_b_id: int
    is_match: bool
    confidence: float = Field(ge=0, le=1)
    reasoning: str

class BlockResolution(BaseModel):
    block_key: str
    matches: list[MatchDecision]
    resolved_entities: list[Entity]
    was_resolved: bool
    original_count: int
    resolved_count: int

class FieldProfile(BaseModel): ...
class DatasetProfile(BaseModel): ...
class IterationMetrics(TypedDict, total=False): ...
class BlockingMetrics(TypedDict, total=False): ...
```

### Task 2.2: Create src/serf/dspy/type_generator.py (NEW)

`entity_type_from_spark_schema()` — auto-generate Pydantic Entity subclass from Spark StructType.

### Task 2.3: Write tests

**Files**: `tests/test_types.py`, `tests/test_type_generator.py`

---

## Phase 3: DSPy Signatures (~30 min)

### Task 3.1: Create src/serf/dspy/signatures.py (NEW)

- `BlockMatch` — match entire blocks
- `EntityMerge` — merge matched entities
- `EdgeResolve` — merge duplicate edges
- `AnalyzeDataset` — profile and recommend strategy

### Task 3.2: Write tests — `tests/test_signatures.py`

---

## Phase 4: Blocking Module (~1.5 hr)

### Task 4.1: src/serf/block/embeddings.py — `EntityEmbedder` (sentence-transformers, Qwen3 default)

### Task 4.2: src/serf/block/faiss_blocker.py — `FAISSBlocker` (IndexIVFFlat, auto-scale)

### Task 4.3: src/serf/block/normalize.py — Name normalization (cleanco, acronyms, stop words)

### Task 4.4: src/serf/block/pipeline.py — `SemanticBlockingPipeline` (embed→cluster→split)

### Task 4.5: Write tests — `tests/test_embeddings.py`, `tests/test_faiss_blocker.py`, `tests/test_normalize.py`

---

## Phase 5: Matching & Merging (~2 hr)

### Task 5.1: src/serf/match/uuid_mapper.py — `UUIDMapper` (UUID↔int, caching, recovery)

### Task 5.2: src/serf/match/matcher.py — `EntityMatcher` (async, semaphore, DSPy calls, error recovery)

### Task 5.3: src/serf/match/few_shot.py — Few-shot examples for merge

### Task 5.4: src/serf/merge/merger.py — `EntityMerger` (field-level merge, master ID, source_ids)

### Task 5.5: Write tests — `tests/test_uuid_mapper.py`, `tests/test_matcher.py`

---

## Phase 6: Evaluation & Benchmarks (~1.5 hr)

### Task 6.1: src/serf/eval/metrics.py — precision, recall, F1, pair completeness, reduction ratio

### Task 6.2: src/serf/eval/benchmarks.py — `BenchmarkDataset` (download, load, evaluate DeepMatcher format)

### Task 6.3: Write tests — `tests/test_metrics.py`, `tests/test_benchmarks.py`

---

## Phase 7: Dataset Analysis (~45 min)

### Task 7.1: src/serf/analyze/profiler.py — `DatasetProfiler`

### Task 7.2: src/serf/analyze/field_detection.py — `detect_field_type()`

### Task 7.3: Write tests — `tests/test_profiler.py`, `tests/test_field_detection.py`

---

## Phase 8: Edge Resolution (~30 min)

### Task 8.1: src/serf/edge/resolver.py — `EdgeResolver`

### Task 8.2: Write test — `tests/test_edge_resolver.py`

---

## Phase 9: Spark Integration (~1 hr)

### Task 9.1: src/serf/spark/schemas.py — Pydantic→Spark schema bridge

### Task 9.2: src/serf/spark/utils.py — UDTF factory, window utilities

### Task 9.3: src/serf/spark/iceberg.py — Iceberg catalog (optional)

### Task 9.4: src/serf/spark/graph.py — Connected components (GraphFrames + manual fallback)

### Task 9.5: Write tests — `tests/test_schemas.py`, `tests/test_graph.py`

---

## Phase 10: DSPy Agents (~45 min)

### Task 10.1: src/serf/dspy/agents.py — `ERAgent` (ReAct, tools, convergence)

### Task 10.2: Write test — `tests/test_agents.py`

---

## Phase 11: CLI (~1 hr)

### Task 11.1: Rewrite src/serf/cli/main.py

Commands: `analyze`, `resolve`, `block`, `match`, `eval`, `edges`, `benchmark`, `download`

### Task 11.2: Write test — `tests/test_cli.py`

---

## Phase 12: Benchmark Evaluation (~2 hr)

### Task 12.1: Download DBLP-ACM, DBLP-Scholar, Walmart-Amazon

### Task 12.2: Run SERF pipeline on each (load→embed→block→match→evaluate)

### Task 12.3: Document baseline results in README

---

## Phase 13: PyPI Preparation (~30 min)

### Task 13.1: pyproject.toml metadata (Apache-2.0 license, classifiers, URLs)

### Task 13.2: Update README.md (install, quickstart, CLI, benchmarks)

### Task 13.3: Quality checks (ruff, zuban, pytest, uv build)

---

## Phase 14: Final Cleanup & Commit
