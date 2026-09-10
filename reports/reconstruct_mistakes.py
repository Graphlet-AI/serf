"""Reconstruct benchmark false positives and false negatives from MLflow traces.

The ``*_results.json`` files that ``serf benchmark`` writes hold only aggregate
counts, so the individual wrong pairs are not persisted anywhere. They can still
be recovered: DSPy autologging records one MLflow trace per matched block, and
each trace carries the block's input records and the model's match output.

This script replays those traces. For every trace it maps the block-local record
ids the LLM saw back to the global entity ids the benchmark scores against, using
each record's source ``id`` from the original CSV, then rebuilds the predicted
pair set exactly as ``serf.match.run.collect_pairs`` does. Diffing that set
against the ground truth yields the false positives, and the gold pairs that are
missing yield the false negatives, split by whether the two records were ever
placed in the same block. A blocking miss and a matching error look identical in
the aggregate recall number but have completely different fixes.

Reconstruction is self-checking: the rebuilt predicted-pair, true-positive and
false-positive counts are compared against the saved aggregates for every run,
and any mismatch is reported.

Usage
-----
    uv run python reports/reconstruct_mistakes.py <traces.jsonl> <output.json>

The traces JSONL is produced by ``reports/dump_traces.py``.
"""

import json
import pathlib
import sys
from typing import Any

from serf.eval.benchmarks import RIGHT_ID_OFFSET, BenchmarkDataset
from serf.eval.sample import sample_records
from serf.logs import get_logger

logger = get_logger(__name__)

DATASETS = ["dblp-acm", "dblp-scholar", "abt-buy", "amazon-google", "walmart-amazon"]

# Benchmark results live under the repo, plus any pinned source snapshot used to keep
# an A/B's code fixed while the working tree moves on.
BENCHMARK_ROOTS = ["data/benchmarks", "/tmp/serf-ab/data/benchmarks"]
ARCHIVE_ROOT = "data/benchmarks/full_baseline"

# How many worked examples to keep per error category. Cross-source false positives
# and false negatives the model actually judged are the most informative, so they get
# the larger budgets.
EXAMPLE_BUDGET: dict[str, int] = {
    "false_positives_cross_source": 3,
    "false_positives_same_source": 2,
    "false_negatives_model_error": 3,
    "false_negatives_failed_block": 2,
    "false_negatives_blocking_miss": 3,
}

# Counts the reconstruction reports, kept in one place so runs that cannot be
# reconstructed still emit the same shape.
RECONSTRUCTED_KEYS = (
    "predicted",
    "true_positives",
    "false_positives",
    "fp_cross_source",
    "fp_same_source",
    "gold",
    "false_negatives",
    "fn_co_blocked",
    "fn_not_co_blocked",
    "fn_failed_block",
    "fn_model_error",
    "gold_co_blocked",
)

# Typed per-dataset signatures name their two record fields after the sources,
# so the pair of field names in a trace's inputs identifies the dataset.
TYPED_FIELDS: dict[frozenset[str], tuple[str, str, str]] = {
    frozenset({"dblp_records", "acm_records"}): ("dblp-acm", "dblp_records", "acm_records"),
    frozenset({"dblp_records", "scholar_records"}): (
        "dblp-scholar",
        "dblp_records",
        "scholar_records",
    ),
    frozenset({"abt_records", "buy_records"}): ("abt-buy", "abt_records", "buy_records"),
    frozenset({"amazon_records", "google_records"}): (
        "amazon-google",
        "amazon_records",
        "google_records",
    ),
    frozenset({"walmart_records", "amazon_records"}): (
        "walmart-amazon",
        "walmart_records",
        "amazon_records",
    ),
}


def load_datasets() -> dict[str, BenchmarkDataset]:
    """Load every benchmark dataset from its cached archive.

    Returns
    -------
    dict[str, BenchmarkDataset]
        Dataset name to loaded dataset
    """
    return {name: BenchmarkDataset.download(name, ARCHIVE_ROOT) for name in DATASETS}


def build_id_maps(
    datasets: dict[str, BenchmarkDataset],
) -> tuple[dict[str, tuple[dict[str, int], dict[str, int]]], dict[str, dict[int, dict[str, str]]]]:
    """Map source CSV ids to global entity ids, and global ids back to field values.

    Parameters
    ----------
    datasets : dict[str, BenchmarkDataset]
        Loaded datasets

    Returns
    -------
    tuple[dict, dict]
        Per dataset (left, right) source-id maps, and per dataset global-id to record
    """
    id_maps: dict[str, tuple[dict[str, int], dict[str, int]]] = {}
    records: dict[str, dict[int, dict[str, str]]] = {}
    for name, dataset in datasets.items():
        left_map: dict[str, int] = {}
        right_map: dict[str, int] = {}
        row_values: dict[int, dict[str, str]] = {}
        for i, (_idx, row) in enumerate(dataset.table_a.iterrows()):
            left_map[str(row["id"])] = i
            row_values[i] = {str(k): ("" if v != v else str(v)) for k, v in row.items()}
        for i, (_idx, row) in enumerate(dataset.table_b.iterrows()):
            gid = i + RIGHT_ID_OFFSET
            right_map[str(row["id"])] = gid
            row_values[gid] = {str(k): ("" if v != v else str(v)) for k, v in row.items()}
        id_maps[name] = (left_map, right_map)
        records[name] = row_values
    return id_maps, records


def parse_generic_block(inputs: dict[str, Any]) -> dict[int, tuple[str, str]]:
    """Read block-local id to (side, source id) from a generic BlockMatch trace.

    Parameters
    ----------
    inputs : dict[str, Any]
        Root span inputs

    Returns
    -------
    dict[int, tuple[str, str]]
        Block-local record id to side marker and source CSV id
    """
    local: dict[int, tuple[str, str]] = {}
    for record in json.loads(inputs["block_records"]):
        attrs: dict[str, Any] = record.get("attributes") or {}
        if "l_id" in attrs:
            local[record["id"]] = ("l", str(attrs["l_id"]))
        elif "r_id" in attrs:
            local[record["id"]] = ("r", str(attrs["r_id"]))
    return local


def parse_typed_block(
    inputs: dict[str, Any], left_field: str, right_field: str
) -> dict[int, tuple[str, str]]:
    """Read block-local id to (side, source id) from a typed per-dataset trace.

    Parameters
    ----------
    inputs : dict[str, Any]
        Root span inputs
    left_field : str
        Signature field holding the left source's records
    right_field : str
        Signature field holding the right source's records

    Returns
    -------
    dict[int, tuple[str, str]]
        Block-local record id to side marker and source CSV id
    """
    local: dict[int, tuple[str, str]] = {}
    for field, side in ((left_field, "l"), (right_field, "r")):
        for record in inputs.get(field) or []:
            local[record["record_id"]] = (side, str(record.get("source_id")))
    return local


def to_global(
    local: dict[int, tuple[str, str]], id_map: tuple[dict[str, int], dict[str, int]]
) -> dict[int, int]:
    """Translate block-local record ids to global benchmark entity ids.

    Parameters
    ----------
    local : dict[int, tuple[str, str]]
        Block-local id to side and source CSV id
    id_map : tuple[dict[str, int], dict[str, int]]
        Left and right source-id to global-id maps

    Returns
    -------
    dict[int, int]
        Block-local id to global entity id
    """
    left_map, right_map = id_map
    mapped: dict[int, int] = {}
    for local_id, (side, source_id) in local.items():
        table = left_map if side == "l" else right_map
        if source_id in table:
            mapped[local_id] = table[source_id]
    return mapped


def classify_dataset(
    local: dict[int, tuple[str, str]],
    id_maps: dict[str, tuple[dict[str, int], dict[str, int]]],
) -> tuple[str, float]:
    """Identify which dataset a generic trace's block came from.

    Parameters
    ----------
    local : dict[int, tuple[str, str]]
        Block-local id to side and source CSV id
    id_maps : dict[str, tuple[dict[str, int], dict[str, int]]]
        Per dataset source-id maps

    Returns
    -------
    tuple[str, float]
        Best matching dataset name and the fraction of records it resolved
    """
    best_name = ""
    best_score = -1.0
    for name, id_map in id_maps.items():
        resolved = len(to_global(local, id_map))
        score = resolved / max(1, len(local))
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def pairs_from_generic(outputs: Any, mapped: dict[int, int]) -> set[tuple[int, int]]:
    """Rebuild predicted pairs from a generic BlockMatch resolution.

    Mirrors ``serf.match.run.collect_pairs``: pairs come from explicit match
    decisions and from merged entities' ``source_ids``.

    Parameters
    ----------
    outputs : Any
        Root span outputs
    mapped : dict[int, int]
        Block-local id to global entity id

    Returns
    -------
    set[tuple[int, int]]
        Predicted pairs as ordered global id tuples
    """
    pairs: set[tuple[int, int]] = set()
    if not isinstance(outputs, dict):
        return pairs
    resolution = outputs.get("resolution")
    if not isinstance(resolution, dict):
        return pairs
    for match in resolution.get("matches") or []:
        if not match.get("is_match"):
            continue
        left = mapped.get(match.get("entity_a_id"))
        right = mapped.get(match.get("entity_b_id"))
        if left is not None and right is not None and left != right:
            pairs.add((min(left, right), max(left, right)))
    for entity in resolution.get("resolved_entities") or []:
        left = mapped.get(entity.get("id"))
        for source_id in entity.get("source_ids") or []:
            right = mapped.get(source_id)
            if left is not None and right is not None and left != right:
                pairs.add((min(left, right), max(left, right)))
    return pairs


def pairs_from_typed(
    outputs: Any, mapped: dict[int, int]
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], str]]:
    """Rebuild predicted pairs and justifications from a typed resolution.

    Parameters
    ----------
    outputs : Any
        Root span outputs
    mapped : dict[int, int]
        Block-local id to global entity id

    Returns
    -------
    tuple[set[tuple[int, int]], dict[tuple[int, int], str]]
        Predicted pairs and the model's justification per pair
    """
    pairs: set[tuple[int, int]] = set()
    reasons: dict[tuple[int, int], str] = {}
    if not isinstance(outputs, dict):
        return pairs, reasons
    for candidates in outputs.values():
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict) or not candidate.get("is_match"):
                continue
            left_id = (candidate.get("left") or {}).get("record_id")
            right_id = (candidate.get("right") or {}).get("record_id")
            if left_id is None or right_id is None:
                continue
            left = mapped.get(int(left_id))
            right = mapped.get(int(right_id))
            if left is None or right is None or left == right:
                continue
            key = (min(left, right), max(left, right))
            pairs.add(key)
            reasons[key] = str(candidate.get("justification") or "")
    return pairs, reasons


def generic_reasons(outputs: Any, mapped: dict[int, int]) -> dict[tuple[int, int], str]:
    """Collect per-pair reasoning from a generic resolution's match decisions.

    Parameters
    ----------
    outputs : Any
        Root span outputs
    mapped : dict[int, int]
        Block-local id to global entity id

    Returns
    -------
    dict[tuple[int, int], str]
        Reasoning text per predicted pair, where the model supplied one
    """
    reasons: dict[tuple[int, int], str] = {}
    if not isinstance(outputs, dict):
        return reasons
    resolution = outputs.get("resolution")
    if not isinstance(resolution, dict):
        return reasons
    for match in resolution.get("matches") or []:
        left = mapped.get(match.get("entity_a_id"))
        right = mapped.get(match.get("entity_b_id"))
        if left is None or right is None or left == right:
            continue
        text = match.get("reasoning") or match.get("justification") or ""
        if text:
            reasons[(min(left, right), max(left, right))] = str(text)
    return reasons


def load_traces(
    path: pathlib.Path,
    id_maps: dict[str, tuple[dict[str, int], dict[str, int]]],
) -> list[dict[str, Any]]:
    """Parse the dumped traces into per-block reconstruction records.

    Parameters
    ----------
    path : pathlib.Path
        JSONL dump produced by ``reports/dump_traces.py``
    id_maps : dict[str, tuple[dict[str, int], dict[str, int]]]
        Per dataset source-id maps

    Returns
    -------
    list[dict[str, Any]]
        One entry per usable trace
    """
    parsed: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        inputs = raw.get("inputs")
        if not isinstance(inputs, dict):
            continue
        typed = TYPED_FIELDS.get(frozenset(k for k in inputs if k.endswith("_records")))
        if typed is not None:
            dataset, left_field, right_field = typed
            local = parse_typed_block(inputs, left_field, right_field)
            kind = "typed"
            confidence = 1.0
        elif "block_records" in inputs:
            local = parse_generic_block(inputs)
            dataset, confidence = classify_dataset(local, id_maps)
            kind = "generic"
        else:
            continue
        if not local:
            continue
        mapped = to_global(local, id_maps[dataset])
        if kind == "typed":
            pairs, reasons = pairs_from_typed(raw.get("outputs"), mapped)
        else:
            pairs = pairs_from_generic(raw.get("outputs"), mapped)
            reasons = generic_reasons(raw.get("outputs"), mapped)
        usage = json.loads(raw["token_usage"]) if raw.get("token_usage") else None
        parsed.append(
            {
                "trace_id": raw["trace_id"],
                "time": raw["request_time"] / 1000.0,
                "state": raw["state"],
                "kind": kind,
                "dataset": dataset,
                "confidence": confidence,
                "block_size": len(local),
                "block_ids": sorted(mapped.values()),
                "pairs": pairs,
                "reasons": reasons,
                "tokens": usage,
            }
        )
    return parsed


def discover_runs() -> list[dict[str, Any]]:
    """Find every saved benchmark run and derive its wall-clock window.

    Returns
    -------
    list[dict[str, Any]]
        Run descriptors sorted by start time
    """
    runs: list[dict[str, Any]] = []
    paths: list[pathlib.Path] = []
    for root in BENCHMARK_ROOTS:
        root_path = pathlib.Path(root)
        if root_path.is_dir():
            paths.extend(sorted(root_path.glob("*/*_results.json")))
    for path in paths:
        saved = json.loads(path.read_text(encoding="utf-8"))
        end = path.stat().st_mtime
        runs.append(
            {
                "path": str(path),
                "group": path.parent.name,
                "dataset": saved["dataset"],
                "signature_mode": saved.get("signature_mode", "generic"),
                "sample_records": saved.get("sample_records"),
                "seed": saved.get("seed") or 42,
                "max_iterations": int(saved.get("max_iterations") or 1),
                "iterations_run": int(saved.get("iterations_run") or 1),
                "start": end - float(saved["elapsed_seconds"]),
                "end": end,
                "saved": saved,
            }
        )
    # Result files get copied between output directories, and a copy carries a fresh mtime
    # that would invent a second wall-clock window for a run that only happened once. Keep
    # the earliest copy of any identical result so traces are attributed to the real run.
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for run in sorted(runs, key=lambda run: run["start"]):
        saved = run["saved"]
        key = (
            saved["dataset"],
            run["signature_mode"],
            run["sample_records"],
            run["seed"],
            run["max_iterations"],
            saved["elapsed_seconds"],
            saved["predicted_pairs"],
            saved["true_positives"],
        )
        unique.setdefault(key, run)
    return sorted(unique.values(), key=lambda run: run["start"])


def sampled_ids(dataset: BenchmarkDataset, count: int, seed: int) -> set[int]:
    """Recompute the exact record sample a run drew, so traces can be attributed to it.

    ``serf.eval.sample.sample_records`` is seeded, so replaying it reproduces the run's
    sample without re-running anything.

    Parameters
    ----------
    dataset : BenchmarkDataset
        Loaded dataset
    count : int
        Record budget the run asked for
    seed : int
        RNG seed the run used

    Returns
    -------
    set[int]
        Global entity ids in the sample
    """
    left, right = dataset.to_entities()
    sample = sample_records(left + right, dataset.ground_truth, count, seed=seed)
    return {entity.id for entity in sample.records}


def assign_traces(
    runs: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    datasets: dict[str, BenchmarkDataset],
) -> dict[str, list[dict[str, Any]]]:
    """Attribute every trace to exactly one run.

    Runs overlap in time whenever a full-table run and a sampled A/B arm are in flight
    together, so a time window alone is not enough. Three criteria pin each trace down:
    the dataset it references, whether its input shape matches the run's signature mode,
    and, for sampled runs, whether every record in the block belongs to that run's
    replayed sample. When several runs still qualify the narrowest one wins.

    Parameters
    ----------
    runs : list[dict[str, Any]]
        Run descriptors from ``discover_runs``
    traces : list[dict[str, Any]]
        Parsed traces
    datasets : dict[str, BenchmarkDataset]
        Loaded datasets

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Traces per run path
    """
    scopes: dict[str, set[int] | None] = {}
    sizes: dict[str, int] = {}
    cache: dict[tuple[str, int, int], set[int]] = {}
    for run in runs:
        dataset = datasets[run["dataset"]]
        if run["sample_records"]:
            key = (run["dataset"], int(run["sample_records"]), int(run["seed"]))
            if key not in cache:
                cache[key] = sampled_ids(dataset, key[1], key[2])
            scopes[run["path"]] = cache[key]
            sizes[run["path"]] = len(cache[key])
        else:
            scopes[run["path"]] = None
            sizes[run["path"]] = len(dataset.table_a) + len(dataset.table_b)

    assigned: dict[str, list[dict[str, Any]]] = {run["path"]: [] for run in runs}
    for run in runs:
        run["scope"] = scopes[run["path"]]
    for trace in traces:
        candidates = []
        for run in runs:
            if run["dataset"] != trace["dataset"]:
                continue
            if not run["start"] - 5 <= trace["time"] <= run["end"] + 5:
                continue
            wanted = "typed" if run["signature_mode"] != "generic" else "generic"
            if trace["kind"] != wanted:
                continue
            scope = scopes[run["path"]]
            if scope is not None and not set(trace["block_ids"]) <= scope:
                continue
            candidates.append(run)
        if not candidates:
            continue
        best = min(candidates, key=lambda run: (sizes[run["path"]], run["start"]))
        assigned[best["path"]].append(trace)
    return assigned


def side_of(entity_id: int) -> str:
    """Return which source table a global entity id belongs to.

    Parameters
    ----------
    entity_id : int
        Global entity id

    Returns
    -------
    str
        ``left`` or ``right``
    """
    return "right" if entity_id >= RIGHT_ID_OFFSET else "left"


def render_pair(
    pair: tuple[int, int],
    records: dict[int, dict[str, str]],
    reason: str,
    co_blocked: bool | None,
    label: str,
) -> dict[str, Any]:
    """Build a serialisable example from a pair and its two source records.

    Parameters
    ----------
    pair : tuple[int, int]
        Global entity id pair
    records : dict[int, dict[str, str]]
        Global id to source field values
    reason : str
        Model justification, when one was recorded
    co_blocked : bool | None
        Whether both records shared a block, or None when not applicable
    label : str
        ``false_positive`` or ``false_negative``

    Returns
    -------
    dict[str, Any]
        Example payload for the report
    """
    left_id, right_id = pair
    return {
        "label": label,
        "left_id": left_id,
        "right_id": right_id,
        "left_side": side_of(left_id),
        "right_side": side_of(right_id),
        "left": records.get(left_id, {}),
        "right": records.get(right_id, {}),
        "reason": reason,
        "co_blocked": co_blocked,
    }


def analyse_run(
    run: dict[str, Any],
    traces: list[dict[str, Any]],
    datasets: dict[str, BenchmarkDataset],
    records: dict[str, dict[int, dict[str, str]]],
    examples_per_kind: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Reconstruct one run's pair-level outcome from its traces.

    Parameters
    ----------
    run : dict[str, Any]
        Run descriptor from ``discover_runs``
    traces : list[dict[str, Any]]
        Traces attributed to this run by ``assign_traces``
    datasets : dict[str, BenchmarkDataset]
        Loaded datasets
    records : dict[str, dict[int, dict[str, str]]]
        Per dataset global-id to source field values
    examples_per_kind : dict[str, int] | None
        How many examples to keep per category, defaulting to ``EXAMPLE_BUDGET``

    Returns
    -------
    dict[str, Any]
        Reconstruction summary, verification status and example mistakes
    """
    dataset = run["dataset"]
    selected = traces
    # A run with more than one ER iteration re-blocks the entities the previous round
    # merged, so its later traces describe merged entities rather than source records and
    # its pair set is expanded through those merges. Neither is recoverable from a trace
    # in isolation, so such runs contribute scores and token usage but no decomposition.
    reliable = run["iterations_run"] <= 1

    predicted: set[tuple[int, int]] = set()
    reasons: dict[tuple[int, int], str] = {}
    blocks: list[tuple[list[int], bool]] = []
    tokens_in = 0
    tokens_out = 0
    for trace in selected:
        predicted |= trace["pairs"]
        reasons.update(trace["reasons"])
        blocks.append((trace["block_ids"], trace["state"] == "ERROR"))
        if trace["tokens"]:
            tokens_in += int(trace["tokens"].get("input_tokens") or 0)
            tokens_out += int(trace["tokens"].get("output_tokens") or 0)

    if not reliable:
        return {
            "path": run["path"],
            "group": run["group"],
            "dataset": dataset,
            "signature_mode": run["signature_mode"],
            "sample_records": run["sample_records"],
            "max_iterations": run["max_iterations"],
            "iterations_run": run["iterations_run"],
            "start": run["start"],
            "end": run["end"],
            "saved": run["saved"],
            "traces": len(selected),
            "error_traces": sum(1 for t in selected if t["state"] == "ERROR"),
            "block_sizes": sorted(len(b) for b, _e in blocks),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "reconstructed": {key: 0 for key in RECONSTRUCTED_KEYS},
            "reconstruction_reliable": False,
            "verified": False,
            "examples": {key: [] for key in EXAMPLE_BUDGET},
        }

    gold = set(datasets[dataset].ground_truth)
    scope = run.get("scope")
    if scope is not None:
        # Restrict gold with the run's replayed sample rather than with the ids seen in
        # traces, so pairs inside blocks that were skipped or lost still count as missed.
        gold = {pair for pair in gold if pair[0] in scope and pair[1] in scope}

    co_blocked: set[tuple[int, int]] = set()
    co_blocked_ok: set[tuple[int, int]] = set()
    co_blocked_failed: set[tuple[int, int]] = set()
    for block, errored in blocks:
        member = set(block)
        for pair in gold:
            if pair[0] in member and pair[1] in member:
                co_blocked.add(pair)
                if errored:
                    co_blocked_failed.add(pair)
                else:
                    co_blocked_ok.add(pair)

    true_positives = predicted & gold
    false_positives = predicted - gold
    false_negatives = gold - predicted
    fn_co_blocked = false_negatives & co_blocked
    fn_not_blocked = false_negatives - co_blocked
    # A gold pair co-blocked only inside blocks whose LLM call failed was never
    # judged at all, so it is a pipeline failure rather than a matching error.
    fn_failed_block = fn_co_blocked & (co_blocked_failed - co_blocked_ok)
    fn_model_error = fn_co_blocked - fn_failed_block
    fp_cross_source = {p for p in false_positives if side_of(p[0]) != side_of(p[1])}
    fp_same_source = false_positives - fp_cross_source

    saved = run["saved"]
    verified = (
        len(predicted) == saved["predicted_pairs"]
        and len(true_positives) == saved["true_positives"]
        and len(false_positives) == saved["false_positives"]
        and len(gold) == saved["true_pairs"]
    )

    row_values = records[dataset]
    budget = EXAMPLE_BUDGET if examples_per_kind is None else examples_per_kind

    def examples(
        pairs: set[tuple[int, int]], category: str, label: str, blocked: bool | None
    ) -> list[Any]:
        # Prefer pairs the model explained, so the report can quote its reasoning.
        ordered = sorted(pairs, key=lambda p: (0 if reasons.get(p) else 1, p))
        return [
            render_pair(pair, row_values, reasons.get(pair, ""), blocked, label)
            for pair in ordered[: budget[category]]
        ]

    return {
        "path": run["path"],
        "group": run["group"],
        "dataset": dataset,
        "signature_mode": run["signature_mode"],
        "sample_records": run["sample_records"],
        "max_iterations": run["max_iterations"],
        "iterations_run": run["iterations_run"],
        "start": run["start"],
        "end": run["end"],
        "saved": saved,
        "traces": len(selected),
        "error_traces": sum(1 for t in selected if t["state"] == "ERROR"),
        "block_sizes": sorted(len(b) for b, _e in blocks),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "reconstructed": {
            "predicted": len(predicted),
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "fp_cross_source": len(fp_cross_source),
            "fp_same_source": len(fp_same_source),
            "gold": len(gold),
            "false_negatives": len(false_negatives),
            "fn_co_blocked": len(fn_co_blocked),
            "fn_not_co_blocked": len(fn_not_blocked),
            "fn_failed_block": len(fn_failed_block),
            "fn_model_error": len(fn_model_error),
            "gold_co_blocked": len(co_blocked),
        },
        "reconstruction_reliable": True,
        "verified": verified,
        "examples": {
            key: examples(pairs, key, label, blocked)
            for key, pairs, label, blocked in (
                (
                    "false_positives_cross_source",
                    fp_cross_source,
                    "false_positive",
                    True,
                ),
                ("false_positives_same_source", fp_same_source, "false_positive", True),
                ("false_negatives_model_error", fn_model_error, "false_negative", True),
                ("false_negatives_failed_block", fn_failed_block, "false_negative", True),
                ("false_negatives_blocking_miss", fn_not_blocked, "false_negative", False),
            )
        },
    }


def main() -> None:
    """Reconstruct every run's mistakes and write the analysis JSON."""
    traces_path = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])

    datasets = load_datasets()
    id_maps, records = build_id_maps(datasets)
    traces = load_traces(traces_path, id_maps)
    logger.info(f"Parsed {len(traces)} traces from {traces_path}")

    runs = discover_runs()
    assigned = assign_traces(runs, traces, datasets)
    analyses = [analyse_run(run, assigned[run["path"]], datasets, records) for run in runs]
    for analysis in analyses:
        recon = analysis["reconstructed"]
        saved = analysis["saved"]
        if not analysis["reconstruction_reliable"]:
            status = "ITER    "
        elif analysis["verified"]:
            status = "OK      "
        else:
            status = "MISMATCH"
        logger.info(
            f"{status} {analysis['dataset']:15s} {analysis['signature_mode']:12s} "
            f"it={analysis['iterations_run']} "
            f"sample={analysis['sample_records']} traces={analysis['traces']} "
            f"pred={recon['predicted']}/{saved['predicted_pairs']} "
            f"tp={recon['true_positives']}/{saved['true_positives']} "
            f"fp={recon['false_positives']}/{saved['false_positives']} "
            f"fn={recon['false_negatives']} (co-blocked {recon['fn_co_blocked']}, "
            f"blocking miss {recon['fn_not_co_blocked']})"
        )

    totals = {
        "tokens_in": sum(int(t["tokens"]["input_tokens"]) for t in traces if t["tokens"]),
        "tokens_out": sum(int(t["tokens"]["output_tokens"]) for t in traces if t["tokens"]),
        "traces": len(traces),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"runs": analyses, "trace_totals": totals}, indent=2), encoding="utf-8"
    )
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
