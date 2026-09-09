# Experiment log: GPT OSS 120b raw baseline

- **Run label:** raw baseline (unoptimized prompts)
- **Run id:** `gpt-oss-120b-raw-baseline-2026-09-09`
- **Started (UTC):** 2026-09-09T09:35:23Z
- **Finished (UTC):** 2026-09-09T10:18:03Z
- **Student / matching LM:** `openai/gpt-oss-120b-maas` (Vertex AI MaaS)
- **Teacher / GEPA reflection LM:** `gemini/gemini-3.5-flash-lite` (not used for this matching run)
- **Prompts:** unoptimized default `BlockMatch` signature
- **Blocking:** multilingual-e5-base name embeddings + FAISS, `target_block_size=30`, `max_block_size=90`
- **Matching:** 1 ER iteration, concurrency 20, `max_tokens=8192`
- **MLflow:** experiment `SERF-Entity-Resolution` at `http://127.0.0.1:5001`

## Raw baseline scores

| Dataset | Left | Right | GT | Blocks | Precision | Recall | F1 | TP | FP | Predicted | Elapsed | Limits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dblp-acm | 2616 | 2294 | 2224 | 88 | 0.9353 | 0.4748 | **0.6299** | 1056 | 73 | 1129 | 767.3s | full tables |
| dblp-scholar | 2616 | 5218 | 5347 | 129 | 0.7754 | 0.2796 | **0.4110** | 1495 | 433 | 1928 | 1163.7s | serf benchmark-all default --max-right-entities 5000; sampled right table to 5218 (all GT-matched rights kept, then unmatched filled to the cap) |
| abt-buy | 1081 | 1092 | 1097 | 49 | 0.9804 | 0.4567 | **0.6231** | 501 | 10 | 511 | 500.7s | full tables |

## Skipped datasets

- `walmart-amazon`: in `config.yml` only; not in `serf benchmark` / `DATASET_REGISTRY`.
- `amazon-google`: in `config.yml` only; not in `serf benchmark` / `DATASET_REGISTRY`.

## Notes

- DBLP-Scholar used the existing `benchmark-all` default `--max-right-entities 5000` because the Scholar table is much larger than ACM/Abt.
- Many blocks hit `max_tokens=8192` truncation on GPT OSS 120b (reasoning + structured XML). That is recorded as a quality caveat on this raw baseline, not a protocol change.
- Full CLI logs: `/opt/cursor/artifacts/raw_baseline_run.log`
- Per-dataset JSON: `data/benchmarks/raw_baseline/<dataset>_results.json`

## Follow-on: first GEPA run

- Linked to this raw baseline (`dblp-acm` F1 **0.6299**).
- Command: `uv run serf optimize --dataset dblp-acm --signature block-match --output data/gepa_logs/dblp-acm-first`
- Student: `openai/gpt-oss-120b-maas`; teacher: `gemini/gemini-3.5-flash-lite`; `optimize.auto=light`.
- Requested splits: 1000 train blocks / 500 val records / 1000 holdout.
- Actual splits after blocking all 4910 DBLP-ACM records (82 blocks, min_block_size 2): **81 train blocks (4909 records), 1 val record, 0 holdout**. Train blocks covered almost the whole table, so val/holdout were capped to leftovers.
- GEPA budget: `auto=light`, about **384 metric calls** (~4.68 train+val evals).
- Early metrics: Iteration 0 base program full valset score **0.0** over 1/1 examples (val set is a single leftover record).
- Status: **in progress** (started 2026-09-09T10:20:06Z). Student `openai/gpt-oss-120b-maas` rollouts and teacher `gemini-3.5-flash-lite` reflection calls are live. Left running in tmux session `gepa-dblp-acm`. Logs: `/opt/cursor/artifacts/gepa_dblp_acm_first.log` and `data/gepa_logs/`.
