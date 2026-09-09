# Experiment log: first GEPA run on DBLP-ACM (no-op result)

- **Run label:** GEPA optimization attempt 1, `dblp-acm`
- **Run id:** `gepa-dblp-acm-first-2026-09-09`
- **Started (UTC):** 2026-09-09T10:20:06Z
- **Finished (UTC):** 2026-09-09T11:25:52Z (exit 0)
- **Command:** `uv run serf optimize --dataset dblp-acm --signature block-match --output data/gepa_logs/dblp-acm-first`
- **Student / task LM:** `openai/gpt-oss-120b-maas` (Vertex AI MaaS)
- **Teacher / reflection LM:** `gemini/gemini-3.5-flash-lite`
- **Budget:** `optimize.auto=light` (384 rollouts; 381 executed)
- **Logs:** `/opt/cursor/artifacts/gepa_dblp_acm_first.log`, `data/gepa_logs/dblp-acm-first/`

## Outcome: no usable optimized prompt

The run exited 0 and wrote `data/gepa_logs/dblp-acm-first/dblp-acm_gepa.json`, but the saved
program carries the **original, unoptimized `BlockMatch` instructions**. GEPA proposed 9 candidate
programs (0-8) and every one of them scored exactly **0.0** on the valset, so the selection was
arbitrary and nothing was learned.

**There is no valid optimized-vs-baseline comparison from this run.** The DBLP-ACM raw baseline
stays the only measured number: Precision 0.9353 / Recall 0.4748 / F1 0.6299
(`experiments/gpt-oss-120b-raw-baseline.md`).

## Root cause 1: starved and structurally unscoreable val/holdout splits

`sample_blocked_splits` sampled train blocks first (up to `train_blocks=1000`), then drew val and
holdout from whatever records were left over. Blocking all 4910 DBLP-ACM records with
`er.blocking.target_block_size=30` produced only **82 blocks** averaging ~60 records, which covered
nearly the whole table.

| Split | Requested | Actual |
| --- | --- | --- |
| train | 1000 blocks | 81 blocks (4909 records) |
| val | 500 records | 1 record |
| holdout | 1000 records | 0 records |

`prepare_dataset_splits` then packed those leftover *random* records into synthetic blocks via
`chunk_records`. Random record chunks contain essentially no true duplicate pairs, so the pair-level
F1 metric is 0.0 no matter how good the prompt is. Every `Individual valset scores for new program`
line in the log reads `{0: 0.0}` over a single val example.

## Root cause 2: Vertex access token expired mid-run

`_create_vertex_maas_lm` minted one access token and passed it to `dspy.LM` as a static `api_key`.
Service-account tokens last about an hour. The run started at 10:20 UTC; from **11:21:10 UTC**
onward every request failed with HTTP 401 `ACCESS_TOKEN_EXPIRED` / `UNAUTHENTICATED` -
**215 failed rollouts**, roughly the last quarter of the budget, each scored as a spurious zero.

## Contributing factor: output truncation

`models.max_tokens` was 8192. The log contains **257** `LM response was truncated due to exceeding
max_tokens=8192` warnings, together with `Failed to unpack prediction and trace` and `No valid
predictions found for any module` warnings that follow from truncated XML output.

## Fixes landed on `cursor/gpt-oss-student-gepa-66f9`

- **Splits partition blocks.** `sample_blocked_splits` now shuffles eligible blocks once and walks
  them, filling the val record budget first, then holdout, then train (capped at `train_blocks`).
  Val and holdout are real semantic blocks, so they contain duplicate pairs and are scoreable.
  `BenchmarkSplits` exposes `train_blocks` / `val_blocks` / `holdout_blocks` plus record counts, and
  `chunk_records` is gone.
- **Token refresh.** `VertexRefreshingLM` (a `dspy.LM` subclass) holds the service-account
  credentials and refreshes the token before a request whenever it is expired or within
  `models.vertex_ai.token_refresh_margin_seconds` (300s) of expiring. A raw `ya29.` token still
  works as-is and now logs a warning that it cannot be refreshed.
- **`models.max_tokens` 8192 -> 32768.** A live probe of the Vertex `gpt-oss-120b-maas` endpoint
  accepted 8192 / 16384 / 32768 / 65536 and rejected 131072, which fails only because input and
  output share the model's 131072 token context.

## Splits after the fix (DBLP-ACM, seed 42)

| Split | Blocks | Records | Gold pairs in split |
| --- | ---: | ---: | ---: |
| train | 59 | 3357 | 1259 |
| val | 7 | 526 | 160 |
| holdout | 15 | 1026 | - |

Six of the seven val blocks contain at least one gold duplicate pair, so the valset can now
discriminate between candidate programs. The three splits are disjoint by entity id.

## Follow-on

- Re-run: `uv run serf optimize --dataset dblp-acm --signature block-match --output data/gepa_logs/dblp-acm-v2`
  (tmux session `gepa-dblp-acm-v2`, log `/opt/cursor/artifacts/gepa_dblp_acm_v2.log`).
