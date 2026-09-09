# Scratchpad

## Background and Motivation

The first GEPA optimization run on `dblp-acm` (student `openai/gpt-oss-120b-maas`, teacher
`gemini/gemini-3.5-flash-lite`) completed with exit 0 but produced nothing usable: the saved program
`data/gepa_logs/dblp-acm-first/dblp-acm_gepa.json` still holds the original `BlockMatch`
instructions. The goal of this task is to find and fix why the run could not learn anything, then
start a clean run that actually has optimization signal.

The DBLP-ACM reference point remains the raw (unoptimized) baseline: Precision 0.9353 /
Recall 0.4748 / F1 0.6299.

## Key Challenges and Analysis

1. **The valset could not score anything.** `sample_blocked_splits` took train blocks first and left
   val/holdout with the leftover loose records; `chunk_records` then packed those random records
   into synthetic blocks. Random chunks contain almost no true duplicate pairs, and pair-level F1 of
   an empty gold set is 0.0, so all 9 candidate programs tied at 0.0 and GEPA kept candidate 0.
   On DBLP-ACM the leftovers were also nearly empty: 81 train blocks / 1 val record / 0 holdout.
2. **The Vertex access token was static.** `_create_vertex_maas_lm` minted one service-account token
   and handed it to `dspy.LM` as `api_key`. Tokens live about an hour; 215 rollouts in the last
   quarter of the run failed with 401 `ACCESS_TOKEN_EXPIRED`, adding spurious zeros.
3. **Output truncation.** 257 warnings about `max_tokens=8192` truncation, plus the
   `Failed to unpack prediction and trace` warnings that follow from it.
4. **Constraints to preserve.** The double `openai/openai/` publisher prefix is required by Vertex's
   OpenAI-compatible endpoint; a raw `ya29.` token must still be accepted as-is; splits must stay
   disjoint by entity id; `train_blocks` counts blocks while `val_records`/`holdout_records` count
   records.

## High-level Task Breakdown

1. Tests first: starvation and normal cases for the split sampler, mocked token-refresh tests.
2. Rewrite `sample_blocked_splits` to partition whole blocks, val and holdout budgets first.
3. Add `VertexRefreshingLM` and wire it into `create_lm`.
4. Update `prepare_dataset_splits`, the `serf optimize` CLI, and the README.
5. Verify the real Vertex `max_tokens` ceiling with a live probe, then raise `models.max_tokens`.
6. `pytest` (excluding live suites), `ruff format`, `ruff check --fix`, `zuban check`.
7. Live smoke: student call, forced-expiry refresh, teacher call, DBLP-ACM split sizes.
8. Commit each logical change, push, document the failed run, update PR 21.
9. Start a fresh GEPA run and confirm non-zero val scores early.

## Project Status Board

- [x] Tests for block-partitioned splits and token refresh
- [x] `sample_blocked_splits` partitions blocks with val/holdout reserved first
- [x] `VertexRefreshingLM` refreshes service-account tokens with a safety margin
- [x] `prepare_dataset_splits`, CLI, and README updated
- [x] `models.max_tokens` raised to 32768 after a live endpoint probe
- [x] 226 tests pass; ruff and zuban clean
- [x] Live smoke: refresh path exercised, DBLP-ACM splits are 59 / 7 / 15 blocks
- [x] Commits pushed to `cursor/gpt-oss-student-gepa-66f9`
- [x] Failed first run recorded in `experiments/gepa-dblp-acm-first-run.md`
- [x] Per-run GEPA `log_dir` so runs stop resuming each other's state
- [x] Fresh run started in tmux session `gepa-dblp-acm-v2`; iteration 0 valset score 0.4109 over 7/7
- [ ] GEPA v2 run finishes and produces an optimized program to compare against F1 0.6299

## Executor's Feedback or Assistance Requests

- The v2 run is deliberately left running; do not kill tmux session `gepa-dblp-acm-v2`.
  Log: `/opt/cursor/artifacts/gepa_dblp_acm_v2.log`, output: `data/gepa_logs/dblp-acm-v2`.
- Two of the seven DBLP-ACM val blocks (the 100-record ones) still fail with
  `Adapter JSONAdapter failed to parse the LM response` and score 0, which caps the achievable
  valset score. Worth investigating separately: either cap `er.blocking.max_block_size` for
  optimization or make the adapter fallback more forgiving.
- A val block that happens to contain no gold pairs still scores 0.0 under `er_metric`, because
  `f1_score` returns 0.0 for an empty gold set. One of the seven DBLP-ACM val blocks is in that
  position. It is a constant drag on the average rather than a blocker, but if future runs look
  flat this is the first thing to revisit.

## Lessons

- **Reserve evaluation budgets before training budgets.** Sampling train first and giving validation
  the leftovers silently starves validation whenever a dataset produces few blocks. On DBLP-ACM,
  4910 records became 82 blocks, so train took 81 of them and val got a single record. Partition
  whole blocks and fill the val and holdout budgets first.
- **A validation set must be able to score.** Packing random leftover records into synthetic blocks
  creates examples with no true duplicate pairs, so pair-level F1 is 0.0 for every candidate and the
  optimizer has nothing to select on. Validation examples have to be the same kind of object as
  training examples: real semantic blocks with real gold pairs. A whole GEPA run can exit 0 and look
  healthy while learning nothing - always check that the initial val score is non-zero.
- **Never hand a long-running optimizer a static OAuth token.** Google service-account tokens expire
  after about an hour. Wrap the LM so it refreshes from the credentials object before each request
  (`credentials.expired` plus a safety margin) instead of minting one token at construction time.
  DSPy 3.3.1 makes this easy: `LM.forward`/`aforward` merge `self.kwargs` per call, and `api_key` is
  excluded from the cache key, so mutating `self.kwargs["api_key"]` is safe.
- **Optimizer state directories must be per run.** GEPA silently resumes from whatever state lives
  in `log_dir`. Because `optimize.log_dir` was the shared `data/gepa_logs`, the first re-run started
  at iteration 93 with the previous run's candidates and its stale 1-example valset coverage. Point
  `log_dir` at the run's own output directory and check the first log lines say `Iteration 0`.
- **Long background runs need their own process group.** The first v2 attempt died at exit 143
  (SIGTERM) three minutes in, with no OOM pressure recorded. Launching the command with
  `setsid nohup ... &` inside the tmux pane kept it alive.
- **Measure endpoint limits, do not guess them.** The Vertex `gpt-oss-120b-maas` endpoint accepted
  `max_tokens` up to 65536 and rejected 131072 only because input and output share a 131072 token
  context, so 8192 was needlessly truncating the student's reasoning plus XML output.
