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

- **A lazily imported module is only safe if one thread reaches it first.** The generic matcher lost
  27 of 33 blocks on `dblp-acm` to `partially initialized module 'litellm' has no attribute
  'completion'`. DSPy resolves litellm through `dspy.utils.lazy_import.require`, which returns
  whatever `sys.modules` already holds, and MLflow's tracing hook runs a plain `import litellm` from
  inside a traced call. While that import was part way through, `sys.modules["litellm"]` held a
  module whose spec was still `_initializing`, so every other matcher thread's first touch raised.
  `EntityMatcher` made this easy to hit by building its LM and predictor lazily *inside* the thread
  pool, so ~8 threads took their first litellm touch at once and each minted its own Vertex token.
  Two fixes, both needed: import litellm at the top of `serf.dspy.lm` so it is executed once
  single-threaded, and `warm_up()` the LM and predictor on the calling thread before
  `resolve_blocks` fans out. The failure was silent - `resolve_block` catches every exception and
  degrades the block to `error_recovery` - so a run reported recall 0.1092 and looked like a result.
  Count and report degraded blocks so this can never read as a real number again.
- **One matching pass cannot beat its own blocking.** Recall of 0.52 / 0.48 / 0.14 against precision
  above 0.9 is the signature of gold pairs that were never co-blocked, not of a weak matcher. The
  benchmark now defaults to `er.max_iterations: 3`, and each round re-blocks the entities the
  previous round merged. The loop used to feed the *matcher's* resolved entities forward, which the
  per-dataset matcher returns unchanged, so `--max-iterations 3` was a no-op for the typed arm; it
  now merges connected components of predicted pairs itself, uniformly for both arms.
- **`auto_scale_by_iteration` does nothing at benchmark scale.** `FAISS_SCRIPT` caps the IVF cell
  count at `sqrt(n)`: `nlist = min(n // target_block_size, sqrt(n))`. At n = 1001 the cap binds for
  every target at or below 31, so targets of 30, 15 and 10 all produce the identical 31 blocks of
  average size 32.3. Auto-scaling only bites while `n < target_block_size^2`, i.e. below ~900
  records for a target of 30. Re-blocking on later iterations still repartitions, but because the
  entity count drops, not because of the scaling. Left as is - changing the cap would move every
  benchmark number.
- **`models.embedding` and the docs disagree.** `config.yml` sets
  `models.embedding: "intfloat/multilingual-e5-base"`, while `CLAUDE.md` lists Qwen3 embeddings as a
  key technology and `README.md` phase 1 says "Qwen3 sentence embeddings" (though its stack table
  correctly says multilingual-e5-base). Not changed - it needs a decision, and swapping the
  embedding model would invalidate every blocking number recorded so far.

- **Typed per-dataset signatures beat the generic one on all five datasets.** On identical
  1,000-record match-group samples (seed 42, one iteration, same blocking) the per-dataset
  signatures scored mean **+0.1128 F1** over `BlockMatch`: dblp-acm +0.0666, dblp-scholar +0.1222,
  abt-buy +0.0827, amazon-google +0.1654, walmart-amazon +0.1272. Precision improved on all five,
  reaching 1.0000 on dblp-scholar and abt-buy, because splitting the block into one typed input
  field per source makes a same-source pair inexpressible; the generic arm merges two Amazon
  listings with each other. They are also cheaper: 2.5x fewer dollars on dblp-scholar and 6.4x on
  walmart-amazon, since the typed output is a list of matched pairs rather than an echo of every
  entity in the block. Full write-up in `experiments/per-dataset-signature-baseline.md`.
- **gpt-oss-120b sometimes wraps its answer in a harmony envelope.** `{"final": "{...}"}` defeats
  XMLAdapter, then defeats the JSONAdapter retry, and the block is lost even though the payload
  inside is a complete answer - one dropped dblp-acm block held 28 well-formed matches. It cost the
  generic arm 12 blocks across the five datasets and the typed arm 1; long outputs trip it more
  often, which is part of why the echo-the-whole-block contract loses. Unwrapping the envelope in a
  custom adapter is the cheapest recall win available.
- **Quote the blocking ceiling before blaming the matcher.** Co-blocked gold pairs cap recall at
  0.9538 / 0.9094 / 0.8563 / 0.5594 / 0.8667 for dblp-acm / dblp-scholar / abt-buy / amazon-google /
  walmart-amazon at target 30. amazon-google's 0.5217 recall is 93% of everything blocking made
  reachable, so its low score is a blocking problem; on dblp-acm the typed arm hit 454 of 454
  reachable pairs, so there is nothing left there for a prompt to win.
- **Pin the code when an experiment spans arms.** Concurrent work on this branch changed
  `serf benchmark` between arms twice (a three-iteration default, then transitive pair expansion),
  which silently redefines the metric mid-A/B. Extracting the commit with
  `git archive <sha> | tar -x -C /tmp/serf-ab` and running that copy made the ten runs comparable.
- **Never name a shell variable `TMUX`.** tmux reads `TMUX` from the environment as its socket path,
  so `TMUX="tmux -f ..."` in a driver script launched from inside a pane makes every child tmux call
  fail with `error creating tmux -f ... (No such file or directory)`. Sessions then vanish in
  seconds and the runs look like instant failures with empty logs.
