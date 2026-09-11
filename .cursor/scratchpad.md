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
- [x] Baseline benchmark report for all five datasets, generic vs per-dataset signatures
- [x] Blocking recall traced as the real cap on recall; three blocking bugs found and fixed
- [x] `serf blocking-sweep` added: scores embeddings on co-blocked gold pairs, no LLM calls
- [x] Nine embeddings swept over all five full datasets; `BAAI/bge-small-en-v1.5` wins
- [x] `models.embedding` / `models.embedding_prompt` set to the winner, docs reconciled
- [x] LOW / HIGH embedding tiers added, selectable with `serf benchmark --embedding-tier`
- [x] Large candidates chosen from MTEB clustering and swept; no large model is worth the CPU
- [x] JSON all-fields blocking strategy added; loses on 27 of 30 cells, kept off by default
- [x] `dspy.XMLAdapter` block loss found and fixed: 6 of 33 blocks parsed, now 29 of 33
- [x] 1K abt-buy sample run on both tiers; LOW F1 0.9038, HIGH F1 0.8963
- [x] Public ER leaderboard located and SERF placed against it in the README

## Executor's Feedback or Assistance Requests

- **The HIGH tier needs a decision.** The instruction was to default HIGH to a bge model after
  trying intfloat. `intfloat/multilingual-e5-large-instruct` was measured and lost (0.8705 mean
  blocking recall against LOW's 0.8765), so HIGH is `BAAI/bge-large-en-v1.5` at 0.8665. But
  `mixedbread-ai/mxbai-embed-large-v1` measured best of every large model at 0.8842 and is also
  faster than bge-large. If HIGH is meant to be "the best large model" rather than "the bge large
  model", change one line in `config.yml`.
- **Nothing large beats LOW by enough to matter.** mxbai is the only large model ahead of the 33M
  default, by 0.0077 mean blocking recall for six times the CPU. Two of the four large models are
  behind it. The LOW default should stay the default.

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

- Every recorded F1 number predates the blocking fixes and the embedding change, so all of them
  understate the current pipeline, worst on amazon-google and walmart-amazon. Re-running the LLM
  baselines costs real inference spend, so it needs a decision rather than an assumption. Related
  open calls: whether to flip the default `--signature-mode` from `generic` to `per-dataset`, and
  whether `er.max_iterations` should be per-dataset given dblp-scholar's precision collapse at
  three passes.

## Lessons

- **Blocking recall is a hard ceiling, so measure it before blaming the LLM.** 2,060 of 3,479 missed
  gold pairs across four full runs were pairs never placed in the same block, against 594 the matcher
  actually looked at and rejected. A blocking-only sweep needs no LLM calls, so it costs CPU time
  instead of inference spend, and it isolates the question completely.
- **Check that a cap cannot swallow the parameter it guards.** `nlist = min(n // target, sqrt(n))`
  makes `target_block_size` silently unreachable above n = target squared, which is 900 records at
  the default. Removing it was a strict improvement on both axes at once: dblp-scholar recall 0.2469
  to 0.9205 *and* blocks shrinking from 84.2 to 29.9 records. A knob that stops responding above a
  threshold is worse than no knob.
- **A measurement taken over a bug ranks the bug.** The embedding sweep was run once before the
  cluster-count fix and once after, and the ranking did not survive: `all-mpnet-base-v2` led the
  first table and finished sixth in the second. Re-run comparative benchmarks after fixing anything
  in the shared path, and discard the earlier numbers rather than reconciling them.
- **Bigger embeddings are not better for blocking.** The 278M `multilingual-e5-base` finished last of
  nine and the 33M `bge-small-en-v1.5` beat it by 0.046 mean recall at 2.6x the speed. Blocking only
  asks whether the true match lands in the same Voronoi cell, and multilingual capacity is wasted on
  English benchmarks. Blocking also re-embeds every record on every ER round, so model size is paid
  `er.max_iterations` times per run, not once.
- **Instruction prefixes barely matter for symmetric tasks.** e5 and bge are trained with a required
  prefix, but adding it to blocking averaged −0.0011 on `multilingual-e5-base`. Both sides of a
  blocking comparison carry the same prefix, so a constant offset largely cancels. It matters for
  asymmetric retrieval, not for clustering.
- **Linux caps a single argv entry at 128 KB (`MAX_ARG_STRLEN`).** Passing entity ids to a subprocess
  as one argument raised `OSError: [Errno 7]` past roughly 13,000 entities. Pass collections through
  a temp file. This one hid for a while because a subagent had patched it at runtime and never
  committed the fix, so the logs showed it working while the committed code could not run at all.
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
- **[OUTDATED - resolved]** **`models.embedding` and the docs disagree.** `config.yml` sets
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

- **An adapter error naming the wrong adapter means a silent fallback fired.** The matcher is
  configured with `dspy.XMLAdapter` and logged `Adapter JSONAdapter failed to parse`, because
  `ChatAdapter.__call__` catches any parse failure, re-runs the request through `JSONAdapter` and
  raises the *fallback's* error. Only 6 of 33 abt-buy blocks were parsing on the first call; the
  other 27 were paying for two inferences and the ones whose fallback also failed were lost
  outright. Replay real blocks with `XMLAdapter(use_json_adapter_fallback=False)` to see the real
  error - a synthetic four-entity block looks like a success from the outside because the fallback
  answers correctly.
- **XML has no null and no bare ampersand.** Three distinct defects, all in the same place. A `dict`
  output field renders as one flat tag, so the model writes JSON into the body and Pydantic gets a
  `str`. An optional field has no null literal, so the model writes the word `null` and Pydantic
  gets `'null'` for a `bool | None`. And `xml.etree.ElementTree` rejects the whole document over one
  unescaped `&`, which every product catalog contains (`Office Home & Student`, `AT&T`). Two
  `field_validator`s on `Entity` and a `RepairingXMLAdapter` that escapes stray metacharacters and
  retries took first-call parses from 6 of 33 to 29 of 33, and abt-buy 1K from F1 0.8875 to 0.9038.
  This supersedes the harmony-envelope lesson above: the envelope only appears in the JSON fallback,
  so keeping the XML parse working avoids it almost entirely.
- **MTEB clustering rank does not predict blocking recall.** `codefuse-ai/F2LLM-0.6B` tops MTEB(eng,
  v2) clustering among models within 3B parameters and 1024 dimensions at 0.6036 and finishes *last*
  on blocking recall at 0.7888. `BAAI/bge-small-en-v1.5` has the second-lowest clustering score in
  the same candidate list and the second-best blocking recall. Choosing embeddings from the
  leaderboard alone would have made the pipeline worse; sweep them on the actual task.
- **Read MTEB from `mteb/results`, not the leaderboard Space.** The Space renders client-side, so a
  fetch of the page returns no numbers. The parquet dataset has all 688 models and every task score.
- **Papers With Code is gone.** `paperswithcode.com` redirects to `huggingface.co/papers/trending`.
  The leaderboards survive at `opencodepapers-b7572d.gitlab.io`, which renders server-side, so the
  entity-resolution boards for abt-buy and amazon-google can be scraped directly. dblp-acm,
  dblp-scholar and walmart-amazon have no board there.
- **Cite what a leaderboard is actually measuring.** Every entry on the ER boards scores pair
  classification: the candidate pairs are given and the model labels them. SERF resolves end to end,
  so its recall carries the pairs blocking never proposed. The numbers belong side by side with that
  caveat attached, not in a rank.
