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

## Current task: apply BENCHMARKS.md lessons to the per-dataset signatures

`BENCHMARKS.md` records measured discriminativeness, agreement-on-matches against
agreement-on-near-misses, and real match/mismatch examples per dataset. The per-dataset DSPy
signatures in `serf.dspy.dataset_signatures` were written from the _literature_ before that
profiling existed, so several of their instructions are now contradicted by measurement. The
task is to move the measured findings into the signature docstrings and field descriptions and
prove the F1 change with an A/B on identical samples.

Contradictions found by reading the two side by side:

| Dataset        | Signature currently says                     | BENCHMARKS.md measured                                                                                                                            |
| -------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| dblp-acm       | year within one year                         | year agrees on 100% of matches, 12.8% of near misses; the headline mismatch is a conference paper and its journal version, differing only in year |
| dblp-acm       | judge venue semantically                     | venue is a five-row bijection; the crosswalk is known exactly                                                                                     |
| abt-buy        | compare model numbers for equality           | equality finds 47% of matches, containment after stripping all separators finds 82% at a 2.5% false positive rate                                 |
| abt-buy        | price cannot decide a match                  | 61.5% of matches within 25% against 20.1% of near misses, when both sides have one                                                                |
| amazon-google  | price cannot decide a match                  | 79.9% within 25% against 24.0%; the most useful non-title attribute of any product dataset here                                                   |
| amazon-google  | (nothing about codes)                        | only 15% of pairs carry a code, so looking for one is wasted effort                                                                               |
| walmart-amazon | different model numbers are evidence against | true, but Amazon's `modelno` is often descriptive text (`high power`, `with csr`), so the value has to be checked first                           |
| walmart-amazon | an incompatible category matters             | category agrees on 4.4% of matches against 2.2% of near misses and Walmart's is frequently wrong; it is noise                                     |
| dblp-scholar   | years should agree within a year             | Scholar writes `2002.0`, so string comparison agrees on 0% and numeric comparison on 99.96%                                                       |

Protocol: `serf benchmark --signature-mode per-dataset --sample-records 1000 --seed 42
--max-iterations 1`, all five datasets, baseline arm measured before the edit and improved arm
after, same samples and same blocking. One iteration isolates the signature from the multi-round
merge cascade.

### Result: the findings help on three datasets and hurt on two

Five arms, same sample and seed throughout. Where an arm left a prompt byte-identical the DSPy
completion cache replayed it exactly, which is what makes the arms comparable at all.

| Dataset        | baseline | vetoes | +coverage | +condensed | +additive | adopted    | delta   |
| -------------- | -------- | ------ | --------- | ---------- | --------- | ---------- | ------- |
| dblp-acm       | 0.9568   | 0.9799 | 0.9883    | 0.9883     | 0.9883    | **0.9788** | +0.0220 |
| dblp-scholar   | 0.9192   | 0.8893 | 0.9026    | 0.8966     | 0.8811    | **0.9189** | -0.0003 |
| abt-buy        | 0.8038   | 0.7747 | 0.8226    | 0.8226     | 0.8226    | **0.8413** | +0.0375 |
| amazon-google  | 0.7521   | 0.6971 | 0.7661    | 0.7661     | 0.7661    | **0.7619** | +0.0098 |
| walmart-amazon | 0.8921   | 0.8000 | 0.8507    | 0.8696     | 0.8551    | **0.8905** | -0.0016 |
| mean           | 0.8648   | 0.8282 | 0.8661    | 0.8686     | 0.8626    | **0.8783** | +0.0135 |

The first arm wrote each measured agreement rate in as a hard rule and lost 3.7 F1 points,
almost all of it recall, because an agreement rate is not a coverage rate. `modelno` decides
Walmart-Amazon when present and is unusable on 31.8% of its gold pairs; Amazon-Google's
`manufacturer` is unusable on 82.2%; Abt-Buy's price on 79.4%; Scholar omits the year on 54.1%
of rows. Stating the first number without the second turns a decider into a vetoer, and the
matcher rejects every pair that could not take the test. DBLP-ACM was the one dataset that
improved on the first arm, and it is the one whose constraint holds on _every_ gold pair.

Two prompts also contradicted themselves, which is the same bug twice: Abt-Buy called code
containment near-decisive and then told the matcher to reject a code differing by a trailing
character, which is exactly what containment matches (`MDREX55WH` in `MDREX55WHI`);
Walmart-Amazon said to compare title-extracted codes character by character and illustrated it
with `cb40` against `cb400a`, a truncation only containment resolves.

DBLP-Scholar and Walmart-Amazon never recovered across four framings, so they keep the shorter
literature prompt with a note in the docstring saying why. Their quirks are mostly corruption an
LLM reads through unaided — mojibake, `2002.0`, collapsed whitespace — so naming them buys no
capability and costs length and rejection pressure. The last arm also moved the
one-attribute-difference rule out of the shared block rules and into the three docstrings that
gained from it, which is what finally returned those two datasets to parity.

Adopted: three datasets up, two at parity, precision up on all five, mean **+0.0135 F1**. Recorded
in `experiments/per-dataset-signature-baseline.md` and summarised in the README.

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
- [x] Magellan and DeepMatcher technical reports converted to Markdown under `docs/papers/`
- [x] `serf profile-benchmark` added: Spark SQL EDA over the five benchmarks
- [x] `BENCHMARKS.md` written: lessons from both reports plus a page per dataset
- [x] `serf mteb-rank` added: scores candidates on any MTEB category from the `mteb/results` dataset
- [x] Measured which MTEB category predicts blocking recall; clustering +0.0165, PairClassification +0.5714
- [x] Five matching-category candidates swept; `GIST-large-Embedding-v0` best ever at 0.9031 mean
- [x] `--blocking-strategy union` added: name and JSON blockings kept together, not swapped
- [x] Distinct blocked-pair accounting so overlapping blocks do not double-bill the matcher
- [x] Union measured on all five datasets: 0.8765 to 0.9284 mean recall for 1.76x the pairs
- [x] BENCHMARKS.md findings moved into the per-dataset signatures and typed field descriptions
- [x] Six-arm prompt A/B on identical samples: mean F1 0.8648 to 0.8783, precision up on all five
- [x] Findings adopted only where they beat the prompt they replaced; the other two say why in-prompt
- [x] Regression tests pin each adopted instruction to the measurement that forced it
- [x] LOW / HIGH tiers removed; one blocking embedding, `BAAI/bge-small-en-v1.5`, always

## Executor's Feedback or Assistance Requests

- **The HIGH tier needs a decision.** The instruction was to default HIGH to a bge model after
  trying intfloat. `intfloat/multilingual-e5-large-instruct` was measured and lost (0.8705 mean
  blocking recall against LOW's 0.8765), so HIGH is `BAAI/bge-large-en-v1.5` at 0.8665. But
  `mixedbread-ai/mxbai-embed-large-v1` measured best of every large model at 0.8842 and is also
  faster than bge-large. If HIGH is meant to be "the best large model" rather than "the bge large
  model", change one line in `config.yml`.
- **Nothing large beats LOW by enough to matter.** mxbai is the only large model ahead of the 33M
  default, by 0.0077 mean blocking recall for six times the CPU. Two of the four large models are
  behind it. The LOW default should stay the default. _(Superseded below.)_

- **The HIGH tier now has a clear answer.** Selecting candidates on MTEB PairClassification instead
  of clustering surfaced `avsolatorio/GIST-large-Embedding-v0`, which beats the configured
  `bge-large-en-v1.5` on both recall and speed: 0.9031 against 0.8665 mean blocking recall, 250s
  against 315s. It is the first large model to dominate HIGH on both axes, so `models.embedding_high`
  is a one-line change away from being strictly better. Left unchanged pending a decision, because
  changing HIGH invalidates the recorded end-to-end abt-buy comparison. _(Resolved below: the tier
  was removed rather than repointed.)_

- **The tier question is closed: there is no HIGH tier.** Every measurement above said the large tier
  was the worse setting — 0.8665 against 0.8765 mean blocking recall, 0.8963 against 0.9038 end-to-end
  F1 on abt-buy, at six times the embedding CPU — so `--embedding-tier` only ever offered a way to
  make a run slower and less accurate at once. `models.embedding_low` / `models.embedding_high` and
  the CLI option are gone; `models.embedding` now names `BAAI/bge-small-en-v1.5` literally. Swapping
  in `GIST-large-Embedding-v0` is still the one-line change it always was, just without a second
  runtime code path to keep honest.

- **Union blocking is off by default and the default may be wrong.** It gains +0.0520 mean blocking
  recall over name-only on all five datasets, and over +0.10 on both product datasets, for 1.76x the
  pairs the matcher must judge. Pairs are LLM calls, so this is a spend decision rather than a
  technical one. The sensible middle is per-dataset: union on walmart-amazon and amazon-google,
  name-only on dblp-acm where it buys 0.0021 for twice the comparisons.

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

- **Benchmark with `--max-iterations 3`, always.** Standing instruction from the planner. The earlier
  prompt A/B arms used `--max-iterations 1` to keep the six runs affordable, which measures a
  single-pass pipeline rather than the shipping one and makes those F1 numbers incomparable to the
  README and `experiments/` figures. Three is also the `er.max_iterations` config default and the
  CLI default, so the flag is belt-and-braces rather than an override.

- **`config.get(key)` raises `KeyError` for a missing key; it does not return `None`.** It returns a
  default only when one is passed as the second argument. A test that a config key is gone has to use
  `pytest.raises(KeyError)`, not `assert config.get(key) is None`.

- **A tuning knob whose every setting but the default measured worse is not a knob, it is a trap.**
  The LOW / HIGH embedding tiers cost a CLI option, four config keys and a second runtime code path
  so that a user could opt into 0.0100 less mean blocking recall for six times the embedding CPU.
  Configure the alternative candidates in the sweep lists where the sweep can measure them; do not
  promote a losing option into the runtime interface.

- **An agreement rate is not a coverage rate, and only the pair of them licenses a veto.** Writing
  every measured "agrees on X% of matches against Y% of near misses" into the prompts as a rule cost
  3.7 mean F1, almost all of it recall, because the attribute is frequently missing: `modelno` is
  unusable on 31.8% of Walmart-Amazon's gold pairs, `manufacturer` on 82.2% of Amazon-Google's,
  price on 79.4% of Abt-Buy's. DBLP-ACM improved on that arm and is the only dataset whose
  constraint holds on 100% of gold pairs with 0% missing. Always pair a discriminativeness claim in
  a prompt with "and here is what to do when it is absent".
- **Read a new prompt rule against the example that is supposed to prove it.** Two signatures
  contradicted themselves on their own worked examples: Abt-Buy called code containment
  near-decisive and then said to reject codes differing by a trailing character, which is
  `MDREX55WH` inside `MDREX55WHI`; Walmart-Amazon said to compare title codes character by character
  and illustrated it with the truncation `cb40` against `cb400a`. Both were measured findings turned
  into rules that fight each other.
- **A rule true of every dataset can still belong in only some of the prompts.** The
  one-attribute-difference finding holds on all five, but "so compare that field exactly" is only
  safe where the field is present. Shared, it cost DBLP-Scholar and Walmart-Amazon recall; moved
  into the three docstrings that gained from it, it returned those two to parity.
- **More measured detail is not monotonically better.** DBLP-Scholar and Walmart-Amazon lost F1
  under four framings of their own profiling and kept the shorter literature prompt. Their quirks
  are corruption an LLM reads through unaided, so naming them adds length and rejection pressure
  without adding capability. Record why a prompt was left alone, in the prompt, or the next reader
  reads it as unfinished work.
- **The DSPy completion cache makes prompt A/Bs both safe and cheap.** It is keyed by the full
  prompt, so a changed docstring is always a cache miss and no arm can contaminate another, while an
  unchanged one replays byte for byte: three datasets reproduced 0.9883 / 0.8226 / 0.7661 exactly in
  16 seconds across three arms. The corollary is that editing a signature while a run is in flight
  silently changes the arm of any dataset that imports afterwards, so finish edits first.
- **Blocking recall is a hard ceiling, so measure it before blaming the LLM.** 2,060 of 3,479 missed
  gold pairs across four full runs were pairs never placed in the same block, against 594 the matcher
  actually looked at and rejected. A blocking-only sweep needs no LLM calls, so it costs CPU time
  instead of inference spend, and it isolates the question completely.
- **Check that a cap cannot swallow the parameter it guards.** `nlist = min(n // target, sqrt(n))`
  makes `target_block_size` silently unreachable above n = target squared, which is 900 records at
  the default. Removing it was a strict improvement on both axes at once: dblp-scholar recall 0.2469
  to 0.9205 _and_ blocks shrinking from 84.2 to 29.9 records. A knob that stops responding above a
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
  `EntityMatcher` made this easy to hit by building its LM and predictor lazily _inside_ the thread
  pool, so ~8 threads took their first litellm touch at once and each minted its own Vertex token.
  Two fixes, both needed: import litellm at the top of `serf.dspy.lm` so it is executed once
  single-threaded, and `warm_up()` the LM and predictor on the calling thread before
  `resolve_blocks` fans out. The failure was silent - `resolve_block` catches every exception and
  degrades the block to `error_recovery` - so a run reported recall 0.1092 and looked like a result.
  Count and report degraded blocks so this can never read as a real number again.
- **One matching pass cannot beat its own blocking.** Recall of 0.52 / 0.48 / 0.14 against precision
  above 0.9 is the signature of gold pairs that were never co-blocked, not of a weak matcher. The
  benchmark now defaults to `er.max_iterations: 3`, and each round re-blocks the entities the
  previous round merged. The loop used to feed the _matcher's_ resolved entities forward, which the
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
  raises the _fallback's_ error. Only 6 of 33 abt-buy blocks were parsing on the first call; the
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
  v2) clustering among models within 3B parameters and 1024 dimensions at 0.6036 and finishes _last_
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
- **Two of the five benchmarks cannot be scored fairly at face value.** Amazon-Google and
  Walmart-Amazon ship a _labelled candidate set_, not a complete mapping: 11,460 of 4,397,038
  possible pairs (0.26%) and 10,242 of 56,376,996 (0.018%). A pair outside the gold set is usually
  a pair nobody looked at, so a correct match there is scored as a false positive. Of the hardest
  non-gold pairs we surface, only 20.8% and 10.8% are confirmed negatives. Abt-Buy, DBLP-ACM and
  DBLP-Scholar ship complete Leipzig mappings and do not have this problem.
- **Measure an attribute on matches _and_ on near misses, or the number means nothing.** Brand
  agrees on 86.5% of Walmart-Amazon matches, which sounds decisive until you see it agrees on 41.5%
  of the hardest non-matches. `modelno` agrees on 67.8% against 0.26%. The gap is the signal, not
  the level.
- **Normalise before comparing or every attribute looks useless.** DBLP-ACM venue agrees on 0% of
  matches as a string and 100% through a five-row crosswalk. DBLP-Scholar year agrees on 0% as a
  string, because Scholar stores it as a float and writes `2002.0`, and 99.96% as a number. Abt-Buy
  model codes agree on 47.3% exactly and 82.0% by substring containment after stripping every
  separator, which beats the entire product name as a discriminator.
- **`pymupdf4llm` via `uvx` converts these PDFs well.** `uvx --from pymupdf4llm --with pymupdf`
  keeps figure text in `<!-- Start of picture text -->` blocks and reconstructs tables. Strip
  `\ufffd`, trailing whitespace and runs of blank lines afterwards.
- **Cache a Spark view before asking it more than one question.** The near-miss token self-join in
  `serf profile-benchmark` was recomputed for each of eight downstream queries and ran the 64K-row
  dblp-scholar side out of heap. One `.cache()` fixed it and cut the dataset's runtime from 24.3s
  to 13.1s.
- **Do not alias a pair-view column `a_name` when a table has a column called `name`.** Abt-Buy
  does, so the name projection and the shared-column projection collided and Spark raised
  `AMBIGUOUS_REFERENCE`. The aliases are `left_key` and `right_key` now, which cannot collide with
  anything `a_`- or `b_`-prefixed.
- **Pick the benchmark category by the decision, not by the mechanism.** Blocking runs k-means, so
  MTEB _clustering_ looked like the matching category. It correlates with measured blocking recall
  at Spearman +0.0165 over thirteen models, which is nothing, and it would have picked the two worst
  models in the list first and second. `PairClassification` correlates at +0.5714, because its tasks
  ask whether two short texts denote the same thing — the decision blocking has to preserve, not the
  algorithm blocking happens to use. Selecting on it found a model 0.0366 better than the configured
  HIGH tier and faster. There is no MTEB task type named "matching"; `PairClassification` is it.
- **A 0.57 correlation chooses the pool, not the winner.** PairClassification's own top scorer among
  the new candidates, `llmrails/ember-v1` at 87.37, measured _last_ of the five, because it collapses
  to 0.7953 on abt-buy while the others sit near 0.89. Use the leaderboard to decide what to sweep,
  then sweep it.
- **Test an augmentation as an augmentation before writing it off.** JSON blocking lost on 27 of 30
  cells as a _replacement_ for name blocking, by up to 0.34, and that looked conclusive. Kept
  alongside name blocking instead of instead of it, the same JSON view gains +0.0520 mean recall and
  wins on all five datasets, including dblp-acm where JSON alone scores 0.2070. Two weak-but-
  uncorrelated views beat one strong view; the question "does A beat B" is not the question "does
  A add to B".
- **Overlapping blocks break every metric that assumes disjointness.** Once a record is in a name
  block and a JSON block, `block_of[left] == block_of[right]` silently under-counts co-blocking and
  `sum(size * (size - 1) / 2)` double-bills the matcher for pairs both views found. Membership has
  to become `dict[int, set[int]]` with a set-intersection test, and the pair count a distinct count
  over packed `(left << 32) | right` ints. Keep the arithmetic fast path for when the blocks really
  are disjoint, guarded by `sum(sizes) == len(distinct ids)`.
- **JSON texts are roughly 10x longer than names, so large models cost 10x on the JSON pass.** A
  five-dataset union sweep with `bge-large-en-v1.5` projected at four to five hours against 32
  seconds for `bge-small`. Scope large-model union runs to the one dataset that motivates them.
- **`mteb/results` is the leaderboard without the browser.** The Space renders client-side, so
  scraping it returns nothing. The dataset is four parquet parts keyed
  `model_name, model_revision, task_name, split, language, subset, score`; filter `split == "test"`
  and `subset == "default"` and take the max score per model and task across revisions.
- **Some leaderboard leaders will not load.** `KiteFishAI/Nano-Em1-0.6B-v2.1` tops
  PairClassification at 89.9 and raises `Cannot use chat template functions because
tokenizer.chat_template is not set`, because it is an LLM-based embedder. Load-test a candidate
  before planning a sweep around it.
