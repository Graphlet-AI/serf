# SERF Experiment Log

Append-only. One entry per run. No entry, no result. Format and rules from
[docs/RESEARCH_LOOP.md](../docs/RESEARCH_LOOP.md) Section 4, adapted for a
**single, $100-total** budget cap (not the $100/day framing in that doc,
per explicit instruction for this build effort) tracked in
`data/budget/gemini.json` and `data/budget/gpt_oss_120b_maas.json`.

---

## BASE-2026-09-07-001 — no-LLM baselines, dblp-acm

**Hypothesis.** Exact-name-match is a strong baseline on DBLP-ACM (bibliographic
titles are often verbatim-identical across sources); TF-IDF cosine trades some
precision for higher recall; blocking at the project's default
`target_block_size=30` retains most, but not all, true pairs.

**Command.**

```
uv run serf baselines --dataset dblp-acm
uv run serf baselines --dataset dblp-acm --target-block-size 50
uv run serf baselines --dataset dblp-acm --target-block-size 100
```

**Config hash.** N/A (no-LLM, deterministic). **Code.** 80cea52 (D1/D4/D5/D6/D7 fixes
+ budget ledger). **Cost.** $0 (no LLM calls) — Gemini ledger unaffected.

**Result** (2,616 left, 2,294 right, 2,224 ground-truth pairs; `full-table` protocol):

| Baseline | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Random (floor) | 0.0004 | 0.0004 | 0.0004 |
| Exact name match | 0.8854 | 0.8826 | 0.8840 |
| TF-IDF cosine (threshold=0.5) | 0.6394 | 0.9951 | 0.7785 |

Blocking recall ceiling (pair completeness, no matcher can exceed this):

| target_block_size | blocks | avg size | max size | pair completeness |
| --- | --- | --- | --- | --- |
| 30 | 82 | 59.9 | 100 | 0.8291 |
| 50 | 82 | 59.9 | 100 | 0.8291 |
| 100 | 72 | 68.2 | 100 | 0.5288 |

**Verdict.** Exact-name-match (F1 0.884) is, as the research loop predicted,
"shockingly strong" on DBLP-ACM — strong enough that it's a real question
whether an LLM matcher earns its cost here at all; that comparison belongs in
the paper. More surprising: **raising `target_block_size` from 30/50 to 100
made the recall ceiling worse (0.83 -> 0.53), not better.** FAISS `IndexIVFFlat`
clustering is not a simple refinement across target sizes — a different
`nlist` produces a qualitatively different partition, and 30 and 50 happened
to land on the identical partition here (82 blocks either way) while 100 did
not. This means block size cannot be tuned by intuition ("bigger blocks see
more candidates") — it must be measured per dataset, which is exactly
Experiment E1's point. Deferred: a full E1-style sweep (multiple sizes x
multiple datasets x multiple seeds) is out of scope for this session's time
budget; used target_block_size=30 (the existing project default, and the
better of the three measured here) for all GEPA work that follows.

---

## BASE-2026-09-07-002 — no-LLM baselines, abt-buy

**Hypothesis.** Abt-Buy product titles are written independently per retailer
(not shared catalog copy like bibliographic titles), so exact-name-match
should do far worse here than on DBLP-ACM; TF-IDF should do better than exact
match but still leave real headroom for semantic matching.

**Command.** `uv run serf baselines --dataset abt-buy`

**Config hash.** N/A. **Code.** 80cea52. **Cost.** $0.

**Result** (1,076 left, 1,076 right — actual downloaded sizes; 1,097 ground-truth
pairs; `full-table` protocol):

| Baseline | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Random (floor) | 0.0000 | 0.0000 | 0.0000 |
| Exact name match | 1.0000 | 0.0091 | 0.0181 |
| TF-IDF cosine (threshold=0.5) | 0.5694 | 0.5123 | 0.5393 |
| Blocking recall ceiling (target_block_size=30) | -- | 0.8241 | -- |

**Verdict.** Confirmed. Exact-name-match is nearly useless on Abt-Buy (F1
0.018) — the two retailers essentially never write the identical string for
the same product — which is the opposite of DBLP-ACM and is exactly the kind
of domain-dependent behavior the research loop says to expect and report
side-by-side, not average away. TF-IDF (F1 0.54) leaves substantial headroom
(1.0 - 0.54 = 0.46) for a semantic matcher to close. Blocking recall ceiling
(0.82) is close to DBLP-ACM's, suggesting the ~0.83 ceiling at
target_block_size=30 is not dataset-specific in this case.

---

## GEPA-2026-09-07-001 — GEPA optimization, dblp-acm (true-pair-only sample)

**Hypothesis.** GEPA-evolved instructions beat the hand-written `BlockMatch`
signature on a sealed test split, per Experiment E4.

**Command.** `uv run python scripts/run_gepa.py dblp-acm 300 light` (student
`gemini-3.5-flash-lite` @ temperature=0.0, reflection_lm `gemini-3.7-flash`
@ temperature=1.0, `target_block_size=30`, seed=0)

**Config hash.** N/A (script run, not yet wired to config hashing). **Code.**
62609e0. **Cost.** ~$18 (dominated by GEPA's `auto="light"` running many
reflective iterations against a training set that turned out to be trivial
-- see verdict). Cumulative Gemini spend at completion: ~$18.9 of $100.

**Result.** `baseline_f1 = 1.0`, `optimized_f1 = 1.0`. n_train=6, n_val=3,
n_test=0 (!) on the first attempt with `sample_size=40` -- degenerate split,
re-run at `sample_size=300` gave n_train=6, n_val=3, n_test=4, both still 1.0.

**Verdict.** Hypothesis untestable as run: `build_candidate_blocks`'s initial
sampling strategy (only entities that are members of a known true pair) made
every block trivially resolvable -- there was no genuine ambiguity, so both
the hand-written baseline and the optimizer's output scored perfectly and
GEPA's own log shows "All subsample scores perfect for parent N. Skipping."
repeated for over 100 iterations. This is itself a useful finding (Stage 1/2
validation: the base pipeline has no capacity bug on easy DBLP-ACM blocks,
consistent with RESEARCH_LOOP.md Stage 2's overfitting check), but it means
this specific run cannot support or refute the E4 hypothesis. Fixed by mixing
in random distractor entities (commit 62609e0) and moved the next attempt to
`abt-buy`, where the no-LLM baselines (BASE-2026-09-07-002) already show much
more headroom (exact-match F1 0.018 vs. DBLP-ACM's 0.884).

**Side findings from this run** (real bugs, fixed, see commit 377b5d5):
oversized blocks (71, 41 entities against a target of 30) from unsupervised
clustering on a small sample, and `dspy.XMLAdapter` failing to parse model
output containing unescaped `&` (~10% of DBLP-ACM's titles/descriptions
contain one) -- both root-caused with single, minimal-repro diagnostic calls
before committing further budget to full runs, per this doc's own guidance
("smallest experiment that could falsify it").

---

## GEPA-2026-09-07-002 — GEPA optimization, abt-buy (with distractors)

**Hypothesis.** With genuine ambiguity in the training blocks (true pairs
mixed with random distractor entities, not true-pair members only), GEPA's
evolved instructions beat the hand-written baseline on Abt-Buy's sealed test
split, where the no-LLM baselines show real headroom.

**Command.** `uv run python scripts/run_gepa.py abt-buy 150 30` (student
`gemini-3.5-flash-lite` @ temperature=0.0, reflection_lm `gemini-3.7-flash`
@ temperature=1.0, `target_block_size=30`, `max_metric_calls=30`, seed=0)

**Config hash.** N/A. **Code.** f9313f4. **Cost.** ~$1.35 (cumulative Gemini
spend: $32.66 of $100, most of the delta from this run's own baseline eval +
30 metric calls + reflection).

**Result.** `baseline_f1 = 0.95`, `optimized_f1 = 1.00`. n_train=6, n_val=3,
n_test=4. Elapsed 347.6s. Zero XML-parsing failures (the ampersand and
block-size fixes held). Optimized instructions saved to
`data/gepa/abt_buy_optimized_150_30.json`.

The evolved instructions are dramatically more detailed than the hand-written
original -- they name specific normalization rules (strip hyphens/underscores/
case in model numbers), specific brand aliases discovered from the training
data (Eureka/Electrolux, Transcend/TRANSCEND INFORMATION), typo patterns
(Tvio/TiVo), and explicit false-positive guards (don't merge different
storage capacities or receiver tiers of the same brand). Full text: see PR
description / `data/gepa/abt_buy_optimized_150_30.json`.

**Verdict.** Directionally positive and the plumbing worked end-to-end for
the first time this session, but **this result must be read with its sample
size, not despite it**: n_test=4 means going from 0.95 to 1.00 could be a
single corrected example, and n_train=6 is small enough that the optimizer
plausibly partly *memorized* specific training-set entities (the brand
aliases and model numbers named in the evolved instructions are exact
examples from the ~13 candidate blocks built for this run) rather than
learning principles that generalize to arbitrary unseen Abt-Buy products.
Per this doc's own honesty rules ("report the optimization budget", "do not
tune on test"): the test set here is sealed and was not touched during
optimization, but it is too small to license a strong claim. Treat this run
as a validated proof that the full loop (gold-label construction -> GEPA ->
sealed-test evaluation) works correctly and *can* show improvement, not as
publishable evidence that it reliably does. A follow-up with a substantially
larger train/val/test split (all bounded by `max_metric_calls` for
predictable cost/runtime, per the fix in commit 62609e0) is needed before
reporting a GEPA effect size with any confidence.

**Note on GPT-OSS-120B-maas.** Per explicit instruction, the intended
configuration for this and subsequent runs is teacher=`gemini-3.5-flash-lite`,
student=`gpt-oss-120b-maas`, reverting to student=`gemini-3.5-flash-lite`/
teacher=`gemini-3.7-flash` (as run here) only as a stopgap until Vertex AI
credentials (`GOOGLE_CLOUD_PROJECT` + Application Default Credentials) are
available in this environment -- currently only `GEMINI_API_KEY` is
configured. `gpt_oss_120b_maas` ledger: $0 spent to date.

---

## GEPA-2026-09-08-001 — GEPA optimization, abt-buy (full dataset, all blocks)

**Hypothesis.** A much larger, unfiltered training set (all 98 blocks from
the full Abt-Buy dataset, not just the 39 containing a labeled true pair)
gives GEPA a more realistic, harder, and more statistically meaningful signal
than GEPA-2026-09-07-002's 13-block, distractor-only sample, and the
optimizer improves on the hand-written baseline here too.

**Command.** `uv run python scripts/run_gepa.py abt-buy full 150 --all-blocks`
(student `gemini-3.5-flash-lite` @ temperature=0.0, reflection_lm
`gemini-3.7-flash` @ temperature=1.0, `target_block_size=30`,
`max_metric_calls=150`, `require_true_pair=False`, seed=0)

**Config hash.** N/A. **Code.** 3985e1a. **Cost.** ~$6.03 (cumulative Gemini
spend: $40.27 of $100). **Wall clock.** 1832.9s (~30.5 min).

**Result.** `baseline_f1 = optimized_f1 = 0.470097...` (identical to full
float precision). n_train=49, n_val=24, n_test=25. Zero XML-parsing
failures across 150+ metric calls plus baseline/final eval on 25 test
examples each.

Inspecting the saved program confirms the optimizer did not change anything:
`data/gepa/abt_buy_optimized_None_150.json`'s instructions are byte-identical
to the original hand-written `BlockMatch` docstring. The log shows this was
not a silent failure -- GEPA proposed and evaluated at least 6 distinct
candidate instructions against the validation set (aggregate scores around
0.21-0.24), but every one scored at or below the original ("Selected program
0" -- the unmodified baseline -- is the most frequent log line, and multiple
"New subsample score N is not better than old score N, skipping" entries
confirm proposed candidates were compared and rejected, not ignored).

**Verdict.** Genuine negative result, reported as such per this doc's own
honesty rules ("say when a number is worse than the baseline" and "report
the optimization budget"). Three candidate explanations, none mutually
exclusive: (1) 150 metric calls may simply be too small a budget for GEPA to
find an improvement on a 49-example training set with this much
heterogeneity (98 blocks across the whole dataset vs. 13 curated ones) --
the prior positive result used a budget-to-trainset-size ratio of 30:6 = 5x,
this run used 150:49 = 3x; (2) instruction-only optimization (no `dspy.Flex`
restructuring) may have limited headroom against Gemini 3.5 Flash-Lite's
intrinsic ceiling on this harder distribution; (3) F1=0.47 itself is
measured by this module's simplified harness (`dspy.Predict(BlockMatch)`
called directly with block-local mapped ids, scored by `er_metric`'s
pairwise F1) rather than the full production path (`EntityMatcher` +
`UUIDMapper` + `evaluate_er_results`), so it is not directly comparable to
`serf benchmark`'s reported numbers or to MISSION.md's cited baseline
(0.844 F1 on Abt-Buy, different model, different harness) -- the drop from
0.95 (previous run) to 0.47 here is a difficulty/realism change in the
*evaluation set*, not evidence the pipeline regressed.

**Follow-up, not done this session (time/budget):** re-run with a larger
`max_metric_calls` (e.g. 400-500) to test whether budget was the binding
constraint, and/or evaluate the same trainset through the full production
harness for a directly comparable number.
