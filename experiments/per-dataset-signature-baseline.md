# Experiment log: per-dataset typed signatures vs the generic BlockMatch signature

- **Run label:** per-dataset signature A/B (unoptimized prompts, both arms)
- **Run id:** `per-dataset-signature-baseline-2026-09-10`
- **Started (UTC):** 2026-09-10T02:14:34Z
- **Finished (UTC):** 2026-09-10T02:47:05Z
- **Student / matching LM:** `openai/gpt-oss-120b-maas` (Vertex AI MaaS, `us-central1`)
- **Teacher LM:** `gemini/gemini-3.5-flash-lite` (not used, no optimization in this experiment)
- **Prompts:** unoptimized in both arms. Arm A is the shared `BlockMatch` signature, arm B is the
  per-dataset signature from `serf.dspy.dataset_signatures`.
- **Blocking:** identical in both arms. multilingual-e5-base name embeddings + FAISS,
  `target_block_size=30`, `max_block_size=90`.
- **Matching:** 1 ER iteration, concurrency 10, `max_tokens=65536`, temperature 0.
- **Sampling:** 1,000 records per dataset, seed 42, sampled by ground-truth match group with
  `serf.eval.sample.sample_records` so gold pairs survive the sample.
- **Code:** commit `ad7f8c5`, run from a pinned checkout so concurrent edits on the branch could not
  change the protocol between arms.
- **MLflow:** experiment `SERF-Entity-Resolution` at `http://127.0.0.1:5001`.
- **CLI:** `serf benchmark --dataset <ds> --signature-mode <generic|per-dataset>
--sample-records 1000 --seed 42 --target-block-size 30 --concurrency 10 --max-iterations 1`

## Verdict

Typed per-dataset signatures beat the generic `BlockMatch` signature on **all five** datasets, by
**+0.0666 to +0.1654 F1**, mean **+0.1128**. They also improved precision on all five and recall on
all five, produced fewer unparseable responses, and cost less than half as much per run. The result
is consistent enough across very different datasets that the direction is not in doubt at this
sample size, though single 1,000-record runs cannot pin down the magnitude precisely.

## Sample composition

Sampling by match group is what makes a 1,000-record sample scoreable: naive uniform sampling over a
5,000-record table keeps only about 4% of gold pairs, because both sides of a pair have to survive.
The blocking recall ceiling below is the fraction of retained gold pairs whose two records land in
the same block, measured offline with the same sample and blocking config. No matcher can exceed it.

| Dataset        | Records sampled | Gold pairs retained | Blocks | Gold pairs co-blocked | Blocking recall ceiling |
| -------------- | --------------: | ------------------: | -----: | --------------------: | ----------------------: |
| dblp-acm       |            1001 |                 476 |     33 |                   454 |                  0.9538 |
| dblp-scholar   |            1000 |                 320 |     33 |                   291 |                  0.9094 |
| abt-buy        |            1000 |                 508 |     31 |                   435 |                  0.8563 |
| amazon-google  |            1002 |                 345 |     31 |                   193 |                  0.5594 |
| walmart-amazon |            1000 |                  75 |     31 |                    65 |                  0.8667 |

Every sample retained hundreds of gold pairs except `walmart-amazon`, which retained 75 because the
full DeepMatcher packaging only has 1,154 matches over 24,628 records. 75 pairs is enough to score
but coarse: one pair is worth 1.3 points of recall there.

## Head to head

| Dataset        | Arm         | Precision | Recall |         F1 |  TP |  FP | Predicted | Elapsed |
| -------------- | ----------- | --------: | -----: | ---------: | --: | --: | --------: | ------: |
| dblp-acm       | generic     |    0.9645 | 0.8571 | **0.9077** | 408 |  15 |       423 |     13s |
| dblp-acm       | per-dataset |    0.9956 | 0.9538 | **0.9742** | 454 |   2 |       456 |     13s |
| dblp-scholar   | generic     |    0.9346 | 0.6250 | **0.7491** | 200 |  14 |       214 |    420s |
| dblp-scholar   | per-dataset |    1.0000 | 0.7719 | **0.8713** | 247 |   0 |       247 |    336s |
| abt-buy        | generic     |    0.9044 | 0.6516 | **0.7574** | 331 |  35 |       366 |     15s |
| abt-buy        | per-dataset |    1.0000 | 0.7244 | **0.8402** | 368 |   0 |       368 |     14s |
| amazon-google  | generic     |    0.7016 | 0.3884 | **0.5000** | 134 |  57 |       191 |    113s |
| amazon-google  | per-dataset |    0.9184 | 0.5217 | **0.6654** | 180 |  16 |       196 |     94s |
| walmart-amazon | generic     |    0.8929 | 0.6667 | **0.7634** |  50 |   6 |        56 |    725s |
| walmart-amazon | per-dataset |    0.9839 | 0.8133 | **0.8905** |  61 |   1 |        62 |     75s |

Elapsed times are not comparable across rows: arms whose exact prompts had already been sent in an
earlier attempt replayed from the DSPy on-disk cache. Token counts below are the honest cost signal.

| Dataset        | F1 generic | F1 per-dataset |    F1 delta | Precision delta | Recall delta |
| -------------- | ---------: | -------------: | ----------: | --------------: | -----------: |
| dblp-acm       |     0.9077 |         0.9742 | **+0.0666** |         +0.0311 |      +0.0966 |
| dblp-scholar   |     0.7491 |         0.8713 | **+0.1222** |         +0.0654 |      +0.1469 |
| abt-buy        |     0.7574 |         0.8402 | **+0.0827** |         +0.0956 |      +0.0728 |
| amazon-google  |     0.5000 |         0.6654 | **+0.1654** |         +0.2168 |      +0.1333 |
| walmart-amazon |     0.7634 |         0.8905 | **+0.1272** |         +0.0910 |      +0.1467 |

Mean F1 delta **+0.1128** across 5 datasets.

## Where the win comes from

**Precision, everywhere.** The typed arm scored perfect precision on `dblp-scholar` and `abt-buy`,
and cut false positives from 57 to 16 on `amazon-google` and from 35 to 0 on `abt-buy`. Two
structural properties explain most of it. First, the typed contract splits the block into two typed
input fields, one per source, so a pair of records from the same table is not expressible; the
generic arm sees one undifferentiated list and can, and does, merge two Amazon listings with each
other. Second, the per-side models name the fields the papers say are decisive, so the model is
looking at `modelno` and `manufacturer` rather than at a prefixed attribute bag.

**Robustness.** The generic arm lost 12 blocks across the five datasets to responses that no adapter
could parse; the typed arm lost 1. A lost block silently contributes zero matches, so those blocks
are pure recall loss. The generic `BlockResolution` output requires the model to echo every entity in
the block, so a 90-record block means a very long structured response; the typed output is just the
list of matched pairs. Shorter outputs are less likely to trip the parser.

| Dataset        | Arm         | Failed blocks | Gold pairs in failed blocks | Recall over answered blocks |
| -------------- | ----------- | ------------: | --------------------------: | --------------------------: |
| dblp-acm       | generic     |             1 |                          37 |                      0.9784 |
| dblp-acm       | per-dataset |             0 |                           0 |                      1.0000 |
| dblp-scholar   | generic     |             4 |                          67 |                      0.8929 |
| dblp-scholar   | per-dataset |             1 |                          10 |                      0.8790 |
| abt-buy        | generic     |             4 |                          54 |                      0.8688 |
| abt-buy        | per-dataset |             0 |                           0 |                      0.8460 |
| amazon-google  | generic     |             2 |                          22 |                      0.7836 |
| amazon-google  | per-dataset |             0 |                           0 |                      0.9326 |
| walmart-amazon | generic     |             1 |                           7 |                      0.8621 |
| walmart-amazon | per-dataset |             0 |                           0 |                      0.9385 |

"Recall over answered blocks" divides true positives by the co-blocked gold pairs that were not
sitting in a block whose call failed. It isolates matching quality from parse robustness, and it is
the one place the picture is mixed: on `dblp-scholar` and `abt-buy` the two arms are within about two
points of each other once the generic arm's lost blocks are excused, while the typed arm is clearly
ahead on `amazon-google` (+0.149) and `walmart-amazon` (+0.076) and saturates `dblp-acm` at 1.0000.
So part of the headline F1 win on the two textual product datasets is the typed arm not throwing
blocks away, and that is a property of the contract, not luck.

**On dblp-acm the typed arm is at the ceiling.** 454 true positives out of 454 co-blocked gold pairs,
with 2 false positives. There is nothing left for a matcher to win there; further gains on that
dataset have to come from blocking.

## Blocking, not matching, is the binding constraint on amazon-google

`amazon-google`'s blocking ceiling is 0.5594: 152 of its 345 retained gold pairs never share a
block, so no matcher can see them. The typed arm recovered 93.3% of what blocking made reachable
while its end-to-end recall still reads 0.5217. Köpcke, Thor and Rahm report that Amazon-Google
titles for the same product are often synonyms with large string distance, which is exactly the case
name embeddings handle worst. Raising `amazon-google` F1 needs better blocking, not a better prompt.

## Cost

Rates: $0.09 per million input tokens, $0.36 per million output tokens. Usage is real, pulled from
MLflow trace metadata (`mlflow.trace.tokenUsage`) and attributed per arm by process path, run
window, and input field names, so concurrent unrelated runs in the same experiment are excluded.
Cache replays emit a trace with no token usage and correctly count as zero.

| Dataset        | Arm             | Billed LLM calls |  Input tokens | Output tokens |        Cost |
| -------------- | --------------- | ---------------: | ------------: | ------------: | ----------: |
| dblp-acm       | generic         |               99 |       359,650 |       321,738 |     $0.1482 |
| dblp-acm       | per-dataset     |               93 |       193,080 |       225,726 |     $0.0986 |
| dblp-scholar   | generic         |               33 |       384,101 |       348,683 |     $0.1601 |
| dblp-scholar   | per-dataset     |               23 |       151,587 |       142,256 |     $0.0649 |
| abt-buy        | generic         |              123 |       444,085 |       468,819 |     $0.2087 |
| abt-buy        | per-dataset     |               62 |       186,888 |       211,758 |     $0.0931 |
| amazon-google  | generic         |               59 |       320,441 |       352,794 |     $0.1558 |
| amazon-google  | per-dataset     |               30 |       111,247 |        92,696 |     $0.0434 |
| walmart-amazon | generic         |               31 |       422,380 |       418,781 |     $0.1888 |
| walmart-amazon | per-dataset     |               27 |       123,297 |        50,518 |     $0.0293 |
| **all**        | **generic**     |              345 | **1,930,657** | **1,910,815** | **$0.8617** |
| **all**        | **per-dataset** |              235 |   **766,099** |   **722,954** | **$0.3292** |
| **all**        | **both**        |              580 | **2,696,756** | **2,633,769** | **$1.1909** |

Call counts include the retries and the repeated attempts described under "Run history", so the
per-dataset totals are not one clean run each. `dblp-scholar` and `walmart-amazon` each ran exactly
once per arm and are therefore the clean cost comparison: the typed arm used 2.5x fewer dollars on
`dblp-scholar` and 6.4x fewer on `walmart-amazon`, almost entirely because it does not echo the
whole block back. Being cheaper and more accurate at the same time is the useful part.

## Not comparable to the earlier raw baseline

`experiments/gpt-oss-120b-raw-baseline.md` reports dblp-acm F1 0.6299, dblp-scholar 0.4110 and
abt-buy 0.6231. **Those numbers are not comparable to anything in this log.** They were measured on
the full tables, not on a 1,000-record match-group sample; with `max_tokens=8192`, where many blocks
were truncated; at concurrency 20; and before the litellm import race described below was fixed, so
an unknown number of their blocks may have been silently dropped. Read this log as generic vs typed
under one fixed protocol, not as progress against the raw baseline.

## Infrastructure finding: a lazy import silently deleted matches

The first attempt at this experiment produced dblp-acm generic F1 **0.1955**, and the reason was not
the prompt. DSPy 3.3.1 imports litellm on first use. With ten matcher threads making their first LLM
call at once, a thread observed a half-executed litellm module and the call died with
`partially initialized module 'litellm' has no attribute 'completion'`. `EntityMatcher.resolve_block`
catches any exception and falls back to `_error_recovery_resolution`, which returns a block with zero
matches, so the run reported a clean 33/33 progress bar and a plausible-looking score while 27 of 33
blocks had never reached the model. Only 3 LLM calls actually happened in a run that claimed to
process 33 blocks.

Fixed in commit `ad7f8c5` by importing litellm in `serf.dspy.lm` and materializing it in the main
thread before any worker thread runs, with a regression test in `tests/test_lm_litellm_import.py`.
After the fix the same arm scored **0.9077**. `serf benchmark` now also prints a warning naming the
number of blocks that fell back to error recovery, so a degraded run cannot be mistaken for a real
one again. Any earlier concurrent matching run on this branch should be treated as suspect.

## Adapter failures observed

The failure the runs still hit is a genuine gpt-oss-120b output-format problem, not a race. The
model sometimes wraps its whole answer in a harmony envelope, `{"final": "{...}"}`, and DSPy's
XMLAdapter fails, retries with JSONAdapter, and fails again with
`Adapter JSONAdapter failed to parse the LM response`. The payload inside the envelope was often a
complete, correct answer: one lost `dblp-acm` block contained 28 well-formed matches. Counts are in
the table above, 12 blocks for the generic arm and 1 for the typed arm. Unwrapping that envelope in a
custom adapter is the highest-value follow-up in this area.

Two smaller notes: `0` candidates were dropped for referencing an unknown record id across all five
typed runs, so the typed contract did not induce id hallucination; and the typed arm skipped blocks
that hold records from only one source (2 on dblp-acm, 10 on dblp-scholar, 0 on abt-buy, 1 on
amazon-google, 4 on walmart-amazon), which is free because no cross-source pair can exist there.

**No 401s and no HTTP 429s occurred in any of the ten runs.** `VertexRefreshingLM` minted tokens
normally and the longest run was 12 minutes, well inside the one-hour token lifetime.

## Run history

The measurement was attempted three times. The first attempt (logs `/opt/cursor/artifacts/ab_*.log`)
was invalidated by the litellm race. The second (`abf_*`) ran after the fix but was interrupted when
concurrent work on the branch changed `serf benchmark` mid-experiment, including a new
three-iteration default and transitive expansion of predicted pairs. The third and reported attempt
(`abp_*`) ran all ten arms from a pinned checkout of commit `ad7f8c5` at `/tmp/serf-ab`, immune to
further edits. Where an arm's prompts had already been sent, the DSPy on-disk cache replayed them
byte for byte: `dblp-acm` generic scored 0.9076751946607341 in both the second and third attempt,
which is also a determinism check on the whole pipeline.

## Artifacts

- Per-arm result JSON: `/tmp/serf-ab/data/benchmarks/ab_pinned/<dataset>[_per-dataset]_results.json`
- Per-arm CLI logs: `/opt/cursor/artifacts/abp_<dataset>_<mode>.log`
- Driver log: `/opt/cursor/artifacts/abp_driver.log`
- This log: `/opt/cursor/artifacts/per-dataset-signature-baseline.md`

---

# Follow-up: applying the BENCHMARKS.md profiling to the same signatures

- **Run label:** per-dataset signature prompt A/B (profiling-derived instructions)
- **Run id:** `signature-benchmarks-lessons-2026-09-11`
- **Started (UTC):** 2026-09-11T19:10:44Z
- **Finished (UTC):** 2026-09-11T22:16:05Z
- **Student / matching LM:** `openai/gpt-oss-120b-maas` (Vertex AI MaaS, `us-central1`)
- **Prompts:** unoptimized in every arm. Only the per-dataset signature docstrings and the typed
  schema field descriptions differ between arms.
- **Blocking:** identical in every arm, so `gold_pairs_retained` is constant per dataset.
- **Matching:** 1 ER iteration, concurrency 20, `max_tokens=65536`, temperature 0.
- **Sampling:** 1,000 records per dataset, seed 42, sampled by ground-truth match group.
- **CLI:** `serf benchmark --dataset <ds> --signature-mode per-dataset --sample-records 1000
--seed 42 --max-iterations 1 --concurrency 20`

## Verdict

Mean F1 over the five datasets rises from **0.8648 to 0.8783 (+0.0135)**. Three datasets improve
(Abt-Buy +0.0375, DBLP-ACM +0.0220, Amazon-Google +0.0098) and two finish at parity within a
fifth of a point (DBLP-Scholar -0.0003, Walmart-Amazon -0.0016). Precision improves on all five.

The useful result is not the gain but what it took to get it. The first arm, which wrote each
measured agreement rate into the prompt as a rule, **lost 3.7 F1 points**.

## Arms

| Dataset        | baseline | vetoes | +coverage | +condensed | +additive | reverted | **adopted** |
| -------------- | -------- | ------ | --------- | ---------- | --------- | -------- | ----------- |
| dblp-acm       | 0.9568   | 0.9799 | 0.9883    | 0.9883     | 0.9883    | 0.9883   | **0.9788**  |
| dblp-scholar   | 0.9192   | 0.8893 | 0.9026    | 0.8966     | 0.8811    | 0.9063   | **0.9189**  |
| abt-buy        | 0.8038   | 0.7747 | 0.8226    | 0.8226     | 0.8226    | 0.8226   | **0.8413**  |
| amazon-google  | 0.7521   | 0.6971 | 0.7661    | 0.7661     | 0.7661    | 0.7661   | **0.7619**  |
| walmart-amazon | 0.8921   | 0.8000 | 0.8507    | 0.8696     | 0.8551    | 0.8657   | **0.8905**  |
| **mean**       | 0.8648   | 0.8282 | 0.8661    | 0.8686     | 0.8626    | 0.8710   | **0.8783**  |

## An agreement rate is not a coverage rate

Every finding in BENCHMARKS.md is reported as agreement on matches against agreement on near
misses. That says how well an attribute _discriminates_ when you can compare it. It says nothing
about how often you can compare it, and the `unusable_` column right beside it does:

| Dataset        | attribute the finding turns on | agrees on matches | unusable on matches |
| -------------- | ------------------------------ | ----------------- | ------------------- |
| walmart-amazon | `modelno`                      | 0.6784            | **0.3181**          |
| amazon-google  | `manufacturer`                 | 0.8029            | **0.8218**          |
| abt-buy        | price comparable               | 0.6150            | **0.7940**          |
| dblp-scholar   | `year`                         | 0.9996            | **0.5410**          |
| dblp-acm       | `year`                         | **1.0000**        | **0.0000**          |

Writing the first number into the prompt without the second turns a decider into a vetoer, and the
matcher then rejects every true pair that could not take the test. Walmart-Amazon lost 0.16 recall
against an unusable rate of 0.32; Amazon-Google lost 0.10. DBLP-ACM was the only dataset to improve
on the first arm, and it is the only one whose constraint holds on every gold pair.

The fix is a shared rule stating that an attribute can only rule a pair out when both records carry
it, that a missing value is not a disagreement, and that whole-field equality is never required.
That single change recovered recall on all four regressed datasets at once.

## Two prompts contradicted themselves

Both are the same mistake: a rule measured on the _columns_ illustrated with an example from the
_titles_, where the string is truncated.

- Abt-Buy called code containment near-decisive (0.8195 of matches against 0.0246 of near misses)
  and then told the matcher to reject a code differing by a trailing character. Containment's own
  worked example, `MDREX55WH` inside `MDREX55WHI`, is exactly that case.
- Walmart-Amazon said to compare title-extracted codes character by character and illustrated it
  with `cb40` against `cb400a`, a truncation that only containment resolves.

## Where the findings did not survive contact

DBLP-Scholar and Walmart-Amazon lost F1 under four successive framings and now keep their original
literature prompts, each with a docstring note recording the measurement so the gap is not mistaken
for unfinished work. Scholar's quirks are corruption an LLM already reads through — `â??`, `2002.0`,
collapsed whitespace — so naming them buys no capability while adding length and rejection pressure.

The one-attribute-difference finding ("the hardest non-matches differ on exactly one short
attribute, so compare that field exactly") is true of all five datasets, but as a _shared_ rule it
cost those two recall: it is only safe advice where the field is present. It now sits in the three
docstrings that measured a gain from it.

## Determinism

DSPy's on-disk completion cache is keyed by the full prompt, so an unchanged docstring replays byte
for byte. Three datasets were untouched between the `+coverage`, `+condensed` and `+additive` arms
and reproduced 0.9883 / 0.8226 / 0.7661 exactly in 16 seconds each. That is both a determinism check
and the reason the arms are comparable: a changed docstring is always a cache miss, so no arm can
contaminate another.

## The adopted prompts at three ER iterations, on all five datasets

Every arm above ran `--max-iterations 1` to keep six arms across five datasets affordable. One pass
is not the pipeline, though: it can only pair records blocking already put together, so the arm
scores understate what a full run reaches and are comparable only to each other. Re-running the
adopted prompts at the default three iterations, same 1,000-record samples at seed 42, same
blocking, same prompts, so the iteration count is the only difference:

| Dataset        | 1 iteration | 3 iterations |   Delta | P @1   | P @3   | R @1   | R @3       | FP @3 |
| -------------- | ----------- | ------------ | ------: | ------ | ------ | ------ | ---------- | ----: |
| dblp-acm       | 0.9693      | **0.9737**   | +0.0044 | 0.9765 | 0.9747 | 0.9622 | **0.9727** |    12 |
| abt-buy        | 0.8413      | **0.9354**   | +0.0941 | 0.9920 | 0.9606 | 0.7303 | **0.9114** |    19 |
| walmart-amazon | 0.8905      | **0.9412**   | +0.0507 | 0.9839 | 0.9231 | 0.8133 | **0.9600** |     6 |
| amazon-google  | 0.7619      | **0.8655**   | +0.1036 | 0.9218 | 0.8732 | 0.6493 | **0.8580** |    43 |
| dblp-scholar   | 0.9244      | **0.9793**   | +0.0549 | 1.0000 | 1.0000 | 0.8594 | **0.9594** |     0 |
| **mean**       | 0.8775      | **0.9390**   | +0.0615 |        |        |        |            |       |

**Three iterations wins on all five datasets, and the mean F1 rises 0.0615.** Recall rose on all
five. Precision fell modestly on four, by between 0.0018 and 0.0608, which is a real cost but a
small one against what recall gains.

This table supersedes two earlier readings in this log, both of which were wrong for the same
reason. The first said three iterations was a uniform win on the strength of two datasets. The
second extended it to five, concluded the opposite — that the mean fell 0.0895 and DBLP-Scholar lost
0.4846 F1 — and blamed closure amplification. That second conclusion was a measurement artifact,
described below. It is kept here rather than deleted because the mistake is instructive.

### The artifact: scoring pairs the ground truth cannot state

These benchmarks are bipartite. `BenchmarkDataset` builds ground truth as
`(a_id, b_id + RIGHT_ID_OFFSET)`, so it only ever asserts that a left record matches a right record.
It never says whether two left records, or two right records, are the same thing.

The ER loop does say that. `merge_matched_entities` collapses each connected component into one
entity, and `expand_pairs` then crosses every member of one entity against every member of the
other. Two entities that each already hold one left and one right record produce four pairs, and
**two of those four are same-source**:

```
round-1 pairs (both correct): [(0, 100000), (1, 100001)]
merged entity 0 covers: [(0, LEFT), (100000, RIGHT)]
merged entity 1 covers: [(1, LEFT), (100001, RIGHT)]
one round-2 decision, assumed perfectly correct: [(0, 1)]
expand_pairs produces:
  (0, 1)                LEFT-LEFT    same-source
  (0, 100001)           LEFT-RIGHT   cross-source
  (1, 100000)           LEFT-RIGHT   cross-source
  (100000, 100001)      RIGHT-RIGHT  same-source
precision ceiling for this one correct decision: 2/4 = 0.50
```

`evaluate_resolution` computed `fp = len(pred) - tp` with no source filter, so every same-source
pair was a guaranteed false positive. A perfectly correct round-2 merge was scored at 50% precision.

Instrumenting the DBLP-Scholar three-iteration run and classifying every false positive settles what
share of the damage this was:

| Class of false positive                      |      Count |
| -------------------------------------------- | ---------: |
| same-source, impossible in the gold          |        783 |
| cross-source, implied by gold but not listed |          0 |
| cross-source, genuine matcher errors         |          0 |
| **precision as scored**                      | **0.2817** |
| **precision over cross-source pairs**        | **1.0000** |

All 783 were same-source. The matcher made **zero** wrong cross-source calls. The entire 0.4846 F1
"collapse" was the scoring harness measuring the gold standard's shape.

`BenchmarkDataset.evaluate` now scores only cross-source pairs, which is the universe its ground
truth is defined over. The same-source claims may well be true — transitive closure does imply
them — but this ground truth does not adjudicate them either way.

### The other half of the same bug: pairs the matcher never wrote down

Dropping same-source pairs fixed the false-positive side and left the false-negative side standing.
`collect_pairs` reads a resolved entity and emits master-to-source pairs — a star, not the clique —
so a group the LLM returned as one entity over two left and two right records reaches scoring as
three pairs when it asserts six:

```
LLM said: one entity covering [1, 2, 100001, 100002]
  as Abzu represents it: id = 1  source_ids = [2, 100001, 100002]

collect_pairs produced 3 pairs (star, master to each source):
  (1, 2)  LEFT-LEFT  same-source
  (1, 100001)  LEFT-RIGHT  cross-source
  (1, 100002)  LEFT-RIGHT  cross-source

the cross-source pairs this entity actually claims (4):
  (1, 100001)  recorded
  (1, 100002)  recorded
  (2, 100001)  MISSING from the scored set
  (2, 100002)  MISSING from the scored set
```

Which pairs of a group get written down is an artifact of how the matcher happened to phrase its
answer, so `evaluate` now resolves the predicted pairs into connected components first and scores
the cross-source pairs those components claim. Scoring the clustering rather than the pair log also
keeps the same-source restriction honest: a same-source merge that welds two gold entities together
is still paid for, because the enlarged cluster goes on to claim cross-source pairs the gold
contradicts.

Measured over the same ten runs, this moves one number. Abt-Buy at three iterations goes from 477 to
482 scored pairs, finding one more true pair and four more false ones: recall 0.9094 to 0.9114, F1
0.9381 to 0.9354. The other nine runs are identical to four decimal places, because outside Abt-Buy
the matcher's stars happened to already cover the cliques. The iteration conclusion is unchanged
either way.

### Why DBLP-Scholar was hit hardest

Its gold standard is the only one with large many-to-many groups, which is exactly the condition
that makes merged entities big and their cross products same-source-heavy:

| Dataset        | Gold pairs | Components | Exactly 1:1 |   Larger | Largest component |
| -------------- | ---------: | ---------: | ----------: | -------: | ----------------: |
| dblp-acm       |       2224 |       2224 |        2224 |        0 |                 2 |
| dblp-scholar   |       5347 |       2351 |        1136 | **1215** |            **37** |
| abt-buy        |       1097 |       1076 |        1055 |       21 |                 3 |
| amazon-google  |       1167 |        995 |         863 |      132 |                 6 |
| walmart-amazon |        962 |        846 |         744 |      102 |                 4 |

DBLP-ACM is perfectly 1:1 with a largest component of 2, so it never built a large entity and never
showed the artifact — which is why the original two-dataset reading looked clean. Google Scholar
legitimately holds several records per publication, giving components up to 37 records.

DBLP-Scholar's gold is also the only one that is not transitively closed: it implies 126 cross-source
pairs it does not list. Those would be unavoidable false positives even under the corrected scoring,
though none appeared in this sample.

### Reading the recall numbers

Abt-Buy's end-to-end recall of 0.9114 is above the 0.8912 single-pass blocking recall measured for
the same dataset, which is only paradoxical if that figure is read as a ceiling. It caps a single
pass, not a run: each round re-blocks what the previous round merged, so pairs blocking missed the
first time can be found later. That is the mechanism iteration buys, and it is worth +0.1811 recall
on Abt-Buy for -0.0314 precision.

### What this means for the standing protocol

`--max-iterations 3` stays, and now on evidence rather than convention: it is worth +0.0615 mean F1
against one iteration, and it wins on every dataset. The earlier suggestion to cap iterations per
dataset or to require more than one predicted pair before joining two multi-record components is
withdrawn — both were solutions to a scoring bug.

Run-to-run variance is not zero. DBLP-ACM's one-iteration F1 read 0.9788 in the `sig_final2` arm and
0.9693 here on the same prompts and sample, so about 0.01 of drift. The effects in the table above
are five to ten times that.

## Why Abzu never saw any of this

Abzu runs the same three stages in the same order with the same block-size schedule,
`max(10, max_block_size // iteration)`, which SERF ported directly. Three things differ, and each
one explains part of why the bug was ours alone.

**Abzu's LLM returns the partition; SERF's returns pairs.** `MultiEntityResolution(CompanyList) ->
CompanyList` asks for the resolved set itself: one record per entity, lowest input id as master,
every other id in `source_ids`, and "Return ALL companies including those that don't merge with
anything". There is no closure step, because there is nothing to close — the model already said
which records are one thing. SERF's signatures emit `candidates[].is_match`, and
`merge_matched_entities` runs union-find to manufacture the partition. That union-find is where a
single spurious pair can weld two components, and `expand_pairs` is where the welded component turns
into a cross product. Neither step exists in Abzu.

**Abzu's evaluation sits inside the loop; SERF's sits after it.** `abzu process er all --iteration N`
is one round per invocation, and for `iteration > 1` the input path is
`config.get("process.kg.er.paths.eval").format(iteration=prev_iteration)` — round N+1 reads round
N's _eval_ output, not its match output. Eval therefore runs before anything downstream sees the
data. Its only mutation is `matches_df.distinct()`; it deliberately does not repair references
("Keep ALL source_uuids, not just validated ones"). But it does report per round, and it fails the
round loudly: coverage of the original companies must be >= 99.99%, invalid references < 0.01%, UUID
overlap with the original and previous iterations < 1.0%. SERF's loop carries the merge output
forward in memory with nothing asserted, and scores `all_predicted_pairs` once at the end.

**Abzu has no accuracy metric at all.** Ripgrep across the repository finds no precision, recall, F1
or gold standard; its `CLAUDE.md` claims "Precision/recall metrics against ground truth" but no such
code exists. What `evaluate_er_matches` measures is lineage conservation and reduction:
`original_coverage_pct`, `source_uuid_error_pct`, `uuid_overlap_with_original_pct`,
`reduction_from_matching_pct`, `ids_dropped_by_baml`, `total_reduction_pct`. So Abzu could not have
hit this bug: scoring pairs against a bipartite gold standard is the thing it never does.

The two metric sets answer different questions and neither contradicts the other. Abzu asks whether a
round lost records; SERF asks whether a round's pairs were right. SERF already has ports of Abzu's
conservation checks in `serf.eval` — `validate_source_uuids` implements the 99.99% coverage rule —
but only `serf evaluate` calls them, so `serf benchmark`'s loop never asserts them per round.

## Artifacts

- Per-arm result JSON: `data/benchmarks/sig_<arm>/<dataset>_per-dataset_results.json`; the adopted
  arm is `sig_final2`
- Per-arm CLI logs: `/opt/cursor/artifacts/sig_<arm>_<dataset>.log`
- Driver logs: `/opt/cursor/artifacts/sig_<arm>_driver.log`
- Corrected one- and three-iteration runs, all five datasets:
  `data/benchmarks/closure_iter{1,3}/<dataset>/<dataset>_per-dataset_results.json`, logs at
  `/opt/cursor/artifacts/closure_<dataset>_iter{1,3}.log`, driver at
  `/opt/cursor/artifacts/rerun_closure_driver.log`, table from
  `/opt/cursor/artifacts/fixed_table.py`
- The same runs scored before the clustering step, differing only on Abt-Buy at three iterations:
  `/opt/cursor/artifacts/fixed_<dataset>_iter{1,3}.log`, driver at
  `/opt/cursor/artifacts/rerun_fixed_driver.log`
- Star versus clique projection of one resolved entity:
  `/opt/cursor/artifacts/star_vs_clique.py`
- Accumulated pairs against the clustering closure, all ten runs:
  `/opt/cursor/artifacts/closure_consistency.log` and `closure_consistency.json` from
  `/opt/cursor/artifacts/closure_consistency.py`
- Loop comparison diagram: `/opt/cursor/artifacts/loop_comparison.png` from
  `/opt/cursor/artifacts/loop_comparison.py`; Abzu read at commit `118f1e9` of
  https://github.com/Graphlet-AI/abzu
- Superseded three-iteration runs scored before the fix:
  `data/benchmarks/tier_removal/<dataset>_per-dataset_results.json` for DBLP-ACM and Abt-Buy,
  `data/benchmarks/iter3_rerun/<dataset>/<dataset>_per-dataset_results.json` for the other three
- False-positive decomposition: `/opt/cursor/artifacts/fp_decomposition.log` and
  `fp_decomposition.json` from `/opt/cursor/artifacts/classify_fp.py`
- Same-source expansion demonstration: `/opt/cursor/artifacts/expansion_same_source.py`
- Gold standard structure: `/opt/cursor/artifacts/gold_structure.py`
- Superseded closure-amplification demonstration:
  `/opt/cursor/artifacts/iteration_closure_mechanism.log`
- Charts: `/opt/cursor/artifacts/signature_f1_by_arm.png`,
  `/opt/cursor/artifacts/signature_precision_recall_shift.png`,
  `/opt/cursor/artifacts/iterations_all_five.png` from
  `/opt/cursor/artifacts/iterations_chart.py`
