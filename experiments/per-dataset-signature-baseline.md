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

| Dataset | Records sampled | Gold pairs retained | Blocks | Gold pairs co-blocked | Blocking recall ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| dblp-acm | 1001 | 476 | 33 | 454 | 0.9538 |
| dblp-scholar | 1000 | 320 | 33 | 291 | 0.9094 |
| abt-buy | 1000 | 508 | 31 | 435 | 0.8563 |
| amazon-google | 1002 | 345 | 31 | 193 | 0.5594 |
| walmart-amazon | 1000 | 75 | 31 | 65 | 0.8667 |

Every sample retained hundreds of gold pairs except `walmart-amazon`, which retained 75 because the
full DeepMatcher packaging only has 1,154 matches over 24,628 records. 75 pairs is enough to score
but coarse: one pair is worth 1.3 points of recall there.

## Head to head

| Dataset | Arm | Precision | Recall | F1 | TP | FP | Predicted | Elapsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dblp-acm | generic | 0.9645 | 0.8571 | **0.9077** | 408 | 15 | 423 | 13s |
| dblp-acm | per-dataset | 0.9956 | 0.9538 | **0.9742** | 454 | 2 | 456 | 13s |
| dblp-scholar | generic | 0.9346 | 0.6250 | **0.7491** | 200 | 14 | 214 | 420s |
| dblp-scholar | per-dataset | 1.0000 | 0.7719 | **0.8713** | 247 | 0 | 247 | 336s |
| abt-buy | generic | 0.9044 | 0.6516 | **0.7574** | 331 | 35 | 366 | 15s |
| abt-buy | per-dataset | 1.0000 | 0.7244 | **0.8402** | 368 | 0 | 368 | 14s |
| amazon-google | generic | 0.7016 | 0.3884 | **0.5000** | 134 | 57 | 191 | 113s |
| amazon-google | per-dataset | 0.9184 | 0.5217 | **0.6654** | 180 | 16 | 196 | 94s |
| walmart-amazon | generic | 0.8929 | 0.6667 | **0.7634** | 50 | 6 | 56 | 725s |
| walmart-amazon | per-dataset | 0.9839 | 0.8133 | **0.8905** | 61 | 1 | 62 | 75s |

Elapsed times are not comparable across rows: arms whose exact prompts had already been sent in an
earlier attempt replayed from the DSPy on-disk cache. Token counts below are the honest cost signal.

| Dataset | F1 generic | F1 per-dataset | F1 delta | Precision delta | Recall delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| dblp-acm | 0.9077 | 0.9742 | **+0.0666** | +0.0311 | +0.0966 |
| dblp-scholar | 0.7491 | 0.8713 | **+0.1222** | +0.0654 | +0.1469 |
| abt-buy | 0.7574 | 0.8402 | **+0.0827** | +0.0956 | +0.0728 |
| amazon-google | 0.5000 | 0.6654 | **+0.1654** | +0.2168 | +0.1333 |
| walmart-amazon | 0.7634 | 0.8905 | **+0.1272** | +0.0910 | +0.1467 |

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

| Dataset | Arm | Failed blocks | Gold pairs in failed blocks | Recall over answered blocks |
| --- | --- | ---: | ---: | ---: |
| dblp-acm | generic | 1 | 37 | 0.9784 |
| dblp-acm | per-dataset | 0 | 0 | 1.0000 |
| dblp-scholar | generic | 4 | 67 | 0.8929 |
| dblp-scholar | per-dataset | 1 | 10 | 0.8790 |
| abt-buy | generic | 4 | 54 | 0.8688 |
| abt-buy | per-dataset | 0 | 0 | 0.8460 |
| amazon-google | generic | 2 | 22 | 0.7836 |
| amazon-google | per-dataset | 0 | 0 | 0.9326 |
| walmart-amazon | generic | 1 | 7 | 0.8621 |
| walmart-amazon | per-dataset | 0 | 0 | 0.9385 |

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

| Dataset | Arm | Billed LLM calls | Input tokens | Output tokens | Cost |
| --- | --- | ---: | ---: | ---: | ---: |
| dblp-acm | generic | 99 | 359,650 | 321,738 | $0.1482 |
| dblp-acm | per-dataset | 93 | 193,080 | 225,726 | $0.0986 |
| dblp-scholar | generic | 33 | 384,101 | 348,683 | $0.1601 |
| dblp-scholar | per-dataset | 23 | 151,587 | 142,256 | $0.0649 |
| abt-buy | generic | 123 | 444,085 | 468,819 | $0.2087 |
| abt-buy | per-dataset | 62 | 186,888 | 211,758 | $0.0931 |
| amazon-google | generic | 59 | 320,441 | 352,794 | $0.1558 |
| amazon-google | per-dataset | 30 | 111,247 | 92,696 | $0.0434 |
| walmart-amazon | generic | 31 | 422,380 | 418,781 | $0.1888 |
| walmart-amazon | per-dataset | 27 | 123,297 | 50,518 | $0.0293 |
| **all** | **generic** | 345 | **1,930,657** | **1,910,815** | **$0.8617** |
| **all** | **per-dataset** | 235 | **766,099** | **722,954** | **$0.3292** |
| **all** | **both** | 580 | **2,696,756** | **2,633,769** | **$1.1909** |

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
