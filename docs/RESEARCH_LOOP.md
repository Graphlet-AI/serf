# The SERF Research Loop

How we get from where we are to a paper. This document defines the iteration loop, the experiment registry, the metrics, and the rules for what counts as a result.

Read [MISSION.md](MISSION.md) first. This is the operational companion to it.

---

## 1. The loop

Karpathy's recipe for training neural networks is a general recipe for empirical research, and it maps onto entity resolution almost unchanged. The ordering is the point: each stage exists to make the next one trustworthy, and skipping ahead produces numbers you cannot defend.

### Stage 0 — Become one with the data

Before writing a signature or a prompt, read the records. Not a summary of the records — the records.

For each benchmark: pull 200 rows at random, plus 100 known-positive pairs and 100 known-negative pairs that a string-similarity baseline gets wrong. Read them. Write down, in prose, what actually distinguishes a match from a non-match _in that dataset_. Abt-Buy product names are truncated differently on each side. DBLP-Scholar has OCR noise and missing venues. Amazon-Google has manufacturer fields that are sometimes the manufacturer and sometimes the reseller.

Then read the paper that introduced the dataset and find out what its authors said the hard cases were. Köpcke, Thor & Rahm (PVLDB 2010) for the Leipzig four; Mudgal et al. (SIGMOD 2018) for the DeepMatcher splits and for Walmart-Amazon.

**Output:** one short EDA note per dataset in `docs/eda/<dataset>.md`, ending with the field semantics and matching rules that should go into the DSPy signature. These notes are the input to Stage 2, and they are also Section 4 of the paper.

This stage is not optional and it is not delegable to the optimizer. GEPA can discover phrasing. It cannot discover that the `manufacturer` column means two different things.ccccccvejlfcnffkgnujtbjnkbbdbbicrvgdntgvgiur

### Stage 1 — End-to-end skeleton and dumb baselines

Get the whole pipeline running on every dataset with the stupidest possible matcher, and get the evaluation harness producing numbers you trust. Nothing else matters until this works.

Baselines to implement and report, all of them, forever:

| Baseline | Why it's in the paper |
| --- | --- |
| **Random** at the observed positive rate | Establishes the floor. |
| **Exact string match** on the name field | Shockingly strong on DBLP-ACM. Publishing it keeps everyone honest. |
| **Jaccard / TF-IDF cosine over tokens**, threshold tuned on the validation split | The pre-neural baseline every reviewer will ask about. |
| **Blocking-only recall ceiling** | Pair completeness after blocking. This is the hard upper bound on recall; no matcher can exceed it. Report it next to every recall number. |
| **Single-record-per-call LLM (pairwise)** | The comparison the whole paper turns on. Same model, same prompt content, one pair per call. |

If the harness cannot produce all of these from one command per dataset, the harness is not done.

### Stage 2 — Overfit

Take one dataset, take a small slice, and drive the metric as high as it will go with no concern for cost or generality. The purpose is to find the ceiling and to prove the pipeline has no capacity bug. If you cannot get near-perfect F1 on fifty hand-picked easy DBLP-ACM blocks with an expensive model, something is broken in the plumbing, not in the method.

### Stage 3 — Regularize and generalize

Now make it work across all five datasets with one configuration. This is where GEPA runs, on training splits only, evaluated on held-out validation splits. Test splits stay sealed.

### Stage 4 — Tune

Sweep the parameters that matter: block size, blocking model, number of rounds, LLM-call penalty λ in the Flex metric. Every sweep is an experiment with a registered ID and a logged command.

### Stage 5 — Squeeze

Student-teacher transfer to the open models, ensembling, and the last few points of F1.

### The inner loop

Within any stage, one change at a time:

> **hypothesis → smallest experiment that could falsify it → run → record → keep or revert**

Write the hypothesis down _before_ the run, in the experiment log. A hypothesis written afterward is a rationalization, and it is how research projects convince themselves of things that are not true.

**Speed is a feature of the loop, not a nicety.** If an iteration takes an hour, you will run five of them. If it takes two minutes, you will run a hundred and you will find things. Budget real effort on: caching LLM responses so re-runs are free, sampling modes that give a signal in under five minutes, and a `--limit` that actually limits.

---

## 2. Metrics

The literature uses at least five mutually incomparable evaluation protocols and papers routinely quote each other's numbers across them. Ditto reports 75.58 F1 on Amazon-Google; Peeters et al. re-run Ditto on a dataset of the same name and report 80.07. Neither is wrong.

**Therefore: every number SERF reports carries its protocol label. No exceptions.**

### Protocols

| Label | Definition | Used by |
| --- | --- | --- |
| `deepmatcher` | The 3:1:1 train/valid/test split of the pre-blocked labeled pair set. Binary classification over the full test split. | Ditto, Jellyfish, HierGAT, most supervised work |
| `peeters-subsample` | Test sets down-sampled to ~1,250 pairs with ≥150 positives. | Peeters, Steiner & Bizer, EDBT 2025 |
| `blocked-topk` | Blocking retrieves top-_k_ candidates; the system selects among them. | ComEM (Wang et al., COLING 2025) |
| `full-table` | No labeled pair set. The system sees both tables, blocks them itself, and emits clusters. Scored against the ground-truth mapping. | LLM-CER; **SERF's native setting** |
| `cross-dataset` | Train on the other _n−1_ datasets, test on the held-out one. | AnyMatch, Unicorn evaluations |

SERF's primary results are `full-table`, because that is the setting the system actually operates in and the only one that exercises canonicalization. We _additionally_ report `deepmatcher` and `peeters-subsample` so that our numbers can be placed against published ones.

### The metric set

Pairwise scores are computed over the transitive closure of the emitted clusters.

- **Pairwise precision / recall / F1** — the default. Biased toward large clusters, since a cluster of size _n_ contributes _n(n−1)/2_ pairs.
- **Cluster F1** — exact cluster match only. Harsh, and unstable when clusters are large.
- **B-cubed precision / recall / F1** (Bagga & Baldwin 1998) — per-record, then averaged, so entities are weighted equally rather than pairs. This is the honest one for the full-table setting and it is what the clustering literature expects.
- **Generalized Merge Distance** (Menestrina, Whang & Garcia-Molina, PVLDB 2010) — an edit distance over cluster splits and merges. Cite this paper when arguing metrics are not interchangeable; its central result is that existing ER measures _rank the same algorithms differently_.
- **Pair completeness** after blocking — the recall ceiling.
- **Reduction ratio** after blocking.

### Cost, reported alongside every quality number

- **USD per 1,000 input records**, at the published list price of the model used, price date noted.
- **LLM calls per 1,000 input records.**
- **Input and output tokens per 1,000 input records.**
- **Wall-clock seconds per 1,000 input records**, with the concurrency setting.

A result without a cost figure is incomplete. An F1 that costs 100× the baseline is a different result from the same F1 at parity, and the paper's argument depends on saying so.

### Canonicalization quality

Match quality is well-served by the metrics above. Merge quality is not, and there is no standard metric — which is precisely the gap this paper can fill. Proposed, to be pinned down in E7:

- **Field-level accuracy** against a golden record: for each field of each output cluster, does the emitted value match the reference?
- **Provenance completeness**: fraction of input records whose identifier is reachable from the output cluster that contains them. This is the identifier conservation invariant expressed as a score, and it should be exactly 1.0.
- **Hallucinated field rate**: fraction of emitted field values that appear in _no_ input record. This should be zero and will not be.

That last metric is important and cheap to compute. A model asked to merge records will sometimes invent a plausible value. Nobody reports this. We should.

---

## 3. Experiment registry

Every experiment has an ID, a written hypothesis, one command, and a result row. IDs are permanent and are cited from the paper.

### E1 — Block size versus record loss

**The core measurement.** Everything about how the system is configured follows from it.

**Hypothesis.** For a given model there is a block size N* below which record loss is negligible and above which it grows sharply, and that *N is substantially larger than the 9 records that prior work ([LLM-CER](https://arxiv.org/abs/2506.02509)) found optimal. Practitioner experience puts it at 100 for Gemini-class models. Prior work says 9. **One of these is wrong, and finding out which is a publishable result either way.**

**Method.** Sweep block size _N_ ∈ {5, 10, 20, 30, 50, 75, 100, 150, 200, 300} × {`gemini-3.8-flash`, `gemini-3.1-flash-lite`, `gpt-oss-120b`, `gpt-oss-20b`} × 5 datasets × 3 seeds. For each cell record:

**Before running this, fix** `max_tokens`**.** It is hardcoded to 8192 in `matcher.py:73`, and the 2026-09-05 baseline run hit 114 truncations that caused 4,441 records to be dropped and recovered. Output length grows with block size, so an unfixed `max_tokens` makes this experiment measure the output-token ceiling instead of the model's working memory — the two are indistinguishable in the drop-rate metric. Raise it to the model maximum, verify zero truncation warnings, and record truncation count as a separate column so the two failure modes stay apart.

| Quantity | Definition |
| --- | --- |
| **Drop rate** | Fraction of input identifiers absent from the output before recovery runs |
| **Provenance truncation rate** | Fraction of incoming `source_ids` entries dropped from surviving records (the Phase 1 case) |
| **Recovery rate** | Fraction of drops the two-phase recovery caught. Must be 1.0 |
| **Quality** | Pairwise F1, cluster F1, B-cubed |
| **Cost** | USD, calls, tokens per 1,000 records |
| **Position effect** | Drop rate as a function of the record's ordinal position in the prompt |

That last row tests for the lost-in-the-middle effect. If records in the middle of a large block are dropped preferentially, the mitigation is ordering, not a smaller block.

**Deliverable.** The loss curve figure, and a defensible recommended block size per model. This is a headline figure of the paper.

### E2 — Set-level versus pairwise

**Hypothesis.** At equal model and equal information content, matching a whole block in one call beats matching each pair separately, on both F1 and cost.

**Method.** Same model, same blocked candidate set, same field content. Arm A: one call per block. Arm B: one call per pair within the block. Arm C: the ComEM-style "select from _k_ candidates" formulation, as the strongest published set-based baseline. Report F1 and USD/1k for all three.

**Why it matters.** This is the paper's central claim. The prior evidence is encouraging — ComEM reports set-based selection beating pairwise matching by ~14 mean F1 at roughly a fifth of the cost — but it has not been shown at block sizes of 100, and it has not been shown with canonicalization in the same call.

### E3 — Recursive scaling

**Hypothesis.** Reduction compounds across rounds. Blocking and matching _R_ times at a per-round reduction factor _r_ reduces _N_ records to _N·r^R_, at a total cost that is dominated by the first round and converges as _R_ grows.

**Method.** Run 10, 100, 1,000, 10,000 and 100,000 synthetic-plus-benchmark records to convergence. Per round record: input count, output count, round reduction, cumulative reduction, LLM calls, cost, wall-clock, F1 against ground truth, and all four validation gates from [ID_INVARIANTS.md](ID_INVARIANTS.md) §8.

**Deliverable.** The cost-versus-scale curve, and the empirical _r_ per dataset. This is the section that answers "does this actually scale", and the answer has to be a measurement, not the arithmetic in the mission statement.

**Watch for:** quality drift across rounds. Round 3 operates on records that are themselves merge products. If precision degrades as merges compound, that is an important negative result and it goes in the paper.

### E4 — GEPA optimization

**Hypothesis.** GEPA-evolved instructions beat hand-written ones by a margin larger than the seed-to-seed variance, and the margin is larger for weaker models.

**Method.** Baseline is the hand-written signature from Stage 0. Optimize with `dspy.GEPA` on training splits, `auto="light"` then `"medium"`, reflection model Gemini 2.5 Pro (or the strongest available), evaluated on validation. Test stays sealed until the end. Report the optimization budget in rollouts and dollars — GEPA is sample-efficient but not free.

**Report the evolved prompts verbatim in the appendix.** What the optimizer discovered is a finding, not an implementation detail. Diff them against the hand-written versions and say what changed.

**Position against:** the OpenSanctions Pairs work already applies DSPy MIPROv2 to entity matching and reports "consistent but modest" gains of +0.5 to +1.9 F1. If GEPA on block-level matching does substantially better than that, say why. If it does not, say that too.

### E5 — Flex cost reduction

**Hypothesis.** Letting GEPA rewrite the module _code_ (`dspy.Flex`), with a metric that penalizes LLM calls, moves the easy decisions into deterministic Python and cuts cost substantially at equal or better F1.

**Method.** Metric declares `program_trace` and subtracts `λ · n_calls` from the score. Sweep λ ∈ {0, 0.05, 0.1, 0.2, 0.4}. Report the accuracy-versus-cost Pareto frontier and, critically, **what code the optimizer wrote** — which comparisons it decided to settle in Python.

```python
LLM_CALL_PENALTY = 0.15

def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
    correct = ...
    n_calls = len(program_trace) if program_trace else 0
    score = max(0.0, (1.0 if correct else 0.0) - LLM_CALL_PENALTY * n_calls)
    fb = f"{'Correct' if correct else 'Wrong'} — used {n_calls} LM call(s). Settle clear cases in Python."
    return dspy.Prediction(score=score, feedback=fb)
```

**Precedent.** A published Flex case study on a place-matching task moved 75% of records to deterministic Python and went from 90.4% to 95.0% accuracy while getting 28% cheaper and 40% faster. The rules handled easy cases _better_ than a small model did. If that reproduces on ER benchmarks it is a strong result; if it does not, the reason is interesting.

**Prerequisites.** `dspy.Flex` is not in the installed DSPy 3.1.3 — it requires ≥ 3.3.x — and its sandbox needs Deno on the machine. Both are Milestone 4 setup tasks.

### E6 — Student-teacher transfer to open models

**Hypothesis.** A prompt optimized against Gemini transfers to `gpt-oss-120b` and recovers most of the quality at a fraction of the cost; `gpt-oss-20b` recovers less but may still beat the pairwise Gemini baseline.

**Method.** Four arms: (a) Gemini with a hand-written prompt, (b) Gemini GEPA-optimized, (c) `gpt-oss-120b` with Gemini's optimized prompt transferred directly, (d) `gpt-oss-120b` re-optimized with GEPA against itself. Repeat (c) and (d) for `gpt-oss-20b`. Report F1 and USD/1k for all six.

**Models**, via LiteLLM on Vertex AI Model Garden, verified GA:

| Model | LiteLLM route | Input $/1M | Output $/1M | Context |
| --- | --- | --- | --- | --- |
| `gpt-oss-120b` | `vertex_ai/openai/gpt-oss-120b-maas` | 0.15 | 0.60 | 131,072 |
| `gpt-oss-20b` | `vertex_ai/openai/gpt-oss-20b-maas` | 0.075 | 0.30 | 131,072 |

Both support structured output and function calling; neither supports extended thinking. Requires `vertex_ai_project` and `vertex_ai_location` (`us-central1`, or `global` for the 120B).

**The interesting question** is whether re-optimizing per model (arm d) beats transferring (arm c). If transfer is nearly as good, prompt optimization is a portable asset and that is a useful claim.

### E7 — Canonicalization quality

**The novel contribution.** A literature search found no peer-reviewed work that uses an LLM to synthesize a golden record from a matched cluster and evaluates it against a benchmark. Treat that as promising rather than settled, and re-verify before making a first-to-do-this claim in the paper.

**Method.** Construct golden-record ground truth for a subset of clusters on two datasets — the benchmarks give matches, not merged records, so this has to be built. Two sources: derive it programmatically where the ground truth is unambiguous (identical values across a cluster), and hand- label the rest. Then measure field-level accuracy, provenance completeness, and hallucinated field rate (§2).

**Baselines.** Longest-value-wins, most-common-value-wins (the classic survivorship rules from Bleiholder & Naumann's data fusion taxonomy), and a learned record-fusion approach.

### E8 — Protocol-matched comparison to published results

**Method.** Reproduce the exact test splits of the `deepmatcher` and `peeters-subsample` protocols. Run SERF. Place the numbers in a table next to published figures, each cited to a specific table in a specific paper.

**Rules.** Do not compare across protocols. Do not quote a number we have not traced to source. Mark anything unverified as unverified. Where a published number came from a model that was instruction- tuned on the dataset in question — Jellyfish saw Amazon-Google, DBLP-ACM and DBLP-Scholar during tuning — say so in the table.

---

## 4. The experiment log

`experiments/log.md`, append-only. One entry per run. No entry, no result.

```markdown
## E1-2026-09-05-001 — block size 100, gemini-2.0-flash, dblp-acm

**Hypothesis.** Drop rate stays under 2% at block size 100 on a bibliographic dataset.

**Command.** uv run serf benchmark --dataset dblp-acm --target-block-size 100 \
 --model gemini/gemini-3.8-flash --seed 0 --out experiments/runs/E1-...

**Config hash.** a3f9c21 **Code.** 529d6fb **Cost.** projected $0.50 / actual $0.42 — day-to-date $12.30 of $100

**Result.** Drop rate 3.8% (84/2210). Recovery 100%. Pairwise F1 0.781, B-cubed F1 0.744. Drops concentrated in prompt positions 40-70.

**Verdict.** Hypothesis rejected — drop rate is nearly 2x the prediction. The positional concentration suggests lost-in-the-middle rather than a capacity limit. Next: E1-...-002 re-runs at 100 with the block shuffled, to separate position from size.
```

The **verdict** field is the one that matters and the one that gets skipped. Write it.

## 4a. Budget

**$100 per day, maximum, across all models and all experiments. No rollover.**

Every log entry carries a projected cost before the run and a recorded actual after. The two together are how the estimates get good enough to plan with; an experiment nobody can estimate is not ready to run.

Sweeps get staged to fit. E1 is 600 cells and spans several days, so order the grid to produce a usable answer early — sweep block size on one model and one dataset first, since that alone addresses the 9-versus-100 question, then widen.

Two pieces of infrastructure are load-bearing here and belong before the large sweeps:

- **Cost accounting** (T2.5). SERF records no token counts today, so daily spend is currently unmeasurable and the cap is unenforceable. Nothing large should run until this exists.
- **Response caching** (T2.6). A cached re-run costs nothing against the day's cap, which effectively multiplies the budget across the many re-runs that iteration demands.

When a run would cross the cap, it waits for tomorrow. Finishing "just one more dataset" is how a daily cap silently becomes a weekly one.

## 5. Reproducibility rules

1. **One command per number.** If a figure in the paper cannot be regenerated by a command recorded in the log, it does not go in the paper.
2. **Seeds are set and recorded.** LLM temperature is 0. Anything still stochastic gets three seeds and a reported standard deviation.
3. **Every run records the code commit and a config hash.**
4. **Raw outputs are kept**, not just the metrics. Reviewers ask for examples, and the failure cases are where the next hypothesis comes from.
5. **Test splits are sealed** until the final run. Optimization touches training and validation only. Every peek is recorded in the log.
6. **Storage is Apache Iceberg**, so that round _N_ results are queryable as of a snapshot and iterations can be compared by time travel rather than by directory naming conventions.

## 6. Honesty rules

These are here because they are easy to violate under deadline pressure.

- **Report the negative results.** If set-level does not beat pairwise on Amazon-Google, that goes in the table. A paper that wins on five of five datasets and never says what it lost on is not believed.
- **Do not tune on test.** Not once.
- **Report the optimization budget.** GEPA numbers without the rollout count and dollar cost are not comparable to anything.
- **Report contamination risk.** Benchmarks from 2010 are in every model's pretraining data. We cannot eliminate this, but we can measure it: ask the model to complete a record from memory given only its identifier, and report the rate. If a model can recite DBLP-ACM, its DBLP-ACM score means less, and the honest move is to lean on WDC Products' unseen-entity split.
- **Say when a number is worse than the baseline.** SERF's current DBLP-Scholar F1 is 0.671 against a published Ditto figure of 0.956 on a different protocol. Those are not comparable, and the correct thing to write is exactly that — not a favorable selection.
- **Attribute gains honestly.** Swapping `gemini-2.0-flash` for `gemini-3.8-flash` moved F1 by +0.073 to +0.137 with no method change at all. Any future gain has to be reported against a fixed, named model, or it is indistinguishable from having picked a newer one.
