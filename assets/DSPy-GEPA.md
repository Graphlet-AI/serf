# DSPy GEPA Guide

GEPA (Genetic-Pareto) is the reflective prompt optimizer SERF uses in `serf optimize`.
This guide summarizes how it works, what its authors measured, which knobs matter in
production, and how the pieces map onto this repository. It is a digest of the seven
sources listed at the end, not a replacement for them.

## The One-Paragraph Version

Reinforcement learning adapts an LLM by collapsing each rollout into a scalar reward and
nudging weights. GEPA throws away almost none of that information: it shows a strong
"reflection" LLM the full execution trace — inputs, intermediate reasoning, tool calls,
error strings, and a natural-language explanation of what the metric penalized — and asks
it to rewrite the prompt. Because language is a far richer learning medium than a single
number, a handful of rollouts can produce a large gain. GEPA keeps a pool of candidate
prompts, samples the next one to mutate from the Pareto frontier rather than always taking
the current best, and stops when the rollout budget is exhausted. Weights never change.

## Why It Matters Here

- **It optimizes a program, not a string.** DSPy modules expose their predictors, so GEPA
  rewrites the instruction of each predictor separately and can credit or blame them
  individually. A multi-stage pipeline is optimizable as a unit.
- **It works on API-only models.** No weight access, no fine-tuning infrastructure. That is
  the only option for the Gemini and Vertex AI MaaS models SERF calls.
- **It is cheap enough to rerun.** 100–500 evaluations, versus 5,000–25,000+ for GRPO. When
  a provider ships a new model, rerunning the optimizer is a practical way to find out
  whether it is worth switching.
- **Its output is readable.** The artifact is an instruction string a human can review,
  diff and argue with, which is exactly what the per-dataset signatures in
  `src/serf/dspy/dataset_signatures.py` are.

## How the Algorithm Works

The loop, from the paper's Algorithm 1 and the `gepa` README:

1. **Select** a candidate from the Pareto frontier of the candidate pool.
2. **Execute** it on a stochastically sampled minibatch of the training set, capturing full
   execution traces.
3. **Score and explain** each rollout with the feedback function, which returns a number
   _and_ text.
4. **Choose a module** to improve (round-robin by default, over the program's predictors).
5. **Reflect and mutate** — the reflection LM reads the current instruction, the traces, the
   scores and the feedback, and proposes a replacement instruction for that one module.
6. **Accept or discard** — rerun the mutated candidate on the same minibatch. If it beat its
   parent, evaluate it on the full validation set, record its ancestry, and add it to the
   pool. Otherwise throw it away.
7. **Repeat** until the budget is spent, then return the candidate with the best aggregate
   validation score.

Two further mechanics are worth knowing.

**Pareto-based candidate selection.** For each validation instance, GEPA records which
candidates achieve the best score on _that_ instance. The union of those sets, minus
dominated candidates, is the frontier; the next parent is sampled from it with probability
proportional to how many instances it wins. This is the single most consequential design
choice in the paper. Ablating it costs most of the gain: on four Qwen3-8B benchmarks,
Pareto sampling gives +12.44% over baseline, while always mutating the current best
("SelectBestCandidate", TextGrad's strategy) gives +6.05% and beam search over the top four
(APO's strategy) gives +5.11%. Greedy selection improves fast and then stalls in a local
optimum, burning the remaining budget on one lineage.

**System-aware merge (crossover).** Two candidates from different lineages may have evolved
different modules. Merge builds a new candidate by taking, per module, whichever parent
actually evolved it. `GEPA+Merge` beat plain GEPA by up to 5% on GPT-4.1 Mini but _hurt_ on
Qwen3-8B under identical hyperparameters — the paper attributes this to merge firing before
the lineages had diverged enough, and leaves adaptive scheduling as future work. It is on by
default in `dspy.GEPA` (`use_merge=True`, `max_merge_invocations=5`).

## What the Paper Measured

"GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (Agrawal et al.,
[arXiv:2507.19457](https://arxiv.org/abs/2507.19457), ICLR 2026 Oral) evaluates six
benchmarks — AIME-2025, LiveBench-Math, HotpotQA, IFBench, HoVer and PUPA — on Qwen3-8B and
GPT-4.1 Mini, against MIPROv2, TextGrad, Trace/OptoPrime and GRPO.

| Comparison                   | Result                                                                                                                                 |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| vs GRPO (24,000 rollouts)    | +6% on average and up to +20%, with up to 35x fewer rollouts                                                                           |
| vs MIPROv2                   | +10% or more; +12% on AIME-2025                                                                                                        |
| Aggregate gain, GPT-4.1 Mini | +12.19% (GEPA), +13.33% (GEPA+Merge), against +5.64% for MIPROv2                                                                       |
| Prompt length vs MIPROv2     | Up to 9.2x shorter                                                                                                                     |
| Cross-model transfer         | Prompts optimized on Qwen3-8B gained +9.00% when moved to GPT-4.1 Mini, beating every baseline that optimized directly on GPT-4.1 Mini |

Four findings are load-bearing for anyone using it.

**Most of the budget is validation, not learning.** Rollouts that produce a learning signal
are a small minority; the rest exist to rank candidates. Restricted to train-set rollouts,
GEPA reached its best performance in 79–737 rollouts, and matched GRPO's best validation
score with as few as 6 — up to 78x GRPO's sample efficiency. The practical consequence is
direct: shrink the validation set and you buy more exploration for the same budget.

**Instruction-only optimization now beats instruction-plus-few-shot.** This reverses the
earlier consensus. The paper credits improved instruction-following in modern LLMs plus
GEPA's design, and notes that reflectively evolved instructions show a _lower_
generalization gap than optimized demonstrations — while also being much shorter, since a
single few-shot demonstration for a complex task can dwarf an instruction.

**Optimized prompts contain declarative rules, not disguised examples.** Earlier
instruction optimizers tended to win by smuggling in quasi-exemplars. GEPA's prompts read
like a domain checklist: the published AIME prompt enumerates base-conversion pitfalls,
palindrome bounds and symmetric-sum identities. Think of it as precomputing reasoning during
optimization so the task model does not have to rediscover it per instance.

**It doubles as inference-time search.** Set `valset=trainset` and GEPA deliberately overfits
the batch in front of it, carrying lessons from one task to another. On AMD NPU kernels,
GPT-4o with ten rounds of sequential refinement reached 4.25% mean vector utilization; +RAG
16.33%; +MIPROv2 19.03%; GEPA 30.52% with individual kernels at 70%, and 26.85% from a single
GEPA prompt with no runtime retrieval at all. On 35 KernelBench CUDA tasks it took GPT-4o
from roughly 0% to above 20% `fast1`.

## The `dspy.GEPA` API

```python
optimizer = dspy.GEPA(
    metric=my_feedback_metric,   # required, five-argument signature
    reflection_lm=strong_lm,     # required (or a custom instruction_proposer)
    auto="light",                # exactly one budget knob
    num_threads=8,
    track_stats=True,
    log_dir="data/gepa_logs",
)
optimized = optimizer.compile(student=my_program, trainset=train, valset=val)
```

### Budget

Exactly one of `auto`, `max_full_evals` or `max_metric_calls` must be set. `auto` takes
`"light"`, `"medium"` or `"heavy"`; light evaluates roughly six candidates. DSPy converts the
preset into a `max_metric_calls` figure derived from the number of predictors, the candidate
count and the validation set size, and logs the result as a count of full evaluations before
starting.

### Parameters That Change Outcomes

| Parameter                             | Default         | Why it matters                                                                                                               |
| ------------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `reflection_lm`                       | none            | Required, and must be strong. The docs suggest `dspy.LM("gpt-5", temperature=1.0, max_tokens=32000)`.                        |
| `candidate_selection_strategy`        | `"pareto"`      | `"current_best"` reproduces the greedy ablation that loses half the gain. Leave it alone.                                    |
| `reflection_minibatch_size`           | `3`             | Rollouts shown to the reflection LM per step. Larger batches reveal patterns across failures; the HF cookbook uses 16.       |
| `component_selector`                  | `"round_robin"` | Cycles predictors one per iteration. `"all"` optimizes every predictor simultaneously.                                       |
| `use_merge` / `max_merge_invocations` | `True` / `5`    | Crossover. Helped GPT-4.1 Mini, hurt Qwen3-8B in the paper. Worth ablating per model.                                        |
| `skip_perfect_score`                  | `True`          | Do not waste reflection on examples that already score `perfect_score`.                                                      |
| `add_format_failure_as_feedback`      | `False`         | Turn on when the task model produces unparseable output, so GEPA can learn the format instead of scoring it zero silently.   |
| `failure_score` / `perfect_score`     | `0.0` / `1.0`   | Tell GEPA your metric's range.                                                                                               |
| `track_stats`                         | `False`         | Populates `detailed_results` with every candidate, its lineage, per-instance subscores and the budget consumed at discovery. |
| `log_dir`                             | none            | Writes elaborate logs and every candidate. **Rerunning with the same `log_dir` resumes from the last checkpoint.**           |
| `seed`                                | `0`             | Reproducibility.                                                                                                             |
| `instruction_proposer`                | none            | Replaces the built-in proposer. Needed for multi-modal inputs, hard length caps, or provider-specific formatting.            |
| `gepa_kwargs`                         | none            | Passthrough to `gepa.optimize` for features DSPy does not surface, including stop callbacks and objective frontier types.    |

### The Feedback Metric

This is where most of the leverage is. GEPA's metric takes five arguments and should return
`dspy.Prediction(score=..., feedback=...)`:

```python
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = evaluate(gold, pred)
    if score == 1.0:
        return dspy.Prediction(score=score, feedback=None)
    return dspy.Prediction(
        score=score,
        feedback=f"Wrong. Expected {gold.answer}. Full solution:\n{gold.solution}",
    )
```

`pred_name` and `pred_trace` identify which predictor GEPA is currently seeking feedback
for, and the slice of the trace belonging to it. Returning per-predictor feedback lets GEPA
do credit assignment inside a pipeline instead of blaming the whole program. Return a plain
float and GEPA substitutes `"This trajectory got a score of {score}."` — technically valid,
and it throws away the entire advantage of the method.

The documented recipe for good feedback:

- **Reuse artifacts you already have.** Compiler errors, stack traces, failing unit tests,
  schema validation messages, profiler output. Surfacing them is usually enough.
- **Decompose the score.** Report correctness, latency, cost and safety separately rather
  than as one blended number, so the reflection LM can see which one it is trading away.
- **Name the stage that failed.** For a parse → compile → run → profile pipeline, say which
  step broke and what it said.
- **Enumerate, do not summarize.** For retrieval, list the documents correctly retrieved,
  wrongly retrieved and missed — not just recall.
- **Ground it in a check.** Validators, schemas and simulators where the task is verifiable;
  LLM-as-a-judge where it is not.

Metrics can also return `objective_scores={"quality": ..., "privacy": ...}` alongside the
scalar, and `gepa_kwargs={"frontier_type": "objective"}` will then track a frontier per
named objective. The scalar still gates acceptance and picks the final candidate; objectives
only steer parent and merge selection.

### Train/Validation Split

`compile` uses the two splits for genuinely different jobs. The `trainset` supplies the
minibatches that drive reflective mutation; the `valset` tracks Pareto scores and decides
which candidate is returned. Keep a third, untouched test set for the number you report.

There is no universal ratio, but the direction is clear and follows from "most of the budget
is validation": keep the trainset as large as you can and make the valset the smallest
sample that still represents the downstream distribution. Every candidate is scored on every
validation example, so an oversized valset directly reduces how many candidates GEPA can
explore. DSPy warns above 35 validation examples for exactly this reason. Omitting `valset`
makes GEPA reuse the trainset and deliberately overfit — correct for inference-time search,
wrong when you care about unseen data.

## Production Lessons From an Ablation Study

Decagon ran 19+ single-variable ablations applying GEPA to a production classifier (a
supervisor model that judges conversations and must justify each decision). Their baseline
was 50 train + 50 validation examples, ~150 LLM calls, GPT-4.1 reflection, batch size 10.
Three findings were decisive.

**1. Less data works better.** 20–100 examples consistently beat 500. Going from 50 to 500
made prompts 75% longer and performance 2% worse, for 10x the compute. GEPA's reflection
accumulates observations across iterations, so more distinct failure modes means more edge
cases encoded, which means a bloated prompt that captures training-set minutiae instead of
the task. Their curve is an inverted U: 10 samples −5%, 20 samples +1% (best, 2.5x cheaper
than baseline), 50 baseline, 100 comparable at 1.8x the cost, 500 worst on both axes.

**2. Reflection model quality is non-negotiable.** GPT-4o-mini failed outright — the
"optimized" prompt came back essentially unchanged from the seed. Every frontier model tested
(GPT-4.1, GPT-5.2, Claude Sonnet, Claude Opus) delivered +5–6%. Prompt optimization is
reasoning about reasoning, and small models cannot do it. The cost argument is one-sided:
the reflection LM is called roughly 10–20 times against hundreds of task-model calls, so it
is 5–10% of the total, and economizing there wastes every task-model call instead.

**3. Length constraints are the regularizer.** Unconstrained, GEPA produced prompts over
5,000 characters — both a latency problem and an overfitting problem. A 1,500-character cap
gave 4x compression for 0.8% performance loss; a 500-character cap over-compressed and cost
3%. `dspy.GEPA` has no length parameter, so they encoded the constraint in a custom
`instruction_proposer`.

Their remaining dimensions were low impact: batch size 10 was sufficient, budget showed
diminishing returns past 1x baseline, and a 50/50 train/val split worked fine. Both positive
and negative examples in the feedback helped, so do not filter to failures only.

## Worked Examples

**Haiku generation** (DSPy getting-started). A `ReAct` program with a `dspy.ChainOfThought`
synthesis step, scored by a spaCy-backed metric checking line count, 5-7-5 syllables,
verbatim input reuse, first-person voice, part-of-speech balance, adjective and article
density, and tense. `gpt-5.4-nano` went from 78.1% to 90.1% with `gpt-5.4` reflecting and
`auto="light"` — past unoptimized `gpt-5.4`'s own 82.4%. The optimized instruction grew from
one line to a structured spec with ranked success criteria, per-input handling rules and a
final quality checklist.

**AIME math** (the `gepa` quickstart, without DSPy). `gepa.optimize` on a bare system-prompt
dict, `task_lm="openai/gpt-4.1-mini"`, `reflection_lm="openai/gpt-5"`,
`max_metric_calls=150`: 46.6% → 56.6% on AIME 2025.

**NuminaMath** (HuggingFace cookbook). `gpt-4.1-nano` as the task LM,
`qwen3-next-80b-a3b-thinking` as the reflection LM, 112 train / 22 val examples,
`auto="light"`, `reflection_minibatch_size=16`, `add_format_failure_as_feedback=True`. The
metric appends the full worked solution to the feedback on every failure. Accuracy went
52.2% → 57.8% for under $0.50 total. The author attributes the modest gain to the tiny
training set and light budget, and frames the design as asymmetric: a cheap model for ~99%
of calls, a smart one for the ~1% that reflect.

**DSPy tutorials** cover four more cases: AIME with a single `ChainOfThought` (+10% on
GPT-4.1 Mini), enterprise structured extraction using predictor-level feedback across a
three-part task, privacy-conscious delegation improving within a single iteration from an
LLM-as-a-judge metric, and code-backdoor classification using a comparative metric so the
prompt learns what separates positives from negatives.

## Beyond DSPy: The `gepa` Package

`dspy.GEPA` is a wrapper; the engine lives in [`gepa-ai/gepa`](https://github.com/gepa-ai/gepa)
(MIT, `uv add gepa` to use it directly). Three things there are worth knowing even if you
only ever use DSPy.

**Adapters.** GEPA connects to any system through `GEPAAdapter` — implement `evaluate` and
`make_reflective_dataset`. DSPy's own integration is just an adapter. Built-ins cover
single-turn system prompts, logprob-aware classification, RAG over ChromaDB, Weaviate,
Qdrant and Pinecone, MCP tool descriptions, LangChain pipelines, the Terminus terminal
agent, and a full-program adapter that evolves DSPy signatures, modules and control flow
together (67% → 93% on MATH).

**`optimize_anything`.** Any text artifact is optimizable if you can evaluate it: code, agent
architectures, configuration, SVGs. You supply an evaluator and call `oa.log(...)` with
whatever diagnostics you have; those become the reflection signal. The repo calls this
Actionable Side Information (ASI) and describes it as the text-optimization analogue of a
gradient — which is the right mental model for writing feedback functions generally.

**Reported results beyond prompts.** ARC-AGI agent accuracy 32% → 89% via architecture
discovery; a GEPA-discovered cloud scheduling policy beating expert heuristics by 40.2% on
cost; a coding agent's Jinja resolve rate 55% → 82% through auto-learned skills; Databricks
reaching state-of-the-art enterprise agents 90x cheaper. It is integrated into MLflow
(`mlflow.genai.optimize_prompts()`), Comet ML Opik, Pydantic AI and Google's ADK.

**When GEPA is the right tool.** Expensive rollouts (simulations, tool-using agents, slow
compiles) where 100–500 evaluations is the difference between feasible and not; scarce data,
down to a handful of examples; API-only models; and cases where you need to read why the
prompt changed. It also complements rather than replaces RL — optimize prompts first, then
fine-tune.

## How SERF Uses It

`src/serf/dspy/optimize.py` holds the whole integration, driven by `serf optimize`.

- **Student and teacher are separate models**, exactly the asymmetry the sources recommend.
  `models.student` is `openai/gpt-oss-120b-maas` and executes the task; `models.teacher` is
  `gemini/gemini-3.5-flash-lite` and is passed as `reflection_lm` at `temperature=1.0`.
- **`er_metric` is the feedback function.** It scores a `BlockMatch` prediction with F1 over
  normalized match pairs and then enumerates the errors: `Missed true pairs: [...]` and
  `Extra predicted pairs: [...]`. That is the "enumerate, do not summarize" rule — the
  reflection LM sees which pairs were wrong, not just that F1 was 0.62.
- **Splits are built by `prepare_dataset_splits`**, which samples records by match group so
  gold pairs survive, then blocks _inside_ each split so no records leak across them, and
  logs how many gold pairs actually landed inside blocks per split. A split whose blocks
  contain no gold pairs cannot be scored, and it warns rather than silently reporting zero.
- **Budget and logging come from `config.yml`**: `optimize.auto` (`light`),
  `optimize.num_threads`, `optimize.log_dir` (`data/gepa_logs`) and `optimize.seed`.
  `track_stats=True` is always on.
- **`log_dir` resumes.** Because GEPA checkpoints into `log_dir` and reuses it, two runs
  sharing a directory will continue each other rather than start fresh. Give each run its
  own directory when you want independent runs.

Two caveats specific to this repository. The matcher output is a nested Pydantic model
rendered through `RepairingXMLAdapter`, so `add_format_failure_as_feedback=True` is worth
considering — a block that fails to parse currently scores as a miss without telling the
reflection LM why. And per the Decagon length finding, the per-dataset signatures in
`src/serf/dspy/dataset_signatures.py` are already 5,900–8,800 characters of hand-written
instruction; anything GEPA adds on top compounds that, so measure end-to-end F1 rather than
assuming a longer prompt is a better one.

## Sources

- [dspy.GEPA API overview](https://dspy.ai/api/optimizers/GEPA/overview/) — full parameter
  reference, feedback-metric protocol, `DspyGEPAResult`, algorithm summary.
- [GEPA optimization, getting started](https://dspy.ai/getting-started/gepa-optimization/) —
  why optimize prompts, the haiku walkthrough, train/val guidance.
- [Reflective Prompt Evolution with GEPA, tutorials](https://dspy.ai/tutorials/gepa_ai_program/) —
  AIME, enterprise extraction, privacy-conscious delegation, backdoor classification.
- [HuggingFace cookbook: Prompt Optimization with DSPy GEPA](https://huggingface.co/learn/cookbook/en/dspy_gepa) —
  end-to-end NuminaMath notebook, two-model architecture rationale.
- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/pdf/2507.19457) —
  Agrawal et al., ICLR 2026 Oral, arXiv:2507.19457.
- [Optimizing GEPA for production](https://decagon.ai/blog/optimizing-gepa-for-production) —
  Decagon, 19+ ablations on sample size, reflection model and length constraints.
- [gepa-ai/gepa](https://github.com/gepa-ai/gepa) — the optimization engine, adapters,
  `optimize_anything`, integrations and the running list of reported uses.
