# DSPy Flex

A guide to `dspy.Flex`, the module whose optimizable parameter is its own source code. Written
against dspy 3.3.1, with every API claim below checked against the installed package rather than
transcribed from the docs. Read [assets/DSPy-GEPA.md](DSPy-GEPA.md) first: Flex is not a separate
optimizer, it is a thing GEPA can optimize.

## The One-Paragraph Version

Every other DSPy module fixes its structure the moment you construct it. `Predict` is one call,
`ChainOfThought` is one call with reasoning, `ReAct` is a tool loop, and optimization only tunes the
prompt wrapped around that fixed shape. `dspy.Flex` moves the shape itself into the search space.
You construct it from a signature, it starts as a single `dspy.Predict` baseline, and `dspy.GEPA`
rewrites its entire implementation — how many predictors there are, which primitives they use, and
what runs in plain Python instead of an LM at all. The reported effect on an entity resolution task
is 90.4% to 95.0% accuracy while routing 75% of records through deterministic code, which made the
optimized program 28% cheaper and 40% faster than the unoptimized one it started from.

## Why It Matters Here

The benchmark both write-ups lead with is entity resolution. Given two place listings, decide
whether they describe the same physical place: `KIN CAFE` and `KIN` at one address are the same
place, `CONCESSION #2 KEN MERCER SPORTS PARK` and `KEN MERCER SPORTS PARK` at one address are not.
That is SERF's problem statement with a different pair of tables.

It matters for a second reason that is specific to how this repo already works. SERF's per-dataset
signatures hold measured decision rules that are _already_ deterministic — "a code is a run of four
or more characters containing at least one digit", "strip every space, dash and slash out of the
other side's name and test containment", "require the years to be equal". Those are instructions to
an LM to behave like a function. Flex is the mechanism for letting the optimizer write the function.

## What Changes: The Optimizable Unit

|                 | Ordinary module                           | `dspy.Flex`                                   |
| --------------- | ----------------------------------------- | --------------------------------------------- |
| Tunable surface | each predictor's instruction string       | one `dspy.Module` subclass as source          |
| Exposed as      | `predictor.signature.instructions`        | `flex.module_src`                             |
| GEPA calls it   | an instruction component                  | a code component                              |
| Proposer        | the instruction proposer                  | the code proposer                             |
| Reflects on     | one predictor's inputs, outputs, feedback | the whole program's inputs, outputs, feedback |
| Saved state     | instructions and demos                    | `{"module_src": ..., "lm": ...}`              |

A `Flex` subclasses both `Module` and `Parameter`, which is what makes a parent program treat it as
one opaque leaf. Verified on the installed package with a parent holding one `Flex` named `flex` and
one `Predict` named `plain`:

```text
parent.named_parameters() paths: ['flex', 'plain']
parent.named_predictors() paths: ['plain']
enumerate_flex_submodules(parent): ['flex']
flex.named_predictors() == []
```

So `BootstrapFewShot`, `set_lm` and the demo optimizers never see inside a `Flex`, and the `Flex`
reports no predictors of its own. Its update unit is the code, not the predictors the code happens
to construct — which is the only consistent choice, because the next candidate may not construct
them at all.

## The Baseline It Starts From

`dspy.Flex(sig)` is immediately runnable and behaves like `dspy.Predict(sig)`. The starting source
is generated, not hand-written, and it carries the signature's instructions into a
`dspy.Signature(...)` string so the docstring is not lost. Actual output for a three-field signature
whose docstring reads "Decide whether two listings are the same physical place.":

```python
class SamePlaceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(dspy.Signature('left: str, right: str -> is_same: bool', 'Decide whether two listings are the same physical place.'))

    def forward(self, **inputs):
        result = self.predict(**inputs)
        return dspy.Prediction(is_same=result.is_same)
```

Pass `tools=[...]` and the baseline is a `dspy.RLM` instead, so it can call them from the start.
Confirmed: `self.rlm = dspy.RLM(dspy.Signature('order: str -> total_cents: int', ...), tools=[lookup_sku])`.

## How the Optimization Works

1. **GEPA discovers `Flex` by type.** `compile` calls `enumerate_flex_submodules(student)` and
   splits its work: each `Flex` becomes a code component keyed by its parameter path, every other
   predictor stays an instruction component. Both are seeded in the same candidate dict
   (`seed_candidate[path] = flex.module_src`) and both count toward the auto budget
   (`len(instruction_predictors) + len(flex_submodules)`). All three lines verified present in
   `GEPA.compile`.
2. **The code proposer sees five things.** `CodeProposalSignature` declares exactly
   `task_description`, `available_context`, `primitives_catalog`, `current_source` and `failures`,
   and returns `revised_source`. The catalog is a fixed 13,135-character document that defines the
   subset of `dspy` the generated code may use.
3. **A code component reflects on whole-program behavior.** Instruction optimization looks at one
   predictor in isolation; code optimization cannot, because the predictors are part of what is
   being rewritten. `code_reflective_records` is documented as using "the whole program's
   inputs/outputs/feedback per example" instead.
4. **The proposer is told to optimize instructions too.** This is easy to miss and it matters: the
   sub-predictors a `Flex` constructs live inside `module_src`, so the code proposer is the _only_
   thing that ever writes their prompts. Its own instructions say so — "These predictors are inside
   a dspy.Flex module, so this source is the ONLY place their prompts get optimized" — and it is
   told to "Fix instructions when a failure is about WHAT the model should do or know; change the
   structure when it is about HOW steps are wired."
5. **A broken candidate costs a search step, not the run.** Source that does not parse raises when
   GEPA binds it, which is caught and scored at the failure score for the whole batch. Source that
   parses but throws at runtime fails per example, scored at the failure score in its own slot by
   example index, so the surviving scores stay aligned. A proposer exception keeps the original
   source: `propose_code` is documented as "keep the original on failure".
6. **Winners are kept the same way prompts are.** Candidates advance on the per-instance Pareto
   frontier, which is the mechanism the GEPA paper measured at +12.44% against +6.05% for greedy
   selection. Nothing about that changes for code.

## What It Measured

One task, two sources reporting the same numbers. Execution model `claude-haiku-4-5`, reflection
model `claude-opus-5`, 1,029 labeled pairs, 240 class-balanced held-out records so 50% is chance,
caches disabled so the cost is what cold production traffic pays.

| Program                 | λ    | Accuracy  | LM calls / record | $ / 1k records | Mean latency |
| ----------------------- | ---- | --------- | ----------------- | -------------- | ------------ |
| `dspy.Predict` baseline | n/a  | 90.4%     | 1.00              | $0.98          | 1,924 ms     |
| GEPA, prompt only       | n/a  | 92.5%     | 1.00              | $2.88          | 2,841 ms     |
| Flex + GEPA             | 0    | **95.0%** | 0.25              | $0.70          | 1,155 ms     |
| Flex + GEPA             | 0.05 | 94.6%     | 0.17              | $0.45          | 726 ms       |
| Flex + GEPA             | 0.1  | 90.8%     | 0.07              | $0.18          | 347 ms       |
| Flex + GEPA             | 0.2  | 91.7%     | 0.08              | $0.09          | 135 ms       |
| Flex + GEPA             | 0.4  | 92.1%     | 0.004             | $0.01          | 65 ms        |

Four findings worth carrying away.

**Prompt-only optimization has one lever, and pulling it costs money on every record.** GEPA without
Flex bought 2.1 points of accuracy by writing a much longer instruction, and every record then pays
for those tokens: $2.88 per thousand against $0.98, 2.9x the baseline cost and 48% slower. This is
the single most relevant number in the table for this repo, whose per-dataset instructions already
run 5,963 to 8,922 characters.

**The optimizer wrote code even when calls were free.** At λ=0 the metric scored accuracy alone; the
only nudge was textual feedback asking for cases to be settled in code. It still routed 75% of
records through deterministic Python and beat calling the model every time, 95.0% against 90.4% with
McNemar p=0.019. The rules handle the easy cases better than a small model does, and the model only
sees what actually needs judgment.

**High penalties buy near-parity accuracy for a hundredth of the cost.** At λ=0.4 the program called
the model once across 240 records and held 92.1%, statistically indistinguishable from the
always-call baseline, at roughly 1/100th the cost and 1/30th the latency.

**The accuracy curve is not monotonic in λ.** 95.0, 94.6, 90.8, 91.7, 92.1 as λ rises. Read the
penalty as a dial to sweep, not a value to reason your way to.

A pilot on SWE-bench Pro is worth knowing about but not worth relying on: Haiku 4.5 resolved 0 of 12
sampled issues, and after GEPA on the Flex program with `max_metric_calls` capped at 60 it resolved
4 of 12, having designed a workflow that mixes Python and LM calls to research, draft, evaluate,
repair and submit. Twelve issues is twelve issues.

## Reading the Code It Wrote

At λ=0.4 the winning program is about two hundred lines of model-written Python in three stages.
Normalize: names uppercased and stripped of franchise numbers (`#30696`), legal suffixes (`LLC`,
`INC`) and some forty generic business words (`CAFE`, `RESTAURANT`, `GRILL`), so `KIN CAFE` reduces
to `KIN`; addresses parsed into a house number and a street core so `AVE` versus `AVENUE` can never
cause a mismatch. Compare: distinctive name tokens scored zero to one by fuzzy similarity and binned
into confident match, confident miss, and unsure. Decide: each bin gets rules combining the name
verdict, the address parts and the geodesic distance — same name and address within 400 m is a
match; same name, different house numbers, more than 120 m apart is two branches of one brand.

Only when no rule fires does the record reach the model, and it does not go alone: the module
forwards its own parsed house numbers, street cores and similarity scores as extra input fields, and
the judge's instructions distill what the optimizer learned into numbered domain rules. The comment
the optimizer wrote at the top of its own architecture is the summary: "the LLM is a LAST-RESORT
fallback."

That shape — normalize, compare, decide, escalate the residue — is what SERF's `abt-buy` docstring
currently asks an LM to perform in its head.

## The `dspy.Flex` API

```python
import dspy

program = dspy.Flex(SamePlace)  # was: dspy.Predict(SamePlace)

dspy.configure(lm=dspy.LM("anthropic/claude-haiku-4-5"))  # runs the program
reflection = dspy.LM("anthropic/claude-opus-5")  # writes the code and the prompts

optimized = dspy.GEPA(
    metric=make_metric(penalty=0.2),
    reflection_lm=reflection,
    max_metric_calls=400,
).compile(program, trainset=train, valset=val)

print(optimized.module_src)  # the discovered program
optimized.save("program.json")
```

Constructor parameters and their defaults, read off `Flex.__init__` in the installed package:

| Parameter             | Default                  | What it does                                                                                                                                    |
| --------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `signature`           | required, positional     | Signature class or `"a, b -> c"` string. Custom types are rendered into the baseline's signature string by name.                                |
| `tools`               | `None`                   | Keyword only. Named callables or `dspy.Tool` instances. Providing any makes the baseline a `dspy.RLM` and tells the proposer they are in scope. |
| `interpreter_factory` | `dspy.PythonInterpreter` | Keyword only. Zero-argument callable returning a fresh `CodeInterpreter`. A bare instance is rejected.                                          |
| `max_predictor_calls` | `100`                    | Keyword only. Cap on bridged predictor calls per `forward`. `None` removes it.                                                                  |

Other surface:

- `module_src` is a read-only property holding the current implementation. GEPA reads it as the seed
  and overwrites it with each accepted candidate.
- `forward(**inputs)` accepts keyword inputs only, and says so: passing a positional raises
  `TypeError: dspy.Flex accepts keyword inputs only`.
- `dump_state()` returns exactly `['lm', 'module_src']`. The internal predictors are not saved
  because they are derived from the code and rebuilt on every `forward`.
- `reset()` clears the LM and keeps `module_src`, the same way resetting a `Predict` keeps its tuned
  instructions.
- The class carries `@experimental(version="3.3.0")`. Treat the API and the serialization format as
  subject to change between minor releases, and pin the version if you depend on it.

## The Sandbox and the Bridge

Model-written code is untrusted code, and `Flex` never runs it in the host process. Every `forward`
creates a fresh interpreter, injects a shim that fakes a tiny `dspy` module, executes `module_src`,
drives its `forward`, and shuts the interpreter down. Exactly three things cross the boundary, as
JSON:

- `__dspy_construct__`: the shim asks the host to build a real predictor and gets back a string
  handle. Only these kinds are bridgeable: `Predict`, `ChainOfThought`, `RLM`, `CodeAct`,
  `ProgramOfThought`, `ReAct`, `ReActV2`.
- `__dspy_call__`: the shim runs a predictor by handle; the host makes the real LM call and returns
  the prediction's fields.
- the tools you passed to `dspy.Flex(tools=...)`, registered by name, with the callables staying on
  the host.

Per-forward state — constructed predictors, the call budget, custom-type originals — lives in a
`_Invocation` created for that forward and never on the shared `Flex`, so threaded evaluation stays
isolated. The budget is enforced on every bridged call, not estimated:

```python
self._calls += 1
budget = self._runtime._max_predictor_calls
if budget is not None and self._calls > budget:
    raise CodeInterpreterError(
        f"Sandboxed dspy.Flex forward exceeded its predictor-call budget "
        f"({budget}). Raise max_predictor_calls if this is expected."
    )
```

Declared output types are enforced on the way out, so a `Flex` returns the same types as a `Predict`
over the same signature and stays substitutable for one. A field declared `int`, `list[str]` or a
pydantic model would otherwise arrive as a bare string or dict, because everything crosses back as
JSON. A candidate whose output is the wrong shape, or is missing a required field entirely, raises a
`CodeInterpreterError` naming the field, which GEPA scores at the failure score for that example.

The default interpreter is `dspy.PythonInterpreter`, which is Deno and Pyodide, **so running a
`Flex` requires Deno on the machine**. It is not installed in this repo's environment
(`shutil.which('deno')` is `None`), which is the first thing to fix before trying any of this here.
`dspy.LocalInterpreter` is the documented alternative and is a weaker boundary: it separates process
memory and stdout but keeps the host user's filesystem, environment, credentials, network and
process authority.

## Penalizing LM Calls With `program_trace`

A GEPA metric returns a score plus feedback. With a `Flex` in the program it can also see how the
answer was produced, by declaring a sixth parameter:

```python
LLM_CALL_PENALTY = 0.15

def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
    correct = getattr(pred, "total_cents", None) == gold.total_cents
    n_calls = len(program_trace) if program_trace else 0
    score = max(0.0, (1.0 if correct else 0.0) - LLM_CALL_PENALTY * n_calls)
    fb = f"{'Correct' if correct else 'Wrong'} - used {n_calls} LM call(s). Settle clear cases in Python."
    return dspy.Prediction(score=score, feedback=fb)
```

The trace is opt-in strictly by declaration. `_metric_scoring_kwargs` inspects the metric's parameter
names and passes only what it finds:

```python
declared = inspect.signature(metric_fn).parameters.keys()
values = {"trace": None, "pred_name": None, "pred_trace": None, "program_trace": program_trace}
return {name: value for name, value in values.items() if name in declared}
```

So a five-argument metric keeps working untouched, and `trace` stays `None` either way, preserving
the eval-mode semantics of non-Flex GEPA scoring. Keep the penalty small relative to correctness: a
decomposition has to hold accuracy to win, and past λ=1.0 a call can never pay for itself, which
means never calling the model.

There is a compliance reading of the same dial, which the Kasparian write-up makes and the cmpnd post
does not: every LM call is a potential data transfer to a third-party API, so penalizing calls
mechanically reduces how much source data leaves the environment. For a project under data-residency
or minimization constraints, λ is a design parameter and not only a cost parameter.

## What The Generated Code May Use

The optimizer-authored code does not run against the real `dspy` package. Inside the sandbox `dspy`
is a shim whose only job is to hand predictor construction and predictor calls back to the host, and
its surface is narrower than the library's.

Available: `dspy.Module` as the base class; `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`,
`dspy.ReActV2`, `dspy.RLM` constructed in the sandbox and built on the host;
`dspy.Signature("in -> out", "instructions")` in the **string form only**, as a marker the host turns
back into a real signature, with no `with_instructions()` and no `InputField`/`OutputField` class
form; `dspy.Prediction(**fields)`; `dspy.Tool(func)` as a pass-through; and the Python standard
library imported _inside_ `forward`.

Not available: adapters, so the generated code never sees a prompt or a raw completion;
`dspy.settings`, `dspy.context`, `dspy.configure`, `dspy.LM`, so it cannot select or reconfigure a
model; `dspy.Example`, `dspy.Evaluate`, the optimizers, retrievers, and `dspy.Flex` itself, so no
nesting; class-based signatures and typed field declarations; and host objects generally, since
values cross as JSON. A helper the code defines inside `forward` can be called directly but cannot
be handed to a bridged sub-predictor.

Anything outside that surface fails when the candidate runs and is scored at the failure score, so a
missing name costs a search step. The shim is `dspy/predict/flex/_sandbox_shim.py` and the
proposer-facing catalog is `dspy/predict/flex/primitives_doc.py`.

## The Four Moves

Across many tasks, cmpnd report GEPA making the same four moves on Flex programs. They are a useful
checklist for reading a generated program, and for deciding whether Flex is worth reaching for at
all.

1. **Decomposition.** Noticing the task has steps — parse, normalize, compare, decide — and giving
   each one its own implementation.
2. **Method selection.** Choosing deterministic code or a model call per step, and picking the right
   primitive when it is a call.
3. **Routing.** Recognizing that different inputs are different tasks: clear cases down the cheap
   path, ambiguous ones to a judge.
4. **Evolution.** Once the structure settles, refining what is inside it — the sub-signatures, their
   instructions, and the code.

Hand-written harnesses make these moves too. What they do not do is make them again when the model,
the data or the tactics change.

## How This Would Apply To SERF

Concrete, and checked against this repo rather than assumed.

- **The typed signatures survive the round trip.** `dspy.Flex(AbtBuyBlockMatch)` builds a baseline
  whose signature string is
  `abt_records: list[AbtProduct], buy_records: list[BuyProduct] -> candidates: list[AbtBuyCandidate]`,
  and a populated `AbtBuyCandidate` serializes to JSON, which is what the bridge requires of an
  output field. Nothing about SERF's nested pydantic contract blocks Flex.
- **`serf train` already produces the metric Flex needs.** `make_dataset_metric` in
  `serf.dspy.train` returns F1 with feedback that enumerates the missed and invented pairs and prints
  the records behind them. Adding `program_trace=None` to its parameter list is the entire change
  needed to make it cost-aware, and `_metric_scoring_kwargs` would start passing the trace on the
  strength of that declaration alone.
- **The baseline source would be 9,439 characters for `abt-buy`,** because the whole 8,922-character
  docstring is embedded in the generated `dspy.Signature(...)` literal. The code proposer has to emit
  a complete replacement class every round, so the instruction size the repo has accumulated becomes
  the proposer's output burden. This is the practical reason to try Flex on the shorter signatures
  (`dblp-scholar` at 5,963 characters) before the longest.
- **The rules most worth moving into Python are already written down.** The `abt-buy` docstring
  defines a model code as "a run of four or more characters containing at least one digit" and asks
  for containment after stripping punctuation; `dblp-acm` requires exact year equality and prefix
  matching on titles; `walmart-amazon` wants normalized `modelno` equality ignoring dashes and
  spaces. Each is a function, described in prose, executed by an LM on every block.
- **Two SERF findings predict what Flex would buy.** `walmart-amazon` is the task where a
  string-similarity baseline beats the best deep model (71.9% against 66.9% F1) and where `modelno`
  agrees on 67.8% of gold pairs against 0.26% of near misses — a task decided by an identifier, where
  deterministic routing should be nearly free. And the repo already has evidence that longer prompts
  are not better: four successive attempts to spell out the `walmart-amazon` profiling scored 0.800,
  0.851, 0.870 and 0.855 F1 against 0.892 for the short version. Flex's second lever is exactly what
  is missing when the first one has already been pulled past its useful range.
- **Deno is the blocker.** `dspy.PythonInterpreter` needs it and it is absent here, so the first step
  is installing it or supplying a custom `interpreter_factory` and accepting that trust boundary.

Nothing in SERF uses `dspy.Flex` today. `serf train` optimizes instructions on a `dspy.Predict` over
the per-dataset signature, which is the prompt-only column of the table above — the one that bought
accuracy and paid for it on every record.

## Caveats

- **Experimental.** `@experimental(version="3.3.0")`, and the write-ups are days old relative to the
  release. The serialization format is explicitly subject to change.
- **One benchmark, one task.** Both blog posts report the same 240 held-out records. The SWE-bench
  Pro result is 12 issues and is labeled a pilot by its own authors.
- **The reflection model has to be strong.** Same constraint as prompt-only GEPA, and stronger here:
  the reflection model is writing code that has to parse, run in a restricted interpreter, and return
  the declared types. The published runs use `claude-opus-5`.
- **Cheap execution model is part of the result.** The headline numbers pair Haiku, the weakest and
  cheapest Claude, with Opus rewriting it. That asymmetry is what makes the trade interesting, and it
  is also what makes the numbers hard to transfer to a setup whose execution model is already strong.

## Sources

- [`dspy.Flex` API reference](https://dspy.ai/api/modules/Flex/)
- [Flex: optimizable module code](https://dspy.ai/diving-deeper/flex/)
- [Introducing Flex: Let the Model Write the Code](https://www.cmpnd.ai/blog/let-the-model-write-the-code.html),
  Michael Isaac, cmpnd, 5 August 2026 - the announcement, the location conflation benchmark, and the
  λ sweep
- [DSPy Flex: Let the AI write your agent's code](https://pierrekasparian.com/en/blog/article/dspy-flex-llm-program-optimization),
  Pierre Kasparian, 7 August 2026 - an independent walkthrough of the same results, plus the
  data-minimization reading of the call penalty
- The installed package: `dspy/predict/flex/{flex,bridge,ctx,primitives_doc,_sandbox_shim}.py` and
  `dspy/teleprompt/gepa/gepa_flex_utils.py` at dspy 3.3.1
