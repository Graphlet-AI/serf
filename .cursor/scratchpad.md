# SERF Build-Mode Scratchpad

## Background and Motivation

Planning is done (docs/SERF_LONG_SHOT_PLAN.md, merged to main via PR #14). The user has
now directed a move to **build mode**: implement Karpathy's research loop
(docs/RESEARCH_LOOP.md), use `dspy.GEPA` to prompt-optimize the ER signatures, and iterate
until SERF's numbers beat its own current baseline / the state of the art, reported
honestly with protocol labels. Work autonomously; don't consult the user unless
absolutely necessary. Hard cap: **$100 total in GCP/Gemini credits** for this effort
(explicit user instruction takes precedence over RESEARCH_LOOP.md's "$100/day" framing).

**Locked-in model policy (explicit user correction, overrides any other model names
mentioned in RESEARCH_LOOP.md/MISSION.md, which were written by a separate research
session before this instruction):**

- Task / student LM (all ER pipeline ops): `gemini/gemini-3.5-flash-lite`
- Reflection LM (GEPA, validation data): `gemini/gemini-3.7-flash`
- Secondary / "paper" model: `gpt-oss-120b-maas` (Vertex AI MaaS)

Do not use `gemini-3.8-flash`, `gemini-3.1-flash-lite`, or `gpt-oss-20b` anywhere.

## Key Challenges and Analysis

`docs/ID_INVARIANTS.md` §8 audits the current `src/serf/match/uuid_mapper.py` /
`matcher.py` against the Abzu reference semantics and finds real correctness bugs
(D1-D8). D1-D3 can silently lose data starting at iteration 2. `RESEARCH_LOOP.md` also
flags `max_tokens=8192` hardcoded in `matcher.py`, which caused real truncation/record
loss in a prior baseline run. GEPA-optimizing prompts on top of a leaky pipeline
produces untrustworthy numbers, so correctness comes first (Stage 1 of the research
loop), before any optimization (Stage 3).

## High-Level Task Breakdown

1. **Correctness first (Stage 1).** Write the 9 required tests from ID_INVARIANTS.md
   §9 (should fail against current code per D1-D5), fix D1-D8 + the max_tokens issue,
   confirm tests pass. Success: all 9 invariant tests pass, full suite still green.
2. **Experiment infrastructure.** `experiments/log.md` (append-only, RESEARCH_LOOP.md
   §4 format), a real budget/cost-tracking ledger (`serf.dspy.budget`, currently
   undocumented in code — check what exists), LM response caching so re-runs are free.
   Success: a budget guard exists, is wired into matcher/optimizer, and caching is on.
3. **Baselines (Stage 1 cont'd).** One command per dataset producing: random,
   exact-string-match, Jaccard/TF-IDF, blocking-recall-ceiling, pairwise-LLM. Success:
   `serf benchmark --dataset X --baseline all` (or equivalent) prints all five plus the
   current block-level SERF number, for DBLP-ACM first.
4. **GEPA optimization (Stage 3).** Build train/val splits from benchmark ground truth
   (test split sealed), a metric with textual feedback, run `dspy.GEPA` on `BlockMatch`
   against Gemini 3.5 Flash-Lite with Gemini 3.7 Flash as `reflection_lm`. Success:
   optimized program improves F1 over hand-written baseline on held-out validation,
   logged as an experiment with cost.
5. **Report.** Update experiment log + a summary of what was achieved vs. baseline vs.
   published numbers (with protocol labels, per RESEARCH_LOOP.md's honesty rules). Open
   a PR.

## Project Status Board

- [ ] Task 1: Fix identifier-conservation correctness bugs (D1-D8) + max_tokens
- [ ] Task 2: Experiment log + budget ledger + caching
- [ ] Task 3: Baseline suite (DBLP-ACM first, then Walmart-Amazon, DBLP-Scholar)
- [ ] Task 4: GEPA optimization loop
- [ ] Task 5: Final report + PR

## Executor's Feedback or Assistance Requests

(none yet)

## Lessons

- Model policy is fixed by explicit user instruction: gemini-3.5-flash-lite (task),
  gemini-3.7-flash (reflection), gpt-oss-120b-maas (secondary). Ignore other model
  names in RESEARCH_LOOP.md/MISSION.md (written before this instruction, by a
  different session).
- `main` already has PR #14 (long-shot plan) merged. This branch is `main` +
  MISSION/CODING_STANDARDS/ID_INVARIANTS/RESEARCH_LOOP.md copied from the still-open
  docs PR #19 (not merged, so not touching its other changes).
