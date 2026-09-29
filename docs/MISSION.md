# MISSION

**SERF is a research project whose goal is a state-of-the-art system for semantic entity resolution and canonicalization — match and merge — that operates on large groups of records at once rather than on pairs, and that proves its claims on the standard benchmarks.**

The primary deliverable is an academic paper. The secondary deliverable is a small, readable open-source codebase that other researchers actually adopt. Both matter. Neither is optional.

---

## 1. What We Claim

Entity resolution research has converged on a pairwise formulation: take two records, decide whether they match, repeat. Every headline number on Amazon-Google, Walmart-Amazon, Abt-Buy, DBLP-ACM and DBLP-Scholar is a binary classification score over a pre-labeled candidate pair set. Canonicalization was treated as a separate problem than entity resolution. That formulation is an artifact of the tools available when the benchmarks were built, and it is the wrong shape for large language models. Master data management has evolved.

SERF makes four claims:

1. **Set-level beats pairwise.** An LLM shown an entire block of records at once resolves them more accurately than the same LLM shown each pair in isolation, because it can triangulate across the block: if A matches B and B matches C, it sees that A matches C without being asked.
2. **Set-level is what makes LLM entity resolution economically possible.** One call per block of _N_ records instead of one call per `N(N-1)/2` pairs is the difference between a research prototype and production. The cost reduction is quadratic in block size.
3. **Resolution and canonicalization should happen in the same call.** Deciding that records match and deciding what the merged golden record should say are the same reasoning problem. Splitting them into two systems throws away information and doubles the cost.
4. **Prompts should be optimized, not written.** We use GEPA to evolve the instructions and DSPy Flex to move work out of the model and into deterministic code. We do not hand-tune prompts, and we report what the optimizer found.

## 2. Canonicalization, and "match and merge"

**Canonicalization** is the process of collapsing a set of records that refer to the same real-world entity into a single authoritative record — choosing the best value for each field, preserving provenance, and emitting one row where there were many. In master data management this output is called a _golden record_.

**We use the term "canonicalization" because it is correct, but we also say "match and merge" immediately because that is what people understand.** Every abstract, README, and talk should do both at some point. This is a deliberate rhetorical choice, not sloppiness.

The literature gap here is real and it is ours to take. Entity _matching_ with LLMs is a crowded field. Entity _canonicalization_ with LLMs, evaluated on a benchmark, appears to be unclaimed.

## 3. The Scaling Argument

The reason this approach scales is compounding reduction, not raw throughput.

Take a block of 100 records and emit 10. That is 10:1. Re-block the survivors and do it again: 10,000 original records have become 100. Again: 1,000,000 have become 100. Each round is cheaper than the last because the dataset is smaller. The number of LLM calls per round falls geometrically, and the number of rounds needed grows only logarithmically in the size of the input.

This is the core scaling claim of the paper and it must be measured, not asserted. See [RESEARCH_LOOP.md](RESEARCH_LOOP.md), Experiment E3.

## 4. The Hard Constraint

A model can only hold so many records in working memory before it starts dropping them. Somewhere above some block size N — which we must measure per model, the model begins silently omitting input records from its output.

Silently losing records is unacceptable in entity resolution. A resolution system that drops rows is not a resolution system.

Therefore SERF enforces a hard invariant:

> **Identifier conservation.** For every integer identifier that enters a block, that identifier must appear on the way out — either as the identifier of an output record, or inside the merge list (`source_ids`) of an output record. Nothing is silently dropped, and nothing is silently merged into a record the model did not explicitly say it belonged to.

The recovery machinery that enforces this was the hardest part of the predecessor system (Abzu) and its semantics are subtle. They are specified precisely in [ID_INVARIANTS.md](ID_INVARIANTS.md) and must be reproduced exactly. Improving the implementation is welcome; changing the semantics is not.

The relationship between block size, records lost, and recovery cost is not a bug to be hidden. It is a **primary experimental result** of the paper. We measure the loss curve, we publish it, and we show that recovery makes the system safe at block sizes where a naive system would be unsound.

## 5. How We Get to State of the Art

The loop, in Karpathy's framing: build the harness first, make the measurement cheap and trustworthy, then iterate against it until the number goes up. Details in [RESEARCH_LOOP.md](RESEARCH_LOOP.md).

1. **Harness.** Every benchmark loads into one canonical internal format, runs through one pipeline, and reports one set of metrics. Changing datasets changes nothing about the code path.
2. **Signatures.** DSPy signatures encode the rules that must always hold — the output contract, the master-record convention, the provenance requirements. These are written by hand, informed by the papers that introduced each dataset and by exploratory data analysis of the actual records. This is the core of ER / MDM product management, and it can't be completely automated.
3. **GEPA.** The optimizer evolves the instruction text against a metric on held-out data. We do not write prompts.
4. **Flex.** Once quality is there, the optimizer rewrites the module _code_ to answer the easy cases in deterministic Python and reserve the model for genuine ambiguity. The metric penalizes LLM calls, so cost falls without accuracy falling.
5. **Student-teacher.** Optimize against Gemini, then transfer to `gpt-oss-120b` and `gpt-oss-20b` on Vertex AI. The claim to test is that a well-optimized prompt lets a much cheaper open model close most of the gap.

## 6. What "state of the art" Means

It does not mean "a big number." The entity resolution literature uses at least five mutually incomparable evaluation protocols, and papers routinely quote each other's scores across them.

The bar is:

- We report **which protocol** every number was produced under, always.
- We beat the best published open model **under the same protocol**, on Amazon-Google, Walmart-Amazon, Abt-Buy, DBLP-ACM and DBLP-Scholar.
- We additionally report the **clustering / full-table** setting — pairwise F1 over the transitive closure, cluster F1, and B-cubed — because that is the setting our system actually operates in and the pairwise setting flatters everyone.
- We publish cost per thousand records alongside every F1. A score that costs 100x more than the baseline is a different result than the same score at parity.

Current SERF baseline, for reference, on its own blocked-candidate protocol with `gemini-3.8-flash`: DBLP-ACM F1 0.849, Abt-Buy F1 0.844, DBLP-Scholar F1 0.671. These are the numbers we are trying to beat first. Precision is now 0.99 on two of three datasets; recall (0.57–0.74) is the binding constraint. They are not competitive yet, and saying so plainly is part of the method.

Note that a false positive merge is more damaging than a false negative merge - we must avoid these at all costs and our metrics should work accordingly. We must be relatively conservative about merging records that do not in fact match as it is an inherently destructive act, even though we track provenance.

## 7. Non-Negotiables

- **Embeddings are for blocking only.** Every match decision goes through an LLM. No cosine similarity thresholds, ever. This is what makes the system _semantic_ rather than a reimplementation of 2021.
- **Identifier conservation holds, always.** A run that violates it is a failed run, not a run with a caveat.
- **Every number is reproducible from one command.** If a result cannot be regenerated by a CLI invocation recorded in the experiment log, it does not go in the paper.
- **The code stays small and readable.** A researcher should be able to read a SERF module top to bottom and understand it. See [CODING_STANDARDS.md](CODING_STANDARDS.md). Boilerplate, defensive layers, and speculative abstraction are how research code becomes unreadable and unadopted.
- **We do not invent numbers.** If a comparison figure cannot be traced to a specific table in a specific paper, it is marked as unverified or it is omitted.

## 8. The Paper

**Title:** _Semantic Entity Resolution and Canonicalization for Agent Memory_

The framing beyond benchmarks: agents accumulate memory, and memory accumulates duplicates. An agent that has met "Acme Corp", "Acme Corporation", and "ACME Inc." across three sessions has three memories of one entity, and every downstream retrieval is degraded by that fragmentation. Entity resolution and canonicalization are the maintenance function that keeps agent memory coherent over time — and it has to run continuously, at low cost, on records nobody labeled.

Draft lives in [paper/](../paper/). It builds to PDF with `../paper/build.sh`.

---

## Reading order for a new contributor

1. This file.
2. [CODING_STANDARDS.md](CODING_STANDARDS.md) — how we write code here, and why.
3. [ID_INVARIANTS.md](ID_INVARIANTS.md) — the identifier conservation semantics.
4. [RESEARCH_LOOP.md](RESEARCH_LOOP.md) — the experiment protocol.
5. [.cursor/scratchpad.md](../.cursor/scratchpad.md) — what is being worked on right now.
