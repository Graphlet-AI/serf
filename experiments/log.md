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

## GEPA-2026-09-07-001 — GEPA optimization setup, dblp-acm

See below; entry added once the optimization run completes.
