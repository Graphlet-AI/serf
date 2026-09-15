# Generic vs per-dataset DSPy signatures, at one and three ER iterations

Student `openai/gpt-oss-120b-maas` (Vertex AI MaaS), `models.max_tokens: 65536`,
blocking `target_block_size: 30` on name-only `intfloat/multilingual-e5-base` embeddings,
1,000-record match-group-aware samples at `seed 42`, concurrency 30.

- 3-iteration results: `data/benchmarks/ab_1000_iter3/`, logs `/opt/cursor/artifacts/ab3_*.log`
- 1-iteration results: `data/benchmarks/ab_fixed/`, logs `/opt/cursor/artifacts/abf_*.log`
- Everything under `data/benchmarks/ab_1000/` predates the litellm fix and is void.

## The earlier "typed is 5x better" reading was a bug, not a result

The first A/B put the generic arm on `dblp-acm` at F1 0.1955 against 0.9742 for the typed arm.
That gap was not real. 27 of the 33 generic blocks died with
`partially initialized module 'litellm' has no attribute 'completion'`, so only 6 blocks ever
reached the LLM and the rest were silently degraded to `error_recovery`. A re-run reproduced it
with a different survivor count and recall 0.1996, which is the signature of a race rather than a
measurement. With the race fixed, the same configuration scores F1 0.9077.

`EntityMatcher` built its LM and its predictor lazily inside the `asyncio.to_thread` pool, so a
dozen worker threads took DSPy's first litellm touch at once. DSPy resolves litellm through
`dspy.utils.lazy_import.require`, which returns whatever `sys.modules` already holds, and MLflow's
tracing hook runs a plain `import litellm` from inside a traced call. A thread that arrived while
that import was part-way through received a module whose spec was still `_initializing`.
`serf.dspy.lm` now imports litellm at module scope, and `EntityMatcher.warm_up()` builds the LM and
the predictor on the calling thread before the pool starts. All ten runs below report zero
occurrences.

## Three iterations, 1,000-record samples, seed 42

| Dataset | Arm | P | R | F1 | TP | FP | Predicted | Gold retained | Iters | Elapsed |
|---|---|---|---|---|---|---|---|---|---|---|
| dblp-acm | generic | 0.9242 | 0.9727 | 0.9478 | 463 | 38 | 501 | 476 | 3 | 1290s |
| dblp-acm | per-dataset | 0.9875 | 0.9958 | **0.9916** | 474 | 6 | 480 | 476 | 3 | 153s |
| dblp-scholar | generic | 0.2894 | 0.9469 | **0.4433** | 303 | 744 | 1047 | 320 | 3 | 456s |
| dblp-scholar | per-dataset | 0.2779 | 0.9563 | 0.4307 | 306 | 795 | 1101 | 320 | 3 | 114s |
| abt-buy | generic | 0.7045 | 0.8307 | 0.7624 | 422 | 177 | 599 | 508 | 3 | 1198s |
| abt-buy | per-dataset | 0.9760 | 0.8799 | **0.9255** | 447 | 11 | 458 | 508 | 3 | 139s |
| amazon-google | generic | 0.5309 | 0.5971 | 0.5621 | 206 | 182 | 388 | 345 | 3 | 713s |
| amazon-google | per-dataset | 0.6464 | 0.6783 | **0.6620** | 234 | 128 | 362 | 345 | 3 | 81s |
| walmart-amazon | generic | 0.7805 | 0.8533 | 0.8153 | 64 | 18 | 82 | 75 | 3 | 555s |
| walmart-amazon | per-dataset | 0.8256 | 0.9467 | **0.8820** | 71 | 15 | 86 | 75 | 3 | 114s |

## One iteration, same samples, litellm race already fixed

| Dataset | Arm | P | R | F1 | TP | FP | Predicted | Gold retained |
|---|---|---|---|---|---|---|---|---|
| dblp-acm | generic | 0.9645 | 0.8571 | 0.9077 | 408 | 15 | 423 | 476 |
| dblp-acm | per-dataset | 0.9956 | 0.9538 | 0.9742 | 454 | 2 | 456 | 476 |
| dblp-scholar | generic | 0.9346 | 0.6250 | 0.7491 | 200 | 14 | 214 | 320 |
| dblp-scholar | per-dataset | 1.0000 | 0.7719 | 0.8713 | 247 | 0 | 247 | 320 |
| abt-buy | generic | 0.9044 | 0.6516 | 0.7574 | 331 | 35 | 366 | 508 |
| abt-buy | per-dataset | 1.0000 | 0.7244 | 0.8402 | 368 | 0 | 368 | 508 |
| amazon-google | generic | 0.7016 | 0.3884 | 0.5000 | 134 | 57 | 191 | 345 |
| amazon-google | per-dataset | 0.9184 | 0.5217 | 0.6654 | 180 | 16 | 196 | 345 |
| walmart-amazon | generic | 0.8929 | 0.6667 | 0.7634 | 50 | 6 | 56 | 75 |
| walmart-amazon | per-dataset | 0.9839 | 0.8133 | 0.8905 | 61 | 1 | 62 | 75 |

## What three iterations bought

Recall rose in all ten pairings, by +0.042 to +0.322. Precision fell in all ten. F1 improved on
three of five datasets for the generic arm and two of five for the typed arm.

| Dataset | Arm | ΔP | ΔR | ΔF1 |
|---|---|---|---|---|
| dblp-acm | generic | -0.040 | **+0.116** | +0.040 |
| dblp-acm | per-dataset | -0.008 | **+0.042** | +0.017 |
| dblp-scholar | generic | -0.645 | **+0.322** | -0.306 |
| dblp-scholar | per-dataset | -0.722 | **+0.184** | -0.441 |
| abt-buy | generic | -0.200 | **+0.179** | +0.005 |
| abt-buy | per-dataset | -0.024 | **+0.156** | +0.085 |
| amazon-google | generic | -0.171 | **+0.209** | +0.062 |
| amazon-google | per-dataset | -0.272 | **+0.157** | -0.003 |
| walmart-amazon | generic | -0.112 | **+0.187** | +0.052 |
| walmart-amazon | per-dataset | -0.158 | **+0.133** | -0.009 |

The diagnosis that recall was capped by blocking, not by the matcher, holds: giving separated
records a second and third chance to meet recovers most of the missing pairs. On `dblp-acm` the
typed arm reaches recall 0.9958, and even the generic arm reaches 0.9727.

The cost is transitive over-merging. Each round merges connected components of predicted pairs, so
one wrong pair fuses two clusters and every cross-cluster pair it implies is then counted as
predicted. `dblp-scholar` is where this breaks down: precision collapses from 0.93 to 0.29 while
predicted pairs balloon from 214 to 1047. Google Scholar titles are truncated and noisy, so the
first round produces enough near-miss merges to start a cascade. `dblp-acm`, whose titles are clean,
shows the opposite: a 42% entity reduction in round one and almost no false cascade.

Rounds two and three do progressively less work everywhere. Entity reduction per round on the
generic arm: `dblp-acm` 42.3% / 4.0% / 5.0%, `abt-buy` 36.6% / 10.6% / 2.8%, `amazon-google`
18.6% / 6.7% / 2.8%, `dblp-scholar` 21.4% / 8.8% / 2.8%, `walmart-amazon` 5.6% / 1.1% / 0.5%.
Three rounds is where the returns have clearly flattened.

## Verdict: do typed signatures beat generic ones?

Yes, but by far less than the corrupted comparison suggested.

At one iteration the typed arm wins on all five datasets: mean F1 0.8483 against 0.7355, a 15%
relative gain. At three iterations it wins on four of five (mean F1 0.7784 against 0.7062), losing
`dblp-scholar` by 0.013 where both arms have already collapsed. The typed arm's advantage is
concentrated in precision: at one iteration it scores perfect precision on `dblp-scholar` and
`abt-buy` and above 0.98 on `dblp-acm` and `walmart-amazon`, because a typed candidate list of
record-id pairs cannot express the loose merges the generic `BlockMatch` signature emits.

So the honest summary is a consistent but moderate win of roughly 10-30% relative F1, not the 5x
that the `dblp-acm` numbers appeared to show. That 5x reading was the race.

## Blocking behaviour across rounds

`er.blocking.auto_scale_by_iteration` is on and does divide the target block size by the iteration
number (30, then 15, then 10), but at these dataset sizes it changes nothing. `FAISS_SCRIPT` caps
the IVF cell count at `sqrt(n)`:

```
nlist = max(1, n // target_block_size)
nlist = min(nlist, int(math.sqrt(n)))
```

At n = 1001 that cap binds for every target at or below 31, so targets of 30, 15 and 10 all produce
the same 31 blocks of average size 32.3, verified directly in
`/opt/cursor/artifacts/block_size_by_iteration.log`. Auto-scaling only has an effect while
`n < target_block_size^2`, which for a target of 30 means fewer than about 900 records. Blocks do
change between rounds, but because the entity count drops, not because of the scaling: on the
generic `dblp-acm` run, 33 blocks at avg 30.3, then 24 at avg 24.1, then 23 at avg 24.1. Left
unchanged; moving the cap would shift every benchmark number recorded so far.

## Sampling

Sampling is match-group aware (`serf.eval.sample`), so a gold pair is never half drawn. Retention
per 1,000-record sample, against what naive uniform sampling would have kept:

| Dataset | Records | Gold pairs | Sampled | Kept | Naive would keep |
|---|---|---|---|---|---|
| dblp-acm | 4,910 | 2,224 | 1,001 | 476 | 82 |
| dblp-scholar | 66,879 | 5,347 | 1,000 | 320 | 1 |
| abt-buy | 2,173 | 1,097 | 1,000 | 508 | 252 |
| amazon-google | 4,589 | 1,167 | 1,002 | 345 | 56 |
| walmart-amazon | 24,628 | 962 | 1,000 | 75 | 0 |

Both arms of a dataset draw the identical sample: the retained counts match exactly, and round-one
blocking is identical per dataset across arms. `walmart-amazon` retains only 75 pairs, so each pair
moves recall by 1.3 points and its numbers are the noisiest in the table.

## Caveats

- **Elapsed times are not comparable.** DSPy caches completions on disk, so a run whose prompts were
  already seen finishes in seconds. The typed arm's short times mostly reflect cache hits from
  earlier sweeps, not a faster signature.
- **Not comparable to the full-table baselines.** Full-data, one-iteration runs at
  `max_tokens=65536` scored `dblp-acm` P 0.9326 / R 0.5162 / F1 0.6645, `abt-buy`
  P 0.8990 / R 0.4786 / F1 0.6246, `amazon-google` P 0.3292 / R 0.1371 / F1 0.1936. Those are whole
  tables, not 1,000-record samples, so they face far more distractors per block.
- **`Adapter JSONAdapter failed to parse the LM response` persists** on large blocks. It cost 1 of
  33 blocks in round one of generic `dblp-acm` and up to 4 blocks in a later round. This is a
  separate, already-known issue; it is now counted and reported per run rather than silent.
- A handful of HTTP 429s appeared on the generic `abt-buy`, `amazon-google` and `walmart-amazon`
  runs (1-2 each) and were retried; every run exited 0.
- `config.yml` sets `models.embedding: "intfloat/multilingual-e5-base"`, while `CLAUDE.md` and the
  README's phase-1 blurb advertise Qwen3 embeddings. Unresolved, and left alone here because
  changing the embedding model would invalidate every blocking number above.
