# A record budget is not an example budget

The split budgets are stated in records: 1,000 train, 200 validation, 1,000
holdout, which is the 80/20 train-to-validation ratio asked for. GEPA does not
see records. It sees **examples**, and one example is one block that holds both
sources and at least one gold pair. This measures the conversion, because the
two numbers are an order of magnitude apart and only the second one decides
whether a run can tell its candidates apart.

## Measured yield

Configured budgets, seed 42, `er.blocking.target_block_size = 30`. Script:
`/opt/cursor/artifacts/example_yield.py`.

| dataset        | train rec | train ex | val rec | val ex | hold rec | hold ex |
| -------------- | --------: | -------: | ------: | -----: | -------: | ------: |
| dblp-acm       |      1001 |       33 |     201 |      6 |     1000 |      34 |
| abt-buy        |       971 |       32 |     201 |      6 |     1001 |      33 |
| amazon-google  |      1000 |       33 |     200 |      6 |     1001 |      32 |
| walmart-amazon |      1000 |       21 |     200 |      5 |     1000 |      29 |
| dblp-scholar   |      1000 |       17 |     200 |      4 |     1000 |      21 |

Two conversions are happening, and both lose an order of magnitude:

1. **Records to blocks**, at roughly the target block size. 1,000 records
   become about 33 blocks; 200 records become 6.
2. **Blocks to usable examples.** A block is dropped when it holds records from
   only one source, or when no gold pair falls inside it, because the best
   possible answer on such a block is the empty partition and every candidate
   prompt already returns it. This is where DBLP-Scholar loses half its blocks:
   34 blocks become 17 examples, since its gold pairs are sparse across 66,879
   records.

So **the 80/20 ratio holds in records and survives into examples** — 33 against
6 is 85/15 — but the absolute size of the validation set is 4 to 6 examples on
every dataset.

## Why that is a problem, measured

A `serf train` smoke run on DBLP-ACM (400/150/150 records, `auto=light`)
saturated its validation set and then spent the rest of its budget doing
nothing:

```
Iteration  1: Base program full valset score: 0.9928918817807707
Iteration 46: New valset pareto front scores: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
Iteration 47: All subsample scores perfect for parent 7. Skipping.
Iteration 48: All subsample scores perfect for parent 7. Skipping.
```

The base prompt already scores 0.9929 on five examples. Once any candidate
reaches 1.0 on all of them, the Pareto front is full, every subsample is
perfect, and reflective mutation stops proposing anything. GEPA cannot order
candidates it cannot separate.

This is the same failure as the first GEPA run recorded in
`gepa-dblp-acm-first-run.md`, in mirror image. That run had one validation
example with no gold pair in it, so **every** candidate scored 0.0 and GEPA
returned the unoptimized prompt as the winner. This one has five examples on
the easiest dataset, so every candidate scores 1.0. In both cases the metric
cannot discriminate and the selection is arbitrary; only the sign differs.
Rule 4 of `.cursor/rules/ml-experiments.mdc` exists because of the first one,
and it catches this one too: _prove the metric can discriminate before trusting
a run_.

## What to do about it

The budget change is not obviously wrong — it is what was asked for, and a
small valset is what DSPy itself recommends. The problem is specific: **the
valset is small and the task is easy at the same time.** Options, with the
numbers behind them:

- **Do not optimise DBLP-ACM.** It sits at 99.26 F1 against a 99.32 published
  ceiling (`state-of-the-art.md`) and its own literature calls it the easiest
  of the five. A valset drawn from it will saturate at almost any size, and a
  gain measured on it would be noise. This is the cheapest fix and costs
  nothing real.
- **Optimise Amazon-Google first.** Best published figure 85.21, SERF at 83.90,
  and its six validation examples score around 0.84 rather than 1.0, so they
  can still order candidates. Same for Abt-Buy at 93.18 against 96.40.
- **Raise the validation budget only if a valset still saturates.** Yield is
  about one example per 30 records, so reaching DSPy's own 35-example threshold
  needs roughly 1,000 validation records. That abandons the 80/20 ratio, and it
  is only worth doing for a dataset whose smaller valset is demonstrably
  saturated — not pre-emptively.
- **Or cut `er.blocking.target_block_size` for training only.** At 30 it yields
  6 examples from 200 records; at 10 it would yield roughly 20. But block size
  is also a matching-quality parameter, and changing it for training alone
  means optimising a prompt against a block shape that does not ship. Not
  recommended without measuring that first.

## The check to run before each optimisation

Two lines from a GEPA log answer whether the run could have learned anything:

```
Iteration 1: Base program full valset score: <x>
Iteration N: New valset pareto front scores: {...}
```

If the base score is already near 1.0, or the Pareto front is all 1.0 while
iterations report "All subsample scores perfect ... Skipping", the run is
spending budget without selecting. Stop it and pick a harder dataset or a
larger valset rather than reading its output as a result.
