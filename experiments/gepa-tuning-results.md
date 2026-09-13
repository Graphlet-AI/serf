# GEPA prompt tuning, scored on held-out data

Budgets are the configured defaults: **1,000 train / 200 validation / 1,000
holdout**, roughly 80/20 train to validation, with the holdout reserved and
read only once, at the end of the run, after GEPA has finished.

```bash
serf train --dataset <ds> --auto light
```

The point of the holdout column is that it is the only one that answers a
question. GEPA selects candidates on validation, so a validation gain is partly
the selection showing through; the holdout measures whether the rewritten
prompt is actually better on records nothing in the run was allowed to see.

## Results

| Dataset       | Train ex | Val ex | Holdout ex | Val: written → trained | Holdout: written → trained | Holdout gain |
| ------------- | -------: | -----: | ---------: | ---------------------- | -------------------------- | -----------: |
| amazon-google |       33 |      6 |         32 | 0.8997 → 0.9376        | 0.8550 → 0.8566            |  **+0.0016** |

Abt-Buy, Walmart-Amazon, DBLP-Scholar and DBLP-ACM are still running; this file
is updated as each finishes.

## The validation set saturates on four of the five datasets

Amazon-Google was the one dataset picked to go first precisely because its six
validation examples score around 0.84 and can therefore order candidates.
Abt-Buy, running second, shows the opposite:

```
Iteration 15: New valset pareto front scores: {0: 0.96, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.966, 5: 1.0}
Iteration 15: Best score on valset: 0.9875862068965517
Iteration 16: All subsample scores perfect for parent 3. Skipping.
```

Four of the six examples are already perfect, the aggregate is 0.9876, and GEPA
has begun skipping iterations because it cannot separate its candidates. That is
the same saturation DBLP-ACM showed in `gepa-example-yield.md`, and it is
expected on Walmart-Amazon and DBLP-Scholar too: every dataset except
Amazon-Google is easy enough that six examples run out of discriminating power.

So the honest summary of this configuration is that **200 validation records
selects usefully on one of five datasets**. The 80/20 ratio is correct in
records; the absolute count is what fails, because 200 records become 4 to 6
scoreable blocks. Reaching DSPy's own 35-example threshold needs roughly 1,000
validation records, which abandons 80/20 — a trade worth making only for the
datasets that demonstrably saturate, which is now most of them.

## What the first result shows

On Amazon-Google, GEPA improved its validation score by **+0.0379** and its
holdout score by **+0.0016**.

Without the holdout pass this run would have been reported as a 3.8-point win.
The honest reading is that it is worth approximately nothing: +0.0016 is far
inside the roughly 0.01 run-to-run variance these tasks show, so the trained
prompt is indistinguishable from the shipped one on data neither was fitted to.
Almost the entire validation gain was GEPA finding the six validation examples
rather than finding a better prompt.

This is the failure mode rule 1 of `.cursor/rules/ml-experiments.mdc` exists to
catch, and it is worth noting that the earlier trained-versus-untrained figure
this project published — DBLP-ACM at 0.9906 against 0.9737 — was a validation
comparison on both sides and is now known to be the same kind of number.

Two things follow for the setup rather than for the prompt:

- **Six validation examples is too few to select on.** 200 records yield 4 to 6
  scoreable blocks (`gepa-example-yield.md`), and a Pareto front over six
  examples can be climbed without generalising. The 80/20 ratio is right in
  records and the absolute count is the problem.
- **The holdout pass has to stay.** It costs one pass over ~32 blocks per prompt
  and it is the difference between reporting +0.0379 and +0.0016.

## Provenance

Amazon-Google was trained on commit `94fc91a`, immediately before the UUID
identity change. GEPA optimises the instruction text against the typed side
records and the block-local partition, none of which the identity change
touches, so the trained program and its scores carry over. The remaining
datasets were trained on `7e9e662` or later.

Logs: `/opt/cursor/artifacts/train_<dataset>.log`. Trained programs:
`data/trained_prompts/<dataset>_gepa.json`.
