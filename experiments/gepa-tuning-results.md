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

(Remaining datasets in progress; this file is updated as each run finishes.)

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
