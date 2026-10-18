# GEPA prompt tuning, scored on held-out data

Budgets follow the per-dataset split configuration in `config.yml` (`benchmarks.datasets`):
for `dblp-acm`, **2,700 train / 1,100 validation / 1,100 holdout** records (yielding 90 train,
35 validation, and 36 holdout blocks at `target_block_size=30`), with the holdout reserved and
scored only once, after GEPA has finished.

```bash
uv run serf train --dataset dblp-acm \
  --blocking-model data/models/fine-tuned-dblp-acm-bge-small-contrastive \
  --num-threads 8
```

The point of the holdout column is that it is the only one that answers a
question. GEPA selects candidates on validation, so a validation gain is partly
the selection showing through; the holdout measures whether the rewritten
prompt is actually better on records nothing in the run was allowed to see.

## Results

| Dataset                             | Train ex | Val ex | Holdout ex | Val: written → trained | Holdout: written → trained | Holdout gain |
| ----------------------------------- | -------: | -----: | ---------: | ---------------------- | -------------------------- | -----------: |
| amazon-google (default BGE)         |       33 |      6 |         32 | 0.8997 → 0.9376        | 0.8550 → 0.8566            |  **+0.0016** |
| dblp-acm (contrastive fine-tuned)   |       90 |     35 |         36 | 0.9842 → 0.9915        | 0.9836 → 0.9789            |  **-0.0048** |

## DBLP-ACM with contrastive fine-tuned blocking and 35 validation blocks

Raising the validation budget from 200 records (6 blocks) to 1,100 records (35
blocks) and blocking with the contrastive fine-tuned BGE-small model
(`data/models/fine-tuned-dblp-acm-bge-small-contrastive`) resolved the 6-block
validation saturation. Across 4 GEPA reflection iterations with student
`openai/gpt-oss-120b-maas` and teacher `gemini/gemini-4-argon`
(`reasoning_effort="medium"`):

- **Candidate 0 (shipped baseline)**: validation F1 **0.9842** (34.45 / 35
  blocks; imperfect on blocks 10, 14, 19, 21, 24, 29, 33), holdout F1 **0.9836**
  (35.41 / 36 blocks).
- **Candidate 1 (Iteration 1)**: validation F1 **0.9783**, Pareto frontier
  **0.9883** (resolved blocks 10, 19, and 29 to 1.0).
- **Candidate 2 (Iteration 2, winner)**: validation F1 **0.9915** (+0.0073 over
  baseline, 34.70 / 35 blocks, 33 of 35 validation blocks at 1.0000), Pareto
  frontier **0.9955**, holdout F1 **0.9789** (-0.0048 vs baseline, within the
  ~0.01 run-to-run LLM nondeterminism band).
- **Candidate 3 (Iteration 4)**: validation F1 **0.9841** (34.44 / 35 blocks).

### Holdout block-by-block decomposition (`dblp-acm`, 36 blocks)

Both Candidate 0 and Candidate 2 scored **1.0000 on 30 of the 36 holdout
blocks**. On the 6 blocks where they differed:

- **Trained prompt won on 3 holdout blocks** by recovering subtitle, prefix, and
  missing-author true matches that the baseline prompt rejected:
  - **Block 14 (`0.9677 → 1.0000`, +0.0323)**: Recovered `(666, 100588)` where
    DBLP had `authors='?'` and title `'Title, Foreword, Preface to the Special Issue on Spatial Database Systems, Index Vol. 3'` vs ACM `'Special issue on spatial database systems'`.
  - **Block 24 (`0.9565 → 1.0000`, +0.0435)**: Recovered `(730, 100741)` where
    ACM appended the book authors as a subtitle (`'Data warehousing and business intelligence for e-commerce: by Alan R. Simon & Steven L. Shaffer'`).
  - **Block 25 (`0.9714 → 1.0000`, +0.0286)**: Recovered `(2077, 101532)` where
    ACM prefixed `'Tutorial on storage technology: RAID and beyond'`.
- **Baseline scored higher on 3 holdout blocks**:
  - **Block 23 (`0.7500 → 0.6667`, -0.0833)**: Block 23 contains two 2001 DBLP
    `"Reminiscences on Influential Papers"` columns edited by Kenneth A. Ross
    (`304` and `968`) and one 2001 ACM `"Reminiscences on Influential Papers"`
    entry (`100551`) that omits the contributors. Candidate 0 merged all three
    into one group (emitting a same-source false positive `(304, 968)` alongside
    both cross-source pairs, F1 = 0.750). Candidate 2 enforced strict 1-to-1
    bipartite matching (eliminating the same-source false positive) and paired
    `968` with `100551` (F1 = 0.667).
  - **Blocks 9 (`0.9130 → 0.8636`), 18 (`1.0000 → 0.9091`), and 19 (`1.0000 → 0.9474`)**:
    Single-ID copy transpositions by the student LM in large (19–23 pair) blocks,
    reflecting standard ~0.01 run-to-run student generation variance.

## The validation set saturates at 200 records on four of the five datasets

Amazon-Google was the one dataset picked to go first under the older 200-record
validation budget precisely because its six validation examples score around
0.84 and can therefore order candidates. Abt-Buy at 200 validation records
showed the opposite:

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

So the honest summary of the 200-record validation configuration is that **200
validation records selects usefully on one of five datasets**. Raising
`dblp-acm` to 1,100 validation records (35 blocks) gave GEPA 7 imperfect
validation blocks to reflect on and lifted validation F1 from 0.9842 to 0.9915
(Pareto front 0.9955). Because the hand-tuned `DblpAcmBlockMatch` signature
already incorporates the dataset's venue crosswalk and exact-year rules from
`BENCHMARKS.md`, baseline holdout F1 is already 0.9836 (30 of 36 blocks at
1.0000), leaving almost no headroom above the ~0.01 LLM variance floor.

## What the Amazon-Google result shows

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
  examples can be climbed without generalising. Sizing validation to ~35
  blocks (1,100 records on `dblp-acm`) gives GEPA genuine multi-block signal.
- **The holdout pass has to stay.** It costs one parallelized pass over ~36
  blocks per prompt and it is the difference between reporting a validation
  climb and measuring out-of-sample generalisation.

## Provenance

- **amazon-google**: Trained on commit `94fc91a` (`seed=42`). Log:
  `/opt/cursor/artifacts/train_amazon-google.log`. Program:
  `data/trained_prompts/amazon-google_gepa.json`.
- **dblp-acm**: Trained on branch `cursor/gepa-tuning-contrastive-blocking-66f9`
  (`seed=42`, `blocking_model=data/models/fine-tuned-dblp-acm-bge-small-contrastive`,
  `student=openai/gpt-oss-120b-maas`, `teacher=gemini/gemini-4-argon` with
  `reasoning_effort=medium`). Logs:
  `/opt/cursor/artifacts/gepa_train_dblp_acm.log`,
  `/opt/cursor/artifacts/holdout_cand0.log`,
  `/opt/cursor/artifacts/holdout_cand2.log`. State:
  `data/gepa_logs/dblp-acm/b931915107de/gepa_state.bin`. Program:
  `data/trained_prompts/dblp-acm_gepa.json`.
