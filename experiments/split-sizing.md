# Recommended training and validation sizes

Based on the four completed GEPA runs and the example-yield measurement. The
short version: **size the splits in examples, not records, and move records
from train into validation.**

## The evidence: most of the validation gain did not transfer

| Dataset        | Val examples | Val gain | Holdout gain | Transferred |
| -------------- | -----------: | -------: | -----------: | ----------: |
| dblp-scholar   |            4 |  +0.3511 |      +0.3277 |     **93%** |
| walmart-amazon |            5 |  +0.2000 |      +0.0510 |         26% |
| dblp-acm       |            6 |  +0.0152 |      +0.0015 |         10% |
| amazon-google  |            6 |  +0.0379 |      +0.0016 |          4% |

On three of four datasets, **under 26% of what GEPA gained on validation
survived to the holdout**. That is a valset being fitted rather than selected
on, and with four to six examples it could hardly be otherwise: a Pareto front
over six points can be climbed by getting six blocks right.

DBLP-Scholar is the exception that confirms it. Its gain transferred at 93%
because the gain was enormous — its shipped prompt scored 0.5805 on the holdout,
so there was real headroom and the improvement swamped the selection noise.
Headroom, not valset size, is what predicted whether a run was worth making;
valset size predicted whether the _measured_ gain could be believed.

## A record budget is not an example budget

Only blocks holding both sources and at least one gold pair are scoreable.
Measured yield per 1,000 records:

| Dataset        | Examples per 1,000 records |
| -------------- | -------------------------: |
| dblp-acm       |                         33 |
| amazon-google  |                         33 |
| abt-buy        |                         32 |
| walmart-amazon |                         21 |
| dblp-scholar   |                         17 |

So 200 validation records bought 4 to 6 examples, and the same 200 records buy
different amounts on different datasets. Any budget stated in records is
implicitly a different experiment per dataset.

## Recommendation

**Validation: target 35 examples.** That is DSPy's own threshold — at or below
it GEPA already evaluates the whole valset every step, so it is the point where
more examples stop buying exploration and start only buying resolution. Going
from 6 to 35 takes the resolution of the aggregate from 17% steps to 3%.

**Holdout: target 35 examples too**, so the number that decides whether a run
worked has the same resolution as the one that selected it.

**Train: whatever is left, up to about 150 examples.** GEPA's light preset runs
roughly 50 iterations and draws `reflection_minibatch_size` (3) examples per
iteration, so a run consumes about 150 example-draws. Beyond ~150 examples most
are provably never seen. DSPy's "as large a trainset as possible" is right in
principle and stops paying here.

Converted to records using each dataset's measured yield, and capped by table
size:

| Dataset        | Train rec | Val rec | Holdout rec | Train ex | Val ex | Holdout ex |
| -------------- | --------: | ------: | ----------: | -------: | -----: | ---------: |
| dblp-acm       |     2,700 |   1,100 |       1,100 |       91 |     37 |         37 |
| amazon-google  |     2,400 |   1,100 |       1,100 |       61 |     34 |         33 |
| walmart-amazon |     6,000 |   1,700 |       1,700 |      141 |     42 |         39 |
| dblp-scholar   |     6,000 |   2,100 |       2,100 |      106 |     41 |         39 |
| abt-buy        |       673 |     750 |         750 |       22 |     26 |         25 |

Those example counts are measured, not projected: the table is the output of
`/opt/cursor/artifacts/example_yield_new.py` against the committed config.

**Abt-Buy cannot have this.** Its whole table is 2,173 records, about 70 usable
examples, so three splits get roughly 23 each however they are cut. It is
allocated 26 validation and 25 holdout in preference to train, because a
believable measurement is worth more there than a slightly richer trainset. Its
GEPA run is the least trustworthy of the five for reasons no budget can fix.

## This supersedes the 80/20 train-to-validation ratio

80/20 was a reasonable prior and the measurement contradicts it. In records the
new split is closer to 55/22/22, and in examples about 56/23/23 on the dense
datasets. Two findings drive the change: validation gains largely failed to
transfer, which says the valset was the binding constraint; and train examples
beyond roughly 150 are never drawn, which says the records 80/20 was assigning
to train were doing nothing. Moving them to validation costs almost nothing and
buys a number that can be believed.

If you want the ratio back, the honest version is 80/20 **in examples above the
35-example validation floor** — pay validation and holdout first, then let
train take the rest. That is what these numbers do.

## What it costs

GEPA's trial count comes from the candidate count alone; valset size enters
only as linear full-evaluation terms, measured at roughly 4 rollouts per
validation example. Going from 6 to 35 examples adds about 116 rollouts to a
light run's 460, a 25% increase. Cloud spend on this project is about a tenth
of agent spend (`.cursor/rules/cost.mdc`), so this is not a trade worth
thinking about for long.

## Also changed: the teacher is Gemini 3.8 Flash

`models.teacher` moves from `gemini/gemini-3.5-flash-lite` to
`gemini/gemini-3.8-flash`. Probed live before committing, because three
plausible identifiers 404: `gemini-3.8-flash-latest`, `gemini-3.8-flash-001`
and `gemini-3.8-flash-preview` are all rejected, and only
`gemini/gemini-3.8-flash` resolves.

It is a **thinking model**, which matters for a reflection LM. A four-word
instruction rewrite cost 382 to 423 reasoning tokens, and at `max_tokens=64` it
returned **empty content with no error**. `models.max_tokens` is 65536 so there
is ample room, but the failure mode is worth knowing: a teacher that silently
returns nothing is indistinguishable, in a GEPA log, from a teacher that
declined to propose a candidate.

## What to run next

Retrain on these budgets and compare holdout gains against the four recorded in
`gepa-tuning-results.md`. The specific thing to watch is the transfer rate: if
35 validation examples is enough, Walmart-Amazon's 26% and Amazon-Google's 4%
should both rise, and Amazon-Google's near-zero holdout gain should either
become real or stay near zero for a reason other than selection noise.

Abt-Buy also needs its `gepa` library crash resolved before it can be retrained
at all; it died in `_evaluate_programs_on_valset` with an `IndexError`.
