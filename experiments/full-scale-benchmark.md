# Full-table benchmark: is SERF state of the art?

Every number this project had published was a 1,000-record sample. This scores
the **complete tables**, so the matcher has to reject every non-matching record
a real run faces rather than the few a match-group sample leaves behind.

```bash
serf benchmark --dataset <ds> --signature-mode per-dataset \
  --eval-split all --max-iterations 3
```

Shipped per-dataset prompts, untrained. `gpt-oss-120b` matching via Vertex AI
MaaS, `bge-small-en-v1.5` blocking, three ER iterations, partition contract.
Logs: `/opt/cursor/artifacts/full_<dataset>.log`.

## Results

| Dataset        | Records | Gold pairs | Precision | Recall |     F1 |   TP |  FP | Records conserved |
| -------------- | ------: | ---------: | --------: | -----: | -----: | ---: | --: | ----------------: |
| dblp-acm       |   4,910 |      2,224 |    0.9712 | 0.9861 | 0.9786 | 2193 |  65 |         100.0000% |
| abt-buy        |   2,173 |      1,097 |    0.9749 | 0.8861 | 0.9284 |  972 |  25 |         100.0000% |
| walmart-amazon |  24,628 |        962 |    0.7098 | 0.8441 | 0.7711 |  812 | 332 |         100.0000% |
| amazon-google  |   4,589 |      1,300 |    0.5611 | 0.7592 | 0.6453 |  886 | 693 |         100.0000% |
| dblp-scholar   |  66,879 |      5,347 |         — |      — |      — |    — |   — |           running |

DBLP-Scholar is still running and is the one gap. It is the largest table by a
factor of three, 2,268 blocks on the first iteration and 4,143 on the second,
and the Vertex endpoint throttles at concurrency 20, so it is taking hours per
iteration. Its first iteration predicted **10,646 pairs against 5,347 gold**
and reduced 66,879 entities to 62,125, which is roughly twice as many pairs as
there are true ones — the same over-merging the product tasks show, on the
dataset whose sampled score was the second highest of the five. The interim
signal therefore points the same way as the rest of the table rather than
against it.

## Answering the question: no, not yet

| Dataset        | SERF full table | SC-Block end-to-end | Best pair classification, full split |                                         Verdict |
| -------------- | --------------: | ------------------: | -----------------------------------: | ----------------------------------------------: |
| abt-buy        |       **92.84** |                92.9 |                                95.15 | level with the only published end-to-end result |
| dblp-acm       |       **97.86** |       not published |                                99.32 |                 1.5 behind, on a saturated task |
| walmart-amazon |       **77.11** |                86.0 |                                91.62 |                                      8.9 behind |
| amazon-google  |       **64.53** |                80.3 |                                81.69 |                                     15.8 behind |

SERF is **competitive on the bibliographic task and on Abt-Buy, and clearly
behind on the two product tasks**. Abt-Buy is the one genuine like-for-like
comparison available — SC-Block is the only published end-to-end pipeline on
these datasets — and 92.84 against 92.9 is a tie. DBLP-ACM at 97.86 is 1.5
below a ceiling that six years of methods have crowded into a single point.

Amazon-Google and Walmart-Amazon are the problem, and it is a **precision**
problem in both: 0.5611 and 0.7098 against recall of 0.7592 and 0.8441. The
matcher is finding the true pairs and then adding a lot of wrong ones — 693
false positives against 886 true ones on Amazon-Google.

## The sample was flattering, and by how much depends on the task

| Dataset        | Sample F1 | Full-table F1 |   Delta | Sample as share of table |
| -------------- | --------: | ------------: | ------: | -----------------------: |
| abt-buy        |    0.9318 |        0.9284 | -0.0034 |                      46% |
| dblp-acm       |    0.9926 |        0.9786 | -0.0140 |                      20% |
| walmart-amazon |    0.9467 |        0.7711 | -0.1756 |                       4% |
| amazon-google  |    0.8390 |        0.6453 | -0.1937 |                      22% |

The `state-of-the-art.md` prediction was half right. It flagged Walmart-Amazon
and DBLP-Scholar as the suspicious rows because their samples were the smallest
fraction of their tables, and Walmart-Amazon duly lost 0.176. But Amazon-Google
lost _more_, 0.194, from a sample the same size as DBLP-ACM's, which lost
0.014. So sample fraction alone does not explain it.

What does: **sampling by match group removes the hard negatives.** A sample
drawn by pulling whole gold match groups keeps the true pairs and discards most
of the records that merely look like them. A task whose difficulty is rejecting
near-misses therefore gets much easier when sampled, and a task decided by a
sharp exact filter barely changes. Precision tells the story directly:

| Dataset        | Precision sampled | Precision full table |
| -------------- | ----------------: | -------------------: |
| dblp-acm       |            0.9958 |               0.9712 |
| abt-buy        |            0.9804 |               0.9749 |
| walmart-amazon |            0.9467 |               0.7098 |
| amazon-google  |            0.8554 |               0.5611 |

DBLP-ACM survives because equal year plus matching title is nearly decisive and
extra records give it nothing new to confuse. The product tasks collapse
because their discriminating evidence — a model code, a capacity, a version —
is exactly what a larger pool supplies more near-duplicates of.

**Every sampled figure this project has published overstates full-table
performance**, by a rounding error on Abt-Buy and DBLP-ACM and by 0.18 to 0.19
on the product tasks. The leaderboard row in the README has been corrected to
the full-table number.

## Conservation held everywhere

All four completed runs account for **100.0000%** of their input records with
nothing recovered at the run level. The block-level repairs did fire, on every
dataset:

| Dataset        | Blocks needing record recovery | Blocks with a record in two groups |
| -------------- | -----------------------------: | ---------------------------------: |
| walmart-amazon |                             18 |                                  2 |
| amazon-google  |                             16 |                                  7 |
| dblp-acm       |                              5 |                                  8 |
| abt-buy        |                              3 |                                  8 |

Every one of those is a record the model left out of every group, or claimed
twice, on a block of 50 to 90 records. Under the old pairwise contract each
would have been an absent pair nobody counted. Under the partition contract
they are holes in a cover, detected and filled, which is why the run-level
figure is 100% rather than merely close to it. No run produced an invented
record id.

## What this says about where to look

- **Precision on the product tasks is the whole gap.** Recall is 0.76 and 0.84,
  respectively; the published systems are not finding more true pairs, they are
  emitting fewer wrong ones. Prompt work aimed at rejection, not at recall, is
  where the 9 to 16 points are.
- **Blocking is not obviously the bottleneck.** Recall at 0.84 on
  Walmart-Amazon means blocking is putting most true pairs together; the
  matcher then over-merges.
- **Tune against full tables or a holdout, never a match-group sample.** A
  prompt tuned to a sampled score is being tuned on a task with the hard
  negatives removed, which is precisely the part the product datasets are
  scored on.

## Caveat on "untrained"

These are the shipped per-dataset signatures with no GEPA optimization. They
are not unfitted: their docstrings were written by hand against these
benchmarks, citing measured A/B results on samples of them. That is a weaker
form of fitting than training, and it is a real one, and it is another reason
the sampled numbers ran high.
