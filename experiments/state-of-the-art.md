# State of the art on the five benchmarks

Collected before starting GEPA optimisation, so there is a target to aim at and
a record of what the target was when we aimed at it. Every figure names a
primary source. Figures that could not be traced to one are marked unverified
rather than carried at a plausible value.

An earlier version of this file is superseded. It quoted the widely-cited
Abt-Buy 95.78 and Amazon-Google 85.21 as the figures to beat without noticing
that they are measured on a **down-sampled** test set, and it reported that no
end-to-end results exist when one paper does publish them. Both corrections are
below.

## Three different protocols wear the same dataset name

| Code        | What it means                                                                         |
| ----------- | ------------------------------------------------------------------------------------- |
| **PC-full** | pair classification on the complete standard DeepMatcher test split                   |
| **PC-down** | pair classification on a test split down-sampled to ≤250 positives + ≤1,000 negatives |
| **E2E**     | end-to-end: blocking over the raw tables, then matching                               |

The down-sampling matters and is easy to miss. The Mannheim group states it
plainly: "In order to keep the OpenAI API usage fees on an affordable level, we
down-sample all test sets to approximately 1250 entity pairs." On Abt-Buy that
discards 710 of 1,710 negatives; on Amazon-Google, 1,059 of 2,059. Fewer
negatives means fewer chances to produce a false positive at a fixed error
rate, so precision and therefore F1 are optimistically biased.

The bias is measurable rather than hypothetical. The same group's later
fine-tuning paper uses the full splits and reports GPT-4o-mini zero-shot at
87.68 on Abt-Buy, against 91.93 for its best down-sampled zero-shot prompt —
about four points. So GPT-4's famous 95.78 likely corresponds to something in
the low 90s on the full split.

## Best verified F1, by protocol

### On the complete test split — the table to compare against

| Dataset            | Best F1 | Method                                  | Trained on it | Year | Source                                                                                     |
| ------------------ | ------: | --------------------------------------- | ------------- | ---- | ------------------------------------------------------------------------------------------ |
| **DBLP-ACM**       |   99.32 | EM-Join, fine-tuned, threshold 0.85     | yes           | 2025 | [SCITEPRESS 134837](https://www.scitepress.org/publishedPapers/2025/134837/pdf/index.html) |
| **DBLP-Scholar**   |   98.51 | Jellyfish-13B, instruction-tuned        | yes           | 2024 | [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.497.pdf)                             |
| **Abt-Buy**        |   95.15 | Qwen3-8B cross-encoder (instruct), LoRA | yes           | 2026 | [arXiv:2607.24688](https://arxiv.org/abs/2607.24688)                                       |
| **Walmart-Amazon** |   91.62 | Qwen3-4B cross-encoder (base), LoRA     | yes           | 2026 | [arXiv:2607.24688](https://arxiv.org/abs/2607.24688)                                       |
| **Amazon-Google**  |   81.69 | Jellyfish-7B, instruction-tuned         | yes           | 2024 | [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.497.pdf)                             |

### On the down-sampled split — the numbers usually quoted as SOTA

| Dataset        | Best F1 | Method                  | Trained on it | Source                                               |
| -------------- | ------: | ----------------------- | ------------- | ---------------------------------------------------- |
| DBLP-ACM       |   99.60 | Llama-3.1-70B, LoRA     | yes           | [arXiv:2310.11244](https://arxiv.org/abs/2310.11244) |
| Abt-Buy        |   95.78 | GPT-4 (0613), zero-shot | no            | [arXiv:2310.11244](https://arxiv.org/abs/2310.11244) |
| Walmart-Amazon |   92.99 | GPT-4o-mini, fine-tuned | yes           | [arXiv:2310.11244](https://arxiv.org/abs/2310.11244) |
| Amazon-Google  |   85.21 | GPT-4 (0613), 10-shot   | in-context    | [arXiv:2310.11244](https://arxiv.org/abs/2310.11244) |

Do not mix the two tables. AnyMatch deliberately reuses these same
down-sampled pair lists, so its figures and the Jellyfish, Ditto and Unicorn
numbers reproduced inside it carry the same bias.

### End-to-end — the only like-for-like comparison for SERF

**SC-Block** (Brinkmann, Shraga and Bizer, CIKM 2023) is the one paper found
that runs complete pipelines on these datasets: a blocker over the full
Cartesian product of the two tables, feeding a matcher, scored against the
benchmark ground truth.

| Dataset        | E2E F1 | Pipeline                     | Candidate set | Source                                               |
| -------------- | -----: | ---------------------------- | ------------- | ---------------------------------------------------- |
| Abt-Buy        |   92.9 | BM25₃ blocker + SupCon-Match | 14k pairs     | [arXiv:2303.03132](https://arxiv.org/abs/2303.03132) |
| Walmart-Amazon |   86.0 | BM25₃ blocker + Ditto        | 31k pairs     | [arXiv:2303.03132](https://arxiv.org/abs/2303.03132) |
| Amazon-Google  |   80.3 | SC-Block + SupCon-Match      | 11k pairs     | [arXiv:2303.03132](https://arxiv.org/abs/2303.03132) |

**No end-to-end figures exist for DBLP-ACM or DBLP-Scholar** in any source
checked. Not low, not withheld — simply not reported.

SC-Block's own conclusion is worth carrying: once blocking recall passes about
99.5%, "the F1 scores of the pipeline mainly depend on the matcher." That is
why its end-to-end numbers land so close to the pair-classification ones, and
it is also why an end-to-end claim is meaningless without its blocker, its
recall and its candidate-set size stated alongside.

## Where SERF currently sits

Measured on the **complete tables**, not a sample: end-to-end resolution, three
iterations, `gpt-oss-120b`, per-dataset signatures, no training, partition
contract. Full breakdown in `full-scale-benchmark.md`.

| Dataset        | SERF full table | SC-Block E2E | Best PC-full | Gap to the nearest comparable                         |
| -------------- | --------------: | -----------: | -----------: | ----------------------------------------------------- |
| DBLP-ACM       |           97.86 |            — |        99.32 | -1.5 against pair classification, on a saturated task |
| Abt-Buy        |           92.84 |         92.9 |        95.15 | level with the only published end-to-end result       |
| Walmart-Amazon |           77.11 |         86.0 |        91.62 | -8.9 against end-to-end                               |
| Amazon-Google  |           64.53 |         80.3 |        81.69 | -15.8 against end-to-end                              |
| DBLP-Scholar   |     in progress |            — |        98.51 | —                                                     |

**SERF is not state of the art.** It ties the one genuine like-for-like
comparison on Abt-Buy, sits 1.5 off a ceiling six years of work has crowded
into a single point on DBLP-ACM, and is well behind on both product tasks. The
gap is precision in both cases, not recall.

An earlier version of this section carried 1,000-record samples and read
99.26 / 96.65 / 94.67 / 93.18 / 83.90. It warned that the two rows where SERF
read highest were the two whose samples were the smallest fraction of the
table, and that was right as far as it went: Walmart-Amazon fell 0.176 on the
full table. It was also incomplete, because Amazon-Google fell further, 0.194,
from a sample the same size as DBLP-ACM's, which fell 0.014. Sample fraction is
not the mechanism. Sampling by match group is: it keeps the true pairs and
throws away most of the records that merely resemble them, so it removes the
hard negatives, and a task decided by rejecting near-misses gets much easier
while a task decided by an exact filter barely moves.

## What this says about the GEPA target

- **DBLP-ACM is done.** 99.26 against 99.32, and six years of methods span 98.4
  to 99.32 — under a point of headroom, well inside seed-to-seed variance. The
  2026 Qwen3 study simply dropped it from its suite. A gain here would be noise.
- **Amazon-Google is the one to optimise.** Hardest for everyone, SERF already
  at 83.90 against 81.69 PC-full and 80.3 end-to-end, and — the deciding factor
  — its validation examples score around 0.84 rather than 1.0, so they can
  still order candidates. See `gepa-example-yield.md`.
- **Abt-Buy is second**, on the cleanest comparison available: a two-point gap
  to 95.15 on a sample that is nearly the full table.
- **DBLP-Scholar and Walmart-Amazon need a full-table number before being
  optimised against**, because their sampled figures are the least trustworthy
  of the five and optimising against an optimistic number optimises the
  sampling.

## Caveats worth carrying

- **Two DBLP-Scholar outliers deserve suspicion.** Jellyfish-13B's 98.51 sits
  about 2.2 points above everything else on a benchmark where nothing has moved
  since 2022, DBLP-Scholar training data was in its instruction-tuning mix, and
  its own 7B and 8B siblings tuned on the same data reach only 94.88 and 95.03 —
  the result is not reproduced within its own model family. Unicorn-ins's 97.08
  is the more interesting case: genuinely zero-shot on DBLP-Scholar, held out of
  multi-task training, and still above Ditto's supervised 95.6.
- **A harsher post-blocking protocol collapses the leaderboard ordering.** ComEM
  ([arXiv:2405.16884](https://arxiv.org/abs/2405.16884)) discards the pair files
  entirely: Sparkly blocking retrieves 10 candidates per record and the model
  picks the match. Ditto falls from 86.76 to **57.75** on Walmart-Amazon. That is
  the clearest evidence anywhere that pair-classification scores overstate real
  pipeline behaviour, and it is the protocol closest to what SERF does.
- **Fine-tuned LLMs are not uniformly better than 2020-era Ditto.** GPT-4o-mini
  fine-tuned on Walmart-Amazon scores 78.85 on the full split, eight points
  below Ditto's 86.76. GPT-4o zero-shot manages 63.45 on Amazon-Google. The 2026
  factorial study finds bigger models lean harder on shortcut learning: one
  dataset's cross-encoder F1 degrades from 97.6 at 0.6B to 85.0 at 8B, driven
  by precision collapse on same-title negatives.
- **Same name, different dataset.** TATEM's 82.2 and 90.56 are on
  Amazon-Google-**Tab** and Walmart-Amazon-**Tab**, enriched variants built by
  scraping extra attributes, and are excluded here. Jellyfish's Abt-Buy test set
  is listed as 1,946 pairs against the standard 1,916, unexplained.
- **Training volume differs under identical evaluation.** Ditto and HierGAT
  train on Abt-Buy's 5,743-pair train split; R-SupCon and the Mannheim LLM
  papers report 7,659 pairs, which is train plus validation. The test split is
  identical, so the evaluation compares, but one side trains on a third more
  labelled data.
- **Papers With Code is gone.** Its boards are mirrored at
  [OpenCodePapers](https://opencodepapers-b7572d.gitlab.io/benchmarks/entity-resolution-on-abt-buy.html),
  which exposes only Abt-Buy and Amazon-Google; the other three 404. Every
  figure above was traced to a paper rather than taken from the mirror.

## Unverified, and excluded from the tables

- **RobEM on Abt-Buy, 90.90.** Appears only on the OpenCodePapers mirror; the
  CIKM 2022 paper is a paywalled four-page short paper with no preprint and the
  repository publishes no results table. RobEM's Amazon-Google 79.06 and
  Walmart-Amazon 86.68 _are_ verified via the CIKM 2023 TATEM paper; its
  DBLP-ACM and DBLP-Scholar figures are unknown.
- **GPT-4 figures tabulated inside the Jellyfish paper** (Walmart-Amazon 90.27,
  Abt-Buy 92.77, Amazon-Google 74.21, DBLP-ACM 97.44, DBLP-Scholar 91.87). Its
  authors state they follow numbers reported in previous works rather than
  running GPT-4 themselves, so the test splits behind them are unconfirmed.
- **Unicorn on Abt-Buy, Amazon-Google and DBLP-ACM.** Unicorn's own suite is
  Walmart-Amazon, DBLP-Scholar, Fodors-Zagats, iTunes-Amazon and Beer. The
  circulating 89.45 / 55.70 / 94.33 are AnyMatch's leave-one-dataset-out
  reproductions on down-sampled splits, not Unicorn's published results.
- **The 2026 Qwen3 per-dataset figures** are not printed as a clean table in the
  paper. They were recomputed from the authors' released per-run results — 1,620
  runs over 27 configurations × 9 datasets × seeds — and validated against the
  macro-averages the paper states (reproduced 83.1 / 84.9 / 85.0 against the
  paper's 83.0 / 84.1 / 84.1 at 0.6B / 4B / 8B). The Abt-Buy 95.15 and
  Amazon-Google 80.84 values are post-hoc best-of-27 selections and are
  optimistic by roughly that margin; Walmart-Amazon's 91.62 is a single fixed
  configuration averaged over ten seeds and is not. Aggregation log:
  `/opt/cursor/artifacts/beyond-scale-2026-per-dataset-f1.log`.
- **A note on what comes next.** MaDI-Bench
  ([arXiv:2606.30371](https://arxiv.org/abs/2606.30371)) is the first genuine
  end-to-end relational data-integration benchmark, but it uses its own five
  domains rather than the Magellan datasets. It is the likely successor venue
  for end-to-end comparison.
