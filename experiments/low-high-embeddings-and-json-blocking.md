# Large blocking embeddings and JSON record blocking

Two questions, measured together on the same grid:

1. Does a large embedding model buy blocking recall that `BAAI/bge-small-en-v1.5` — the 33M
   winner of [embedding-blocking-sweep.md](embedding-blocking-sweep.md) — does not already have?
2. Does embedding the whole record as JSON, with the field names inline, beat embedding the
   name alone?

Both answers are no, with one narrow exception worth keeping. Both were then asked again, and
the follow-up in the second half of this document overturns the framing of each: the large
models were chosen on the wrong MTEB category, and JSON blocking was tested as a replacement
when it belongs as an augmentation.

Protocol: `serf blocking-sweep --candidate-set large --blocking-strategy both`, 2,000-record
samples per dataset at `seed 42`, `target_block_size: 30`, one pass, CPU-only on a 4-core box.
Sampling keeps ground-truth match groups whole, so gold pairs survive; the gold-pair counts are
942 (dblp-acm), 587 (dblp-scholar), 1,011 (abt-buy), 144 (walmart-amazon) and 669
(amazon-google).

## Candidates

Selected from MTEB(eng, v2) clustering, on the reasoning that blocking clusters records,
filtered to 3B parameters and 1,024 output dimensions or fewer, read from the `mteb/results`
dataset rather than the leaderboard Space, which renders client-side. That reasoning turns out
to be wrong, and the follow-up below measures how wrong.

| Model | Params | Dim | MTEB clustering |
|---|---|---|---|
| `codefuse-ai/F2LLM-0.6B` | 596M | 1024 | 0.6036 |
| `Qwen/Qwen3-Embedding-0.6B` | 596M | 1024 | 0.5642 |
| `intfloat/multilingual-e5-large-instruct` | 560M | 1024 | 0.5180 |
| `mixedbread-ai/mxbai-embed-large-v1` | 335M | 1024 | 0.4776 |
| `BAAI/bge-large-en-v1.5` | 335M | 1024 | 0.4738 |
| `BAAI/bge-small-en-v1.5` *(the LOW default)* | 33M | 384 | 0.4702 |

Four higher-ranked candidates were dropped because they do not run here, and the reasons are
recorded next to `benchmarks.embedding_candidates_large` in `config.yml`: two Qwen2-based models
need `rope_theta` at the top level of the config, which transformers 5.16 moved into
`rope_parameters`; `stella_en_400M_v5` requires xformers; `Jasper-Token-Compression-600M`
advertises 1,024 dimensions and emits 2,048; `embeddinggemma-300m` is gated.

## Name-only blocking recall

| Model | dblp-acm | dblp-scholar | abt-buy | walmart-amazon | amazon-google | Mean | Secs |
|---|---|---|---|---|---|---|---|
| `mxbai-embed-large-v1` | 0.9904 | **0.9744** | **0.9001** | **0.9236** | 0.6323 | **0.8842** | 291 |
| **`bge-small-en-v1.5`** *(LOW)* | 0.9745 | 0.9659 | 0.8912 | 0.8750 | **0.6756** | 0.8765 | **47** |
| `multilingual-e5-large-instruct` | **0.9947** | 0.9455 | 0.8952 | 0.9028 | 0.6143 | 0.8705 | 334 |
| `bge-large-en-v1.5` *(HIGH)* | 0.9936 | 0.9489 | 0.8813 | 0.8750 | 0.6338 | 0.8665 | 324 |
| `F2LLM-0.6B` | 0.9172 | 0.8944 | 0.7636 | 0.7708 | 0.5979 | 0.7888 | 183 |
| `Qwen3-Embedding-0.6B` | 0.9299 | 0.8739 | 0.7280 | 0.7847 | 0.6099 | 0.7853 | 186 |

Only one large model beats the 33M default, and it beats it by 0.0077 for six times the CPU.
Two of the four large models finish *below* it. The two best MTEB clustering scores in the
candidate list produce the two worst blocking scores: `F2LLM-0.6B` leads clustering at 0.6036
and finishes last here at 0.7888, while `bge-small-en-v1.5` has the second-lowest clustering
score in the table and finishes second. MTEB clustering rank does not predict blocking recall,
and picking models by it would have been actively misleading.

`intfloat/multilingual-e5-large-instruct` was measured because it has worked well on this
project before. It is the strongest model on dblp-acm (0.9947) and it is 0.0060 behind the
default overall, so the earlier good experience holds for bibliographic data and does not
generalize to the product benchmarks.

## JSON record blocking

`Entity.json_for_embedding()` serializes every populated field as a JSON object with the field
names inline, stripping the `l_`/`r_` source prefixes and the id, so a record embeds as
`{"description": "...", "manufacturer": "...", "name": "...", "price": "..."}` rather than as
its name. Select it with `er.blocking.strategy: json` or `--blocking-strategy json`.

| Model | dblp-acm | dblp-scholar | abt-buy | walmart-amazon | amazon-google | Mean | Secs |
|---|---|---|---|---|---|---|---|
| `multilingual-e5-large-instruct` | 0.5552 | 0.9267 | 0.7260 | 0.7361 | **0.7429** | **0.7374** | 891 |
| `mxbai-embed-large-v1` | 0.4756 | 0.8637 | 0.8675 | 0.7500 | 0.7190 | 0.7351 | 840 |
| `F2LLM-0.6B` | 0.4575 | 0.7973 | 0.7695 | 0.5347 | 0.5620 | 0.6242 | 582 |
| `bge-large-en-v1.5` | 0.2718 | 0.6354 | 0.7092 | 0.7431 | 0.5919 | 0.5903 | 934 |
| `bge-small-en-v1.5` | 0.2070 | 0.6065 | 0.6113 | 0.7917 | 0.4753 | 0.5384 | 105 |
| `Qwen3-Embedding-0.6B` | 0.3333 | 0.6286 | 0.5193 | 0.3750 | 0.4604 | 0.4633 | 521 |

JSON loses on 27 of the 30 model-dataset cells, by between 0.1331 and 0.3381 on the mean. The
collapse is worst where the extra fields are least discriminative: dblp-acm drops by 0.46 to
0.77, because venue, year and author list are shared by thousands of papers and drown the title
that actually identifies the work. Embedding more text does not add signal when the added text
is mostly the same across records; it dilutes the part that is not.

The exception is amazon-google, and it is the dataset that most needs help. The two strongest
large models gain there — `multilingual-e5-large-instruct` goes 0.6143 to 0.7429 and
`mxbai-embed-large-v1` goes 0.6323 to 0.7190 — and 0.7429 is the highest amazon-google blocking
recall anything in this project has reached, against 0.6756 for the best name-only run. Product
names in that dataset are short, abbreviated and frequently share a manufacturer token, so the
manufacturer and price fields carry information the name does not. The weaker models do not get
this; they lose on amazon-google too. Only models with enough capacity to read structure out of
a JSON blob benefit from being given one.

So JSON blocking loses as a *replacement* for name blocking. That is the only thing this grid
tested, and it is not the only way to use it; the union below treats it as an augmentation and
the answer changes completely.

# Follow-up: the matching category, and JSON as an augmentation

Two corrections to the above, measured afterwards on the same 2,000-record samples at seed 42.

## Which MTEB category predicts blocking recall

The candidates above were picked on clustering score and the measured order did not follow it,
which leaves the better question: does *any* MTEB category order candidates the way blocking
recall does? `serf mteb-rank` answers it. It reads published scores straight from the
`mteb/results` parquet dataset, averages the tasks named under `benchmarks.mteb_tasks` for a
category, and Spearman-correlates that against measured blocking recall from a sweep file.

Run over all thirteen small-and-large candidates on one shared set of samples, so the MTEB rank
and the measurement are compared on the same footing:

| MTEB(eng, v2) category | Spearman, 13 models | Spearman, 18 models |
|---|---|---|
| **PairClassification** | **+0.5714** | **+0.5129** |
| STS | +0.0879 | +0.2322 |
| Classification | +0.0220 | +0.1208 |
| Clustering | +0.0165 | +0.0941 |
| Reranking | +0.0165 | +0.3044 |
| Retrieval | −0.0549 | +0.1662 |

The 18-model column re-runs the same correlation after adding the candidates the matching
category surfaced. PairClassification stays first and clustering stays last.

Clustering carries no information about blocking recall — 0.0165 is a rank correlation of
nothing. It would have ranked `F2LLM-0.6B` and `Qwen3-Embedding-0.6B` first and second of the
thirteen, and they measure last and second from last.

There is no MTEB task type literally named "matching", but `PairClassification` is it.
SprintDuplicateQuestions, TwitterSemEval2015 and TwitterURLCorpus all ask whether two short
texts denote the same thing, scored by average precision over cosine similarity, which is the
entity matching decision. It is the only category with signal, it ranks `F2LLM-0.6B` last, and
its top pick `mxbai-embed-large-v1` is the best large model in the grid above.

The measured recalls behind that correlation, best instruction prefix per model:

| Model | dblp-acm | dblp-scholar | abt-buy | amazon-google | walmart-amazon | Mean | Secs |
|---|---|---|---|---|---|---|---|
| `gte-base` | 0.9841 | **0.9779** | **0.9179** | **0.7100** | 0.9028 | **0.8985** | 86 |
| `mxbai-embed-large-v1` | 0.9904 | 0.9744 | 0.9001 | 0.6323 | **0.9236** | 0.8842 | 294 |
| `gte-small` | 0.9904 | 0.9370 | 0.9100 | 0.6697 | 0.9097 | 0.8834 | **44** |
| `bge-base-en-v1.5` | 0.9841 | 0.9659 | 0.8853 | 0.6487 | 0.9097 | 0.8787 | 111 |
| **`bge-small-en-v1.5`** *(LOW)* | 0.9745 | 0.9659 | 0.8912 | 0.6756 | 0.8750 | 0.8765 | 47 |
| `multilingual-e5-large-instruct` | **0.9947** | 0.9455 | 0.8952 | 0.6143 | 0.9028 | 0.8705 | 355 |
| `bge-large-en-v1.5` *(HIGH)* | 0.9936 | 0.9489 | 0.8813 | 0.6338 | 0.8750 | 0.8665 | 315 |
| `all-mpnet-base-v2` | 0.9915 | 0.9727 | 0.8853 | 0.6203 | 0.8403 | 0.8620 | 90 |
| `all-MiniLM-L6-v2` | 0.9639 | 0.9574 | 0.8408 | 0.6173 | 0.8681 | 0.8495 | 34 |
| `multilingual-e5-base` | 0.9798 | 0.9727 | 0.8724 | 0.5546 | 0.8056 | 0.8370 | 200 |
| `multilingual-e5-small` | 0.9798 | 0.9455 | 0.8417 | 0.5725 | 0.7917 | 0.8262 | 54 |
| `F2LLM-0.6B` | 0.9289 | 0.8910 | 0.7596 | 0.5859 | 0.7708 | 0.7873 | 188 |
| `Qwen3-Embedding-0.6B` | 0.9299 | 0.8705 | 0.7250 | 0.5755 | 0.7986 | 0.7799 | 207 |

These differ slightly from the grid at the top of this document because every model here was
measured on one identical set of samples in a single sweep, rather than across separate runs.

A correlation of 0.57 is worth selecting candidates on and is not worth trusting instead of
measuring. PairClassification's own top pick finishes second, the model that wins is only sixth
on it, and the ordering is not stable across record sets: `gte-small` beats the default by
0.0069 here and loses to it by 0.0046 on the full tables.

## Candidates selected on the matching category

Taking the top PairClassification scorers inside the same 3B-parameter, 1,024-dimension budget
surfaces five models the clustering ranking never did. Four of the five beat every
clustering-selected large model:

| Model | Selected on | dblp-acm | dblp-scholar | abt-buy | amazon-google | walmart-amazon | Mean | Secs |
|---|---|---|---|---|---|---|---|---|
| **`GIST-large-Embedding-v0`** | **matching** | 0.9820 | **0.9813** | 0.9149 | 0.7070 | **0.9306** | **0.9031** | 250 |
| `gte-base` | clustering-era sweep | 0.9841 | 0.9779 | **0.9179** | 0.7100 | 0.9028 | 0.8985 | **86** |
| **`b1ade-embed`** | **matching** | 0.9904 | 0.9830 | 0.8872 | **0.7130** | 0.9097 | 0.8967 | 245 |
| **`gte-modernbert-base`** | **matching** | 0.9830 | 0.9761 | 0.8912 | 0.6607 | **0.9306** | 0.8883 | 111 |
| **`UAE-Large-V1`** | **matching** | 0.9915 | 0.9710 | 0.8912 | 0.6741 | 0.9028 | 0.8861 | 248 |
| `mxbai-embed-large-v1` | clustering | 0.9904 | 0.9744 | 0.9001 | 0.6323 | 0.9236 | 0.8842 | 294 |
| `bge-small-en-v1.5` *(LOW)* | clustering-era sweep | 0.9745 | 0.9659 | 0.8912 | 0.6756 | 0.8750 | 0.8765 | 47 |
| `bge-large-en-v1.5` *(HIGH)* | clustering | 0.9936 | 0.9489 | 0.8813 | 0.6338 | 0.8750 | 0.8665 | 315 |
| **`ember-v1`** | **matching** | 0.9915 | 0.9710 | 0.7953 | 0.6413 | 0.9028 | 0.8604 | 245 |
| `F2LLM-0.6B` | clustering *(ranked 1)* | 0.9289 | 0.8910 | 0.7596 | 0.5859 | 0.7708 | 0.7873 | 188 |

`avsolatorio/GIST-large-Embedding-v0` is the first large model to dominate the configured HIGH
tier on both axes: 0.9031 against 0.8665 mean blocking recall, in 250 seconds against 315. It
leads dblp-scholar and ties walmart-amazon, and it is the best mean anything in this project has
measured.

`llmrails/ember-v1` is the counterexample that keeps the correlation honest. It has the highest
PairClassification score of the five at 87.37, above every model in the table, and the lowest
measured recall of the five, because it collapses to 0.7953 on abt-buy while the others sit near
0.89. The category picks a good pool; it does not pick the winner.

`KiteFishAI/Nano-Em1-0.6B-v2.1` leads PairClassification outright at 89.9 and could not be
measured at all: it is an LLM-based embedder whose tokenizer ships no chat template, so
`sentence-transformers` refuses to load it.

## JSON as an augmentation: the union of both blockings

The grid above tested JSON blocking as a replacement. `--blocking-strategy union` runs both
blockings and keeps the blocks from each, so a record sits in a name block *and* a JSON block
and a gold pair only has to be caught by one view. Blocks stop being disjoint, so two things in
the pipeline had to change: co-blocking became a shared-*any*-block test rather than a
same-block test, and the pair count became a distinct count over a packed-int set, so the
matcher is not billed twice for a pair both views found.

With the default `bge-small-en-v1.5`, the union wins on all five datasets:

| Strategy, `bge-small-en-v1.5` | dblp-acm | dblp-scholar | abt-buy | walmart-amazon | amazon-google | Mean |
|---|---|---|---|---|---|---|
| name only *(default)* | 0.9745 | 0.9659 | 0.8912 | 0.8750 | 0.6756 | 0.8765 |
| json only | 0.2070 | 0.6065 | 0.6113 | 0.7917 | 0.4753 | 0.5384 |
| **union** | **0.9766** | **0.9813** | **0.9248** | **0.9792** | **0.7803** | **0.9284** |
| union gain over name | +0.0021 | +0.0154 | +0.0336 | **+0.1042** | **+0.1047** | +0.0520 |
| pairs to judge, union/name | 2.07x | 1.88x | 1.36x | 1.71x | 1.79x | 1.76x |

The two views fail on different records, which is the whole reason this works. JSON blocking on
its own reaches 0.2070 on dblp-acm — a near-total failure — and the union still comes out 0.0021
above name-only there, so even a broken view contributes a few pairs the good view missed.

The gain concentrates exactly where the name is weakest. Walmart-amazon and amazon-google, the
two datasets with short abbreviated product names, gain over 0.10 each. Dblp-acm, where the
title is nearly a primary key, gains 0.0021 for 2.07x the comparisons and is not worth it. Pairs
are LLM calls, so the 1.76x mean pair count is real inference spend, not a rounding error.

On amazon-google, the dataset that has resisted every other change in this project, the union
sets a new ceiling:

| Strategy | amazon-google | Pairs to judge |
|---|---|---|
| **union** + `multilingual-e5-large-instruct` | **0.8251** | 65,253 |
| union + `bge-small-en-v1.5` *(default)* | 0.7803 | 74,112 |
| union + `bge-large-en-v1.5` | 0.7638 | 70,938 |
| json + `multilingual-e5-large-instruct` | 0.7429 | 40,182 |
| name + `bge-small-en-v1.5` *(default)* | 0.6756 | 41,337 |
| name + `bge-large-en-v1.5` | 0.6338 | 43,601 |
| name + `multilingual-e5-large-instruct` | 0.6143 | 39,293 |

0.8251 is +0.0822 over the previous project best and +0.2108 over the same model reading the
name alone, for 1.66x the comparisons. The union stays off by default because those comparisons
cost money, but it is the setting to reach for on product data.

## End to end, not just blocking

Blocking recall is a ceiling, not a score. Running both tiers all the way through matching on
`abt-buy`, 1,000-record sample at seed 42, three ER iterations, `gpt-oss-120b` as the matcher:

| Tier | Embedding | Precision | Recall | F1 | Seconds |
|---|---|---|---|---|---|
| **LOW** | `bge-small-en-v1.5` | 0.9204 | **0.8878** | **0.9038** | 410 |
| HIGH | `bge-large-en-v1.5` | **0.9265** | 0.8681 | 0.8963 | 508 |

LOW wins by 0.0075 F1, which tracks the blocking sweep: HIGH's 2K blocking recall on abt-buy is
0.8813 against LOW's 0.8912, and matching cannot recover a pair that blocking never proposed. The
ordering holds end to end, so the default is the right default.

Both of these runs are post-fix; the pipeline was losing whole blocks to an adapter bug until
[xml-adapter-block-loss.md](xml-adapter-block-loss.md), and any earlier end-to-end number here is not
comparable.

## What is configured

```yaml
models:
  embedding_low: "BAAI/bge-small-en-v1.5"
  embedding_high: "BAAI/bge-large-en-v1.5"
  embedding: "${models.embedding_low}"

er:
  blocking:
    strategy: name
```

LOW is the default for every command and `name` is the default strategy. HIGH is opt-in through
`serf benchmark --embedding-tier high` and earns its keep only on the bibliographic datasets,
where it leads by 0.019 on dblp-acm. `avsolatorio/GIST-large-Embedding-v0` is the better large
model on both recall and speed and is a one-line change to `models.embedding_high` for anyone
who wants it. The candidate lists in `benchmarks.embedding_candidates_large` now carry the
matching-category picks alongside the clustering ones, with the correlation recorded in the
comment so nobody repeats the clustering mistake.

## Reproducing

```bash
serf blocking-sweep --candidate-set large --blocking-strategy both --sample 2000 \
  --output data/blocking_sweep_large.json
serf blocking-sweep --candidate-set all --blocking-strategy all --sample 2000 \
  --output data/blocking_sweep_all.json
serf mteb-rank --candidate-set all --sweep data/blocking_sweep_all.json
serf benchmark -d abt-buy --sample-records 1000 --embedding-tier high --blocking-strategy union
```
