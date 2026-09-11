# Large blocking embeddings and JSON record blocking

Two questions, measured together on the same grid:

1. Does a large embedding model buy blocking recall that `BAAI/bge-small-en-v1.5` — the 33M
   winner of [embedding-blocking-sweep.md](embedding-blocking-sweep.md) — does not already have?
2. Does embedding the whole record as JSON, with the field names inline, beat embedding the
   name alone?

Both answers are no, with one narrow exception worth keeping.

Protocol: `serf blocking-sweep --candidate-set large --blocking-strategy both`, 2,000-record
samples per dataset at `seed 42`, `target_block_size: 30`, one pass, CPU-only on a 4-core box.
Sampling keeps ground-truth match groups whole, so gold pairs survive; the gold-pair counts are
942 (dblp-acm), 587 (dblp-scholar), 1,011 (abt-buy), 144 (walmart-amazon) and 669
(amazon-google).

## Candidates

Selected from MTEB(eng, v2) clustering — the task category that actually matches what blocking
does — filtered to 3B parameters and 1,024 output dimensions or fewer, read from the
`mteb/results` dataset rather than the leaderboard Space, which renders client-side.

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

So JSON blocking stays available and stays off by default. It is the right setting for
amazon-google-shaped data — many short, ambiguous names plus discriminative side fields — paired
with a large model, and the wrong setting everywhere else.

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

LOW is the default for every command. HIGH is opt-in through `serf benchmark --embedding-tier
high` and earns its keep only on the bibliographic datasets, where it leads by 0.019 on
dblp-acm. `mxbai-embed-large-v1` is the better large model on the mean and is a one-line change
to `models.embedding_high` for anyone who wants it.

## Reproducing

```bash
serf blocking-sweep --candidate-set large --blocking-strategy both --sample 2000 \
  --output data/blocking_sweep_large.json
serf benchmark -d abt-buy --sample-records 1000 --embedding-tier high --blocking-strategy json
```
