# Choosing the blocking embedding

Blocking recall — the share of gold pairs whose two records land in a shared block — is a hard
ceiling on end-to-end recall, because matching only ever sees pairs blocking put together. This
compares nine small sentence-transformer models on that one number, with no LLM in the loop, so
the sweep costs CPU time and nothing else.

Protocol: `serf blocking-sweep`, full tables, name-only embedding, `target_block_size: 30`,
`max_block_size: 100`, one pass. FAISS clustering is deterministic here (three repeat runs on
abt-buy returned 904/1097 every time), so the differences below are real rather than k-means
noise. Timings are single-threaded CPU on a 4-core box with no GPU, summed over all five
datasets.

## Results

| Model | Params | Dim | DBLP-ACM | DBLP-Scholar | Abt-Buy | Amazon-Google | Walmart-Amazon | Mean | Secs |
|---|---|---|---|---|---|---|---|---|---|
| `thenlper/gte-base` | 109M | 768 | 0.9708 | 0.9211 | 0.8724 | **0.6821** | 0.8514 | **0.8595** | 661 |
| **`BAAI/bge-small-en-v1.5`** | 33M | 384 | 0.9654 | 0.9048 | **0.8952** | 0.6512 | **0.8545** | 0.8542 | **273** |
| `thenlper/gte-small` | 33M | 384 | 0.9604 | **0.9233** | 0.8824 | 0.6410 | 0.8410 | 0.8496 | 228 |
| `BAAI/bge-base-en-v1.5` | 109M | 768 | **0.9856** | 0.9020 | 0.8569 | 0.6461 | **0.8545** | 0.8490 | 898 |
| `all-MiniLM-L6-v2` | 22M | 384 | 0.9717 | 0.9205 | 0.8551 | 0.6435 | 0.7775 | 0.8337 | 133 |
| `all-mpnet-base-v2` | 109M | 768 | 0.9793 | **0.9278** | 0.8323 | 0.6590 | 0.7391 | 0.8275 | 697 |
| `multilingual-e5-small` | 118M | 384 | 0.9699 | 0.8945 | 0.8642 | 0.5810 | 0.8108 | 0.8241 | 251 |
| `multilingual-e5-base` *(former default)* | 278M | 768 | 0.9735 | 0.8898 | 0.8724 | 0.5047 | 0.8025 | 0.8086 | 718 |
| `multilingual-e5-base` + prefix | 278M | 768 | 0.9636 | 0.8915 | 0.8587 | 0.5296 | 0.7942 | 0.8075 | 786 |

The bge and e5 rows are measured with their instruction prefix, the rest without; see below.

## The pick

`BAAI/bge-small-en-v1.5` is now the default. It is not the top of the table, but it is the only
candidate that dominates the model it replaces on both axes at once:

| Dataset | `multilingual-e5-base` | `bge-small-en-v1.5` | Recall delta | Embed secs |
|---|---|---|---|---|
| dblp-acm | 0.9735 | 0.9654 | −0.0081 | 33 → 14 |
| abt-buy | 0.8724 | 0.8952 | +0.0228 | 20 → 17 |
| amazon-google | 0.5047 | 0.6512 | +0.1465 | 36 → 15 |
| walmart-amazon | 0.8025 | 0.8545 | +0.0520 | 193 → 71 |
| dblp-scholar | 0.8898 | 0.9048 | +0.0150 | 436 → 157 |
| **mean** | **0.8086** | **0.8542** | **+0.0456** | **718 → 273** |

It wins four of five datasets, gives up 0.008 on dblp-acm, and does it in 38% of the time at
384 dimensions instead of 768 — which also shrinks the FAISS index.

`gte-base` scores 0.0053 higher on the mean and is the better choice if recall is all that
matters. It costs 2.4x the embedding time for that, and blocking re-embeds every record on
every ER round rather than once per corpus, so the multiplier applies `er.max_iterations` times
per run. On CPU-only hardware that is the wrong trade for half a point. Set `models.embedding`
to `thenlper/gte-base` and clear `models.embedding_prompt` to take it.

Two things worth noting about the table beyond the winner:

- **Bigger is not better here.** The two 109M models beat the 33M models on dblp-scholar and
  amazon-google but lose badly on walmart-amazon, and the 278M `multilingual-e5-base` is last.
  Blocking asks a narrow question — does the true match land in the same Voronoi cell — and
  multilingual capacity spent on languages these English benchmarks do not contain is capacity
  not spent on product and paper titles.
- **Amazon-Google is the binding constraint.** Every model lands between 0.50 and 0.69 there,
  far below its score anywhere else, so roughly a third of that dataset's gold pairs are
  unreachable by matching no matter how good the LLM is. That is the dataset to fine-tune
  against if blocking recall is ever worth training for.

## The instruction prefix

The e5 and bge families are trained with a prefix on every input, and the pipeline was not
supplying one. `models.embedding_prompt` now threads it through `EntityEmbedder` and the
blocking subprocess.

The effect is smaller than the training recipes imply, and it is not the clear bug it first
looked like. On `multilingual-e5-base`, `query: ` moves the five datasets by −0.0099, −0.0137,
+0.0249, −0.0083 and +0.0017, a mean of −0.0011. Blocking compares two texts that both carry
the same prefix, so a constant offset in embedding space largely cancels, which is a reasonable
explanation for why a setting that matters for asymmetric retrieval barely registers for
symmetric clustering. The prefix stays configured because it is what the bge default is trained
to expect, not because it was measured to pay.

An earlier version of this comparison, run before the cluster-count fix described in
[blocking-nlist-cap.md](blocking-nlist-cap.md), reported the prefix gaining +0.0200 on abt-buy
and +0.0728 on amazon-google. Those runs were dominated by oversized blocks being chopped by
list order, so they were measuring the bug more than the prefix; the numbers here supersede
them.

## Why the earlier ranking was wrong

This sweep was first run with the FAISS `sqrt(n)` cluster-count cap still in place, and the
ranking it produced does not survive the fix. `all-mpnet-base-v2` led that table and finishes
sixth here; `multilingual-e5-base` on dblp-acm went from 0.8291 to 0.9735. When most gold pairs
are being separated by an arbitrary chunking of oversized clusters, the measurement is mostly
about the chunking. Any embedding comparison run against this pipeline before commit `2388eae`
should be discarded.

## Where the candidate list went next

These nine were hand-picked. The large-model follow-up in
[low-high-embeddings-and-json-blocking.md](low-high-embeddings-and-json-blocking.md) picked its
candidates from MTEB(eng, v2) clustering score instead, which turned out to carry no
information about blocking recall — Spearman +0.0165 over thirteen models. `serf mteb-rank`
shows `PairClassification` is the category that does, at +0.5714, and selecting on it surfaced
`avsolatorio/GIST-large-Embedding-v0` at 0.9031 mean blocking recall, the best measured here.
`--blocking-strategy union` adds a further +0.0520 on top of the default model by blocking a
second time over the record as JSON and keeping both sets of blocks.

## Reproducing

```bash
serf blocking-sweep --output data/blocking_sweep.json
```

Candidates default to `benchmarks.embedding_candidates` in `config.yml`; `--model`, `--prompt`,
`--dataset`, `--sample` and `--target-block-size` override. The full run is about 4.2 hours of
CPU for nine models across five datasets, dominated by dblp-scholar's 66,879 records.
