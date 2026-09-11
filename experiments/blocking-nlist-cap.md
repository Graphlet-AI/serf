# The FAISS cluster-count cap, and why extra ER rounds could not fix it

Blocking decides which record pairs the matcher is ever allowed to compare, so the share of
gold pairs that land in a shared block is a hard ceiling on end-to-end recall. Tracing every
missed gold pair across the four full-data runs showed 2,060 of 3,479 misses were pairs that
were never co-blocked, against only 594 the matcher looked at and rejected. That pointed at
blocking rather than at GPT OSS 120b, and the cause turned out to be arithmetic.

All measurements below use name-only blocking, `target_block_size: 30`, `max_block_size: 100`,
and full datasets. FAISS clustering is deterministic here: three repeat runs of the same
configuration on abt-buy returned 904/1097 every time, so the deltas are real rather than
k-means noise.

## The bug

`FAISS_SCRIPT` sized the IVF cell count as:

```python
nlist = max(1, n // target_block_size)
nlist = min(nlist, int(math.sqrt(n)))
```

The `sqrt(n)` term wins whenever `n > target_block_size ** 2`, which is 900 records at the
default target. Past that point `target_block_size` is silently unreachable and average block
size grows as `sqrt(n)` instead of staying at the target:

| Records | `n // 30` | `sqrt(n)` | nlist used | Avg cluster |
|---|---|---|---|---|
| 2,173 | 72 | 46 | 46 | 47.2 |
| 4,589 | 152 | 67 | 67 | 68.5 |
| 4,910 | 163 | 70 | 70 | 70.1 |
| 24,628 | 820 | 156 | 156 | 157.9 |
| 66,879 | 2,229 | 258 | 258 | 259.2 |

So every benchmark was affected, not only the large ones. `split_oversized_block` then chopped
each oversized cluster into contiguous `max_block_size` chunks by list order, with no regard for
similarity, which separated pairs that clustering had correctly grouped.

Isolating the cluster count on dblp-scholar, embedding once with `all-MiniLM-L6-v2` and only
re-clustering, gives:

| Cluster count | Blocks after split | Avg block | Max block | Blocking recall |
|---|---|---|---|---|
| `min(n//30, sqrt(n))` = 258 | 794 | 84.2 | 100 | 0.2469 |
| `n // 39` = 1,714 | 1,745 | 38.3 | 100 | 0.9039 |
| `n // 30` = 2,229 | 2,240 | 29.9 | 100 | **0.9205** |

Removing the cap is a strict improvement rather than a trade: recall rises from 0.2469 to
0.9205 *and* blocks shrink from 84.2 to 29.9, so matching gets cheaper at the same time. The
fix is to drop the `sqrt(n)` line.

## Extra rounds do not substitute for the fix

The reasonable objection is that iterative ER should recover these pairs anyway: a round that
merges entities re-blocks them, so a pair the first partition separated gets another chance.
Three rounds of 100 ought to do at least as well as one round of 300.

Measured with `serf blocking-sweep --rounds 3`, which merges every co-blocked gold pair between
rounds and therefore models a matcher that never errs, on dblp-scholar with `all-MiniLM-L6-v2`:

| Round | Entities | Blocks | Recall | Cumulative |
|---|---|---|---|---|
| 1 | 66,879 | 794 | 0.2469 | 0.2469 |
| 2 | 65,574 | 779 | 0.3163 | 0.3163 |
| 3 | 65,209 | 781 | 0.3488 | 0.3488 |

Against 0.9071 for a single round with splitting disabled, three oracle rounds closed only 18%
of the gap. Two things caused that:

1. **The cap pinned the partition.** Because `nlist` tracks `sqrt(n)` rather than the target,
   and because merging only took 66,879 entities down to 65,209, each round rebuilt nearly the
   same 259-record clusters and re-chopped them at nearly the same boundaries — 794, then 779,
   then 781 blocks.
2. **`auto_scale_by_iteration` could not help.** It shrinks the target to 15 then 10, but
   `sqrt(n)` still binds at those targets, so `nlist` stayed between 255 and 258 every round.

The loop was working. Blocking kept returning the same answer. With the cap removed a single
round reaches 0.9205, which makes the question moot.

## Two further blocking defects found alongside

- **Clustering raised `OSError: [Errno 7] Argument list too long` past roughly 13,000
  entities.** Entity ids were serialised into a single argv entry, and Linux caps one argument
  at 128 KB. Reproduced at n=20,000 and fixed by passing the ids through a temp file, the way
  embeddings were already handed to the subprocess. The full-data walmart-amazon and
  dblp-scholar runs had only worked because a runtime patch was in play that was never
  committed.
- **e5 and bge embeddings ran without their instruction prefix.** Both families are trained
  with a required prefix on every input. `models.embedding_prompt` now threads one through
  `EntityEmbedder` and the blocking subprocess. This is a smaller effect than expected and is
  dataset dependent rather than a straight bug: `query: ` on `multilingual-e5-base` gained
  +0.0200 on abt-buy and +0.0728 on amazon-google but lost 0.0611 on dblp-acm and 0.0117 on
  dblp-scholar, a mean of +0.0023.

## Consequence for the recorded baselines

Every F1 number recorded before these fixes understates what the pipeline does, most severely
on amazon-google and walmart-amazon where blocking misses were 61.4% and 64.0% of all gold
pairs. Re-running the LLM benchmarks costs real inference spend, so the existing numbers are
left in place and marked as pre-fix rather than silently replaced.
