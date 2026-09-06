# SERF Scalability: Beyond System RAM

## Current Architecture

SERF uses **FAISS IndexIVFFlat** for semantic blocking — clustering entity embeddings into blocks for LLM matching. FAISS runs entirely in-memory:

- **Index type**: IVF (Inverted File) with flat inner product search
- **Operation**: Cluster assignment — each entity is assigned to its nearest centroid
- **Memory**: All embeddings must fit in RAM (~4 bytes × dimensions × entities)
- **Scale limit**: ~10-50M entities on a 64GB machine (with 1024-dim embeddings)

### Memory Requirements

| Entities | Dimensions | Memory (embeddings only) |
| -------- | ---------- | ------------------------ |
| 100K     | 1024       | ~400 MB                  |
| 1M       | 1024       | ~4 GB                    |
| 10M      | 1024       | ~40 GB                   |
| 100M     | 1024       | ~400 GB                  |
| 1B       | 1024       | ~4 TB                    |

Beyond ~10M entities, FAISS requires either quantization (lossy), memory-mapped indexes (slow), or a distributed solution.

## Embedding Model Choice

Dimension is the multiplier in the table above, so the embedding model is a scalability lever before any vector engine is involved. It is also a _quality_ lever: blocking bounds recall, and recall is currently SERF's binding constraint (0.57–0.74 against precision as high as 0.99), so a better embedder attacks the actual bottleneck.

SERF currently uses `intfloat/multilingual-e5-base`. Microsoft's [harrier-oss-v1](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) family is the strongest open alternative — MIT licensed, multilingual, and state of the art on Multilingual MTEB v2 at release.

| Model | Params | Dim | Max tokens | MTEB v2 | RAM @ 1M entities |
| --- | --- | --- | --- | --- | --- |
| `intfloat/multilingual-e5-base` (current) | 278M | 768 | 512 | — | 3.1 GB |
| [`microsoft/harrier-oss-v1-270m`](https://huggingface.co/microsoft/harrier-oss-v1-270m) | 268M | 640 | 32,768 | 66.5 | 2.6 GB |
| [`microsoft/harrier-oss-v1-0.6b`](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | 596M | 1,024 | 32,768 | 69.0 | 4.1 GB |
| [`microsoft/harrier-oss-v1-27b`](https://huggingface.co/microsoft/harrier-oss-v1-27b) | 27B | 5,376 | 32,768 | 74.3 | 21.5 GB |

The MTEB v2 column is from the harrier model card. No verified MTEB v2 score for e5-base was available, so that cell is blank rather than guessed — do not fill it in without a source.

Read the two axes separately:

- **`harrier-oss-v1-270m` is the scalability play.** At 640 dimensions it needs _less_ memory than the current model while scoring 66.5 on MTEB v2, and it raises the practical FAISS ceiling by roughly 20% at equal RAM.
- **`harrier-oss-v1-0.6b` is the quality play.** It costs 33% more memory than e5-base (768 → 1,024 dims) and buys a substantially stronger embedder. At SERF's current benchmark scale this is free — the datasets are thousands of records, not millions — so it is the right default until the corpus is large enough for dimension to matter.
- **`harrier-oss-v1-27b` is not a blocking model.** At 5,376 dimensions it quadruples index memory and needs serious GPU capacity to encode. Useful as an upper bound when measuring how much blocking quality caps recall, not for production blocking.

Three practical notes before swapping the model:

**Query instructions are mandatory.** These are instruction-tuned decoder models; omitting the prompt degrades quality. For entity resolution blocking the right prompt is `sts_query` ("Retrieve semantically similar text"), not `web_search_query`. Documents take no instruction. Since SERF embeds records rather than running query-document retrieval, the sensible reading is to embed every record instruction-free, or apply `sts_query` uniformly — this is worth an ablation rather than an assumption.

**Pooling and metric already match.** harrier uses last-token pooling with L2 normalization, so inner product equals cosine similarity and SERF's existing `METRIC_INNER_PRODUCT` FAISS configuration needs no change.

**The 32,768-token context is a real gain for wide records.** e5-base truncates at 512 tokens, which silently discards content on long product descriptions or bibliographic records. That truncation is invisible in the metrics and could be part of the recall gap.

Switching is a config change, since the model name is not hardcoded:

```yaml
# config.yml
models:
  embedding: "microsoft/harrier-oss-v1-0.6b"
```

Encoding cost rises with parameter count, and blocking re-embeds the surviving records every iteration, so measure encode throughput alongside quality before adopting a larger model.

## What SERF Needs From a Vector Engine

SERF's blocking step has specific requirements that differ from typical vector search:

1. **Cluster assignment** — Assign every entity to a cluster (centroid), not just find nearest neighbors for a query. This is the IVF "quantizer.search" pattern.
2. **Batch operations** — Process millions of entities at once, not one-at-a-time queries.
3. **Configurable cluster count** — Control `nlist` (number of clusters) to target specific block sizes.
4. **Inner product metric** — Normalized embeddings use inner product (equivalent to cosine similarity).
5. **Iterative re-clustering** — Each ER iteration re-embeds and re-clusters the (smaller) dataset.
6. **No persistence required** — Blocking is ephemeral; we don't need to persist the index between runs.

## Recommended Vector Engines for Scale

### Tier 1: Drop-in FAISS Replacements (Easiest Migration)

#### FAISS with Memory-Mapped Indexes

FAISS itself supports on-disk indexes via `faiss.write_index` / `faiss.read_index` with memory mapping. For IVF indexes, only the inverted lists are memory-mapped while centroids stay in RAM.

```python
# Write index to disk
faiss.write_index(index, "blocks.index")
# Read with memory mapping (inverted lists on disk)
index = faiss.read_index("blocks.index", faiss.IO_FLAG_MMAP)
```

**Pros**: Zero migration effort. Same API. **Cons**: Slower for random access. Still single-machine. Limited by disk I/O. **Scale**: ~100M entities on a single machine with fast SSD.

#### FAISS with GPU

For machines with GPUs, FAISS GPU indexes are 10-100x faster:

```python
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
```

**Pros**: Massive speedup for clustering. Same API. **Cons**: GPU memory is even more limited than RAM (typically 16-80GB). **Scale**: ~5M entities per GPU. Multi-GPU for more.

### Tier 2: Vector Databases (Production Scale)

#### Milvus (Recommended for SERF)

[Milvus](https://milvus.io) is the best fit for SERF's blocking needs:

- **IVF_FLAT index** — Same algorithm as FAISS, same clustering behavior
- **Billion-scale** — Handles billions of vectors with distributed architecture
- **Disk index** — DiskANN-based indexes for beyond-RAM datasets
- **GPU acceleration** — Optional GPU support for index building
- **Batch operations** — Efficient bulk insert and search
- **Open source** — Apache 2.0 license, self-hosted or Zilliz Cloud managed

**Migration path**: Replace `FAISSBlocker` with a Milvus client that:

1. Creates a collection with IVF_FLAT index
2. Bulk-inserts all entity embeddings
3. Uses `search` with `nprobe=1` to get cluster assignments
4. Groups results by cluster ID to form blocks

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
client.create_collection("entities", dimension=1024)
client.create_index("entities", "embedding", {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": num_clusters}
})
client.insert("entities", embeddings)
# Search each vector against centroids for cluster assignment
results = client.search("entities", embeddings, limit=1)
```

**Scale**: Billions of entities. Distributed across multiple nodes.

#### Qdrant

[Qdrant](https://qdrant.tech) is a strong alternative:

- **Rust-based** — High performance, low memory overhead
- **Quantization** — Scalar and product quantization reduce memory 4-32x
- **On-disk storage** — Memory-mapped HNSW indexes
- **GroupBy API** — Native grouping of results by payload field (useful for blocking)
- **Filtering** — Filter by entity type, source table, etc. during search

**Pros**: Excellent developer experience. GroupBy is directly useful for blocking. **Cons**: No native IVF — uses HNSW which is NN-search oriented, not clustering. **Scale**: ~100M entities per node, multi-node clusters.

#### Weaviate

[Weaviate](https://weaviate.io) offers:

- **Hybrid search** — Combine vector similarity with BM25 text search
- **Multi-tenancy** — Isolate datasets per tenant
- **Compression** — Product quantization and binary quantization
- **Schema-based** — Define entity classes with typed properties

**Pros**: Best hybrid search. Good for combining embedding blocking with keyword blocking. **Cons**: Heavier infrastructure. HNSW-based (not IVF clustering). **Scale**: ~50M entities per node.

### Tier 3: Approximate Clustering at Scale

#### Spark MLlib KMeans

For very large datasets already in Spark:

```python
from pyspark.ml.clustering import KMeans
kmeans = KMeans(k=num_clusters, featuresCol="embedding")
model = kmeans.fit(entity_df)
assignments = model.transform(entity_df)
```

**Pros**: Distributed. Integrates with SERF's PySpark pipeline. No external service. **Cons**: Slower than FAISS. Less precise clustering. **Scale**: Billions of entities across a Spark cluster.

#### ScaNN (Google)

[ScaNN](https://github.com/google-research/google-research/tree/master/scann) is Google's vector search library:

- **Asymmetric hashing** — Better accuracy/speed tradeoff than IVF
- **Partitioning** — Built-in tree-based partitioning similar to IVF
- **TensorFlow integration** — Works with TF Serving for production

**Scale**: ~100M entities in-memory. No distributed mode.

## Recommendation

| Dataset Size | Recommended Engine | Notes |
| --- | --- | --- |
| < 1M | **FAISS (current)** | Fast, simple, in-memory |
| 1M - 10M | **FAISS memory-mapped** | Same API, disk-backed inverted lists |
| 10M - 100M | **Milvus** or **Qdrant** | Distributed, disk-based indexes |
| 100M - 1B | **Milvus** (distributed) | Multi-node, GPU-accelerated |
| > 1B | **Milvus** + **Spark KMeans** | Hybrid: Spark for initial partitioning |

### Implementation Strategy

SERF should define a `Blocker` **protocol** (Python Protocol class) that `FAISSBlocker` implements. Alternative backends (Milvus, Qdrant, Spark KMeans) implement the same protocol:

```python
from typing import Protocol

class Blocker(Protocol):
    def block(
        self,
        embeddings: NDArray[np.float32],
        ids: list[str],
    ) -> dict[str, list[str]]:
        """Assign entities to blocks. Returns {block_key: [entity_ids]}."""
        ...
```

This allows swapping the blocking backend without changing any pipeline code:

```yaml
# er_config.yml
blocking:
  backend: milvus # or "faiss", "qdrant", "spark"
  target_block_size: 30
  max_block_size: 100
  milvus_uri: "http://milvus:19530"
```

## Cost Considerations

| Engine       | Infrastructure Cost (1B entities)    | Operational Complexity |
| ------------ | ------------------------------------ | ---------------------- |
| FAISS        | $0 (in-process)                      | None                   |
| Milvus       | ~$500-2000/mo (3-node cluster)       | Medium                 |
| Qdrant Cloud | ~$300-1000/mo                        | Low (managed)          |
| Pinecone     | ~$1000-5000/mo (serverless)          | Very Low (managed)     |
| Spark KMeans | Variable (cluster compute time only) | High (Spark ops)       |

For SERF's use case — ephemeral blocking indexes rebuilt each iteration — the cost of a persistent vector database may be unnecessary for datasets under 10M. FAISS with memory mapping or GPU acceleration covers most practical ER workloads. A vector database becomes worthwhile when:

1. The dataset exceeds 10M entities
2. You need incremental updates (new entities added between iterations)
3. You want to persist blocking indexes across pipeline runs
4. You're running multiple ER pipelines concurrently against the same data
