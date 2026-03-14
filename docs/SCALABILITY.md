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

**Pros**: Zero migration effort. Same API.
**Cons**: Slower for random access. Still single-machine. Limited by disk I/O.
**Scale**: ~100M entities on a single machine with fast SSD.

#### FAISS with GPU

For machines with GPUs, FAISS GPU indexes are 10-100x faster:

```python
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
```

**Pros**: Massive speedup for clustering. Same API.
**Cons**: GPU memory is even more limited than RAM (typically 16-80GB).
**Scale**: ~5M entities per GPU. Multi-GPU for more.

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

**Pros**: Excellent developer experience. GroupBy is directly useful for blocking.
**Cons**: No native IVF — uses HNSW which is NN-search oriented, not clustering.
**Scale**: ~100M entities per node, multi-node clusters.

#### Weaviate

[Weaviate](https://weaviate.io) offers:

- **Hybrid search** — Combine vector similarity with BM25 text search
- **Multi-tenancy** — Isolate datasets per tenant
- **Compression** — Product quantization and binary quantization
- **Schema-based** — Define entity classes with typed properties

**Pros**: Best hybrid search. Good for combining embedding blocking with keyword blocking.
**Cons**: Heavier infrastructure. HNSW-based (not IVF clustering).
**Scale**: ~50M entities per node.

### Tier 3: Approximate Clustering at Scale

#### Spark MLlib KMeans

For very large datasets already in Spark:

```python
from pyspark.ml.clustering import KMeans
kmeans = KMeans(k=num_clusters, featuresCol="embedding")
model = kmeans.fit(entity_df)
assignments = model.transform(entity_df)
```

**Pros**: Distributed. Integrates with SERF's PySpark pipeline. No external service.
**Cons**: Slower than FAISS. Less precise clustering.
**Scale**: Billions of entities across a Spark cluster.

#### ScaNN (Google)

[ScaNN](https://github.com/google-research/google-research/tree/master/scann) is Google's vector search library:

- **Asymmetric hashing** — Better accuracy/speed tradeoff than IVF
- **Partitioning** — Built-in tree-based partitioning similar to IVF
- **TensorFlow integration** — Works with TF Serving for production

**Scale**: ~100M entities in-memory. No distributed mode.

## Recommendation

| Dataset Size | Recommended Engine            | Notes                                  |
| ------------ | ----------------------------- | -------------------------------------- |
| < 1M         | **FAISS (current)**           | Fast, simple, in-memory                |
| 1M - 10M     | **FAISS memory-mapped**       | Same API, disk-backed inverted lists   |
| 10M - 100M   | **Milvus** or **Qdrant**      | Distributed, disk-based indexes        |
| 100M - 1B    | **Milvus** (distributed)      | Multi-node, GPU-accelerated            |
| > 1B         | **Milvus** + **Spark KMeans** | Hybrid: Spark for initial partitioning |

### Implementation Strategy

SERF should define a **`Blocker` protocol** (Python Protocol class) that `FAISSBlocker` implements. Alternative backends (Milvus, Qdrant, Spark KMeans) implement the same protocol:

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
