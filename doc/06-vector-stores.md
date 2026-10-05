# 06 · Vector stores

All stores implement the same `VectorStore` port and pass the same contract test suite
(`tests/integration/test_store_contract.py`): dense search, sparse search, identical filter
semantics, payload patches, deletes by ID and filter, pagination, idempotent upserts.

| | `memory` | `qdrant` | `pgvector` |
|---|---|---|---|
| Use for | tests, demos, tiny single-replica | **default for production** | teams standardised on PostgreSQL |
| Dense index | exact (NumPy) | HNSW (+ optional scalar/binary quantization) | HNSW (`vector_cosine_ops` / `ip` / `l2`) |
| Sparse index | exact dot product | inverted index | `sparsevec` inner product (sequential scan) |
| IDF for BM25-style sparse | ✅ computed live | ✅ `Modifier.IDF` | ❌ — use SPLADE |
| Native hybrid fusion | ❌ | ✅ prefetch + weighted RRF / DBSF | ❌ |
| Filters | reference implementation | payload filter + payload indexes | JSONPath `@?` + GIN index |
| Persistence / HA | none | snapshots, replication, sharding | PostgreSQL tooling (PITR, replicas) |
| Extra | – | `qdrant` | `pgvector` |

## Qdrant

* One collection per logical collection: `<collection_prefix><name>` with named vectors `dense` and
  `sparse`.
* Payload indexes: `source_id` (keyword) and `chunk_index` (integer) always; plus
  `collections.<name>.payload_indexes`.
* Modes: server (`url`, `api_key`, `prefer_grpc`), embedded in-memory (`location: ":memory:"`),
  embedded on-disk (`path`). Embedded mode ignores payload indexes and weighs native RRF
  differently — prefer client-side fusion there.
* Scale knobs: `hnsw.m` / `ef_construct` / `ef_search`, `quantization: scalar` (≈4× less RAM),
  `on_disk_vectors`, `on_disk_payload`, `shard_number`, `replication_factor`.
* Multi-tenancy: for many tenants (sites, customers) prefer **one collection** partitioned by a
  payload field over one collection per tenant. Declare it with
  `payload_indexes: [{field: site, tenant: true}]` and Qdrant creates an `is_tenant` keyword index
  that co-locates each tenant's points. Callers must filter by the field; see
  [cookbook recipe 03](../cookbook/recipes/03-cms-multi-site/README.md).

## pgvector

* Requires the `vector` extension ≥ 0.7 (`sparsevec`); iterative index scans (≥ 0.8) are enabled
  automatically for filtered queries so `LIMIT` is honoured.
* Table `<db_schema>.<table_prefix><name>`: `id uuid PK, source_id, chunk_index, payload jsonb,
  dense vector(D), sparse sparsevec(S)`; indexes on `(source_id, chunk_index)`, GIN on `payload`,
  HNSW on `dense`.
* Filters compile to parameterised JSONPath (`payload @? $1::jsonpath`); values are JSON-encoded and
  bound, so filters cannot inject SQL. `source_id` filters use the column.
* The registry can live in the same database (`registry.url: postgresql+psycopg://…`).

## Choosing embedding dimensions

Memory ≈ `points × dim × 4 bytes` (+ HNSW links ≈ `points × m × 8 bytes`). One million 384-d chunks
≈ 1.5 GB of vectors; 1024-d ≈ 4 GB. Scalar quantization cuts vector RAM by ~4×.

## Changing the embedding model without downtime

A model change makes vectors incompatible; the collection turns **stale** and refuses traffic
until rebuilt (`POST /collections/{c}/reindex?recreate=true`, which drops and re-embeds). To avoid
the gap:

1. `POST /collections` a new collection (e.g. `products-v2`) with the new `embedding` override.
2. Re-send documents to it (or copy sources via your pipeline); verify with `/evaluate`.
3. Point clients at `products-v2` (for the OpenAI facade: model `rag:products-v2`).
4. `DELETE /collections/products`.

## Adding another store

Implement the `VectorStore` Protocol (`domain/ports.py`) in `adapters/vectorstores/<name>.py`,
register it with `@VECTOR_STORES.register("<name>")`, translate `FilterSpec` to the native query
language mirroring `domain.filters.matches`, and add it to the contract test parameters. See
[08-extending.md](08-extending.md) and the `add-rag-component` skill.
