---
description: Add a dense or sparse embedding provider
agent: agent
argument-hint: "provider name, dense|sparse"
---

Add a `${input:kind:dense or sparse}` embedding provider named `${input:name:provider name}`.

1. Implement `DenseEmbedder` (`model_id`, `dimension`, `distance`, `embed_documents`, `embed_query`) or
   `SparseEmbedder` (`model_id`, `requires_idf`, …) from `domain/ports.py` in
   `src/rag_service/adapters/embedding/${input:name}.py`.
2. Batch by `config.batch_size`; run blocking inference in `asyncio.to_thread`; load models lazily and
   thread-safely; provide `async def warmup()`; apply `query_prefix` / `document_prefix` for dense.
3. If the dimension is not known from config, discover it in `warmup()` (see the OpenAI embedder).
4. Register in `DENSE_EMBEDDERS` / `SPARSE_EMBEDDERS`; add an optional extra in `pyproject.toml`.
5. Unit-test with a stubbed client (no network); document in `doc/02-configuration.md`.
6. Remember: changing a collection's embedding makes it stale → reindex with `recreate=true`.
