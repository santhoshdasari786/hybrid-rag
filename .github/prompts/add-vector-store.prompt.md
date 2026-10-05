---
description: Add a new vector database adapter that passes the store contract suite
agent: agent
argument-hint: "store name, e.g. weaviate / elasticsearch / milvus"
---

Add a vector store adapter named `${input:store:store name}` to the Advanced RAG Service.

Follow the `add-rag-component` skill and these steps:

1. Read `src/rag_service/domain/ports.py` (`VectorStore`, `CollectionAdmin`, `VectorWriter`,
   `VectorReader`, `StoreCapabilities`) and one existing adapter
   (`src/rag_service/adapters/vectorstores/qdrant.py` or `pgvector.py`) as the reference.
2. Create `src/rag_service/adapters/vectorstores/${input:store}.py`:
   - import the client library lazily; raise `DependencyMissingError.for_extra(...)`;
   - one physical index per logical collection; dense + sparse vectors if the database supports them;
   - translate `FilterSpec` to the native filter language so results match `domain.filters.matches`
     (array membership for `eq`/`in`, missing fields pass `ne`/`nin`, ranges on numbers and ISO strings);
   - map client errors to `CollectionNotFoundError`, `ProviderError`, `ProviderUnavailableError`;
   - set `capabilities` truthfully; raise `CapabilityNotSupportedError` from `search_hybrid` if not native;
   - register with `@VECTOR_STORES.register("${input:store}")` and import the module in
     `adapters/vectorstores/__init__.py`.
3. Add a config section (connection settings) to `VectorStoreConfig` in `core/config.py` **or** read
   settings from `vector_store.options` — prefer `options` unless the settings are essential.
4. Add an optional dependency extra in `pyproject.toml`.
5. Add the store to `_params()` in `tests/integration/test_store_contract.py` (skip unless an env var
   such as `RAG_IT_${input:store}_URL` is set) and run the suite against a local container.
6. Document it in `doc/06-vector-stores.md` and `doc/02-configuration.md`.
7. Run `make lint test` and report results.
