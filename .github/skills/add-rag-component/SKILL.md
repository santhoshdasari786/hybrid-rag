---
name: add-rag-component
description: Add a pluggable pipeline component to the Advanced RAG Service — document parser, chunker, dense/sparse embedder, vector store, fusion method, reranker, query transform or LLM provider — registered by name, with tests and docs. Use when extending retrieval/ingestion capabilities or integrating a new provider.
argument-hint: "[component kind] [name]"
---

# Add a RAG pipeline component

Components are selected by name from registries (`src/rag_service/core/registry.py`) and implement
Protocols from `src/rag_service/domain/ports.py`. Adding one never requires editing services.

## 1. Pick the registry and contract

| Kind | Registry | Protocol | Factory kwargs | Reference implementation |
|---|---|---|---|---|
| parser (by MIME) | `PARSERS` | `DocumentParser` | – | `adapters/parsers/builtin.py` |
| chunker | `CHUNKERS` | `Chunker` | `config, length_fn, dense_embedder` | `adapters/chunking/text_splitters.py` |
| dense embedder | `DENSE_EMBEDDERS` | `DenseEmbedder` | `config` | `adapters/embedding/openai_embedder.py` |
| sparse embedder | `SPARSE_EMBEDDERS` | `SparseEmbedder` | `config` | `adapters/embedding/local.py` |
| vector store | `VECTOR_STORES` | `VectorStore` | `config` (VectorStoreConfig) | `adapters/vectorstores/qdrant.py` |
| fusion | `FUSIONS` | `FusionStrategy` | – | `adapters/fusion/strategies.py` |
| reranker | `RERANKERS` | `Reranker` | `config` | `adapters/rerank/rerankers.py` |
| query transform | `QUERY_TRANSFORMS` | `QueryTransformer` | `config, llm` | `adapters/query/transforms.py` |
| LLM | `LLMS` | `LLMClient` | `config` | `adapters/llm/openai_llm.py` |

Read the Protocol and the reference implementation before writing code.

## 2. Implement

Start from [templates/component.py](./templates/component.py). Rules:

- Module in the matching `src/rag_service/adapters/<package>/`; import it from the package's
  `__init__.py` so it registers at startup.
- Optional libraries imported inside `__init__`; on `ImportError` raise
  `DependencyMissingError.for_extra("<pkg>", "<extra>")`; add the extra to `pyproject.toml`
  (and to `all`).
- Provider-specific settings from `config.options` (no schema change needed).
- Async methods; CPU-bound work in `asyncio.to_thread`; lazy, thread-safe model loading; expose
  `async def warmup()` if loading is slow.
- Map SDK errors: timeouts/429/5xx → `ProviderUnavailableError`, other → `ProviderError`.
- Vector stores: filter semantics must equal `domain.filters.matches`; use `PayloadKeys`.
- Register: `@REGISTRY.register("name", description="one line")`.

## 3. Test

- Unit tests next to similar ones in `tests/unit/` (template: [templates/test_component.py](./templates/test_component.py)).
- Vector store → add to `_params()` in `tests/integration/test_store_contract.py`.
- Provider needing network → unit-test request construction with a stub; add an opt-in test gated
  by an env var.
- `make lint test` must pass.

## 4. Document

- Config keys/options → `doc/02-configuration.md`; usage guidance → `doc/04-retrieval-and-tuning.md`
  or `doc/06-vector-stores.md`; mention the extra in `README.md` if user-facing.
- Verify with `GET /api/v1/system/info` that the component is listed.
