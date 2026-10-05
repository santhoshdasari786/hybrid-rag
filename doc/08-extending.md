# 08 · Extending the service

Every pipeline strategy is resolved by name from a registry in `rag_service.core.registry`.
Adding a component = implement the Protocol from `rag_service.domain.ports` + register a factory.
No existing code changes (Open/Closed).

## Registries and factory signatures

| Registry | Config key that selects it | Factory keyword arguments | Returns (Protocol) |
|---|---|---|---|
| `PARSERS` (keyed by MIME type) | document `mime_type` | – | `DocumentParser` |
| `LENGTH_FUNCTIONS` | `chunking.length_unit` / `tokenizer` | `encoding` (optional) | `LengthFunction` |
| `CHUNKERS` | `chunking.strategy` | `config: ChunkingConfig, length_fn, dense_embedder` | `Chunker` |
| `DENSE_EMBEDDERS` | `embedding.dense.provider` | `config: DenseEmbeddingConfig` | `DenseEmbedder` |
| `SPARSE_EMBEDDERS` | `embedding.sparse.provider` | `config: SparseEmbeddingConfig` | `SparseEmbedder` |
| `VECTOR_STORES` | `vector_store.provider` | `config: VectorStoreConfig` | `VectorStore` |
| `FUSIONS` | `retrieval.fusion.method` | – | `FusionStrategy` |
| `RERANKERS` | `retrieval.rerank.provider` | `config: RerankConfig` | `Reranker` |
| `QUERY_TRANSFORMS` | `retrieval.query_transform.strategy` | `config: QueryTransformConfig, llm` | `QueryTransformer` |
| `LLMS` | `llm.provider` | `config: LLMConfig` | `LLMClient` |

Provider-specific settings go in the `options: {}` dict of the relevant config section, so new
components need no schema changes.

## Example: a paragraph chunker

```python
# my_company_rag/paragraph.py
from rag_service.adapters.chunking.text_splitters import BaseChunker
from rag_service.core.registry import CHUNKERS


class ParagraphChunker(BaseChunker):
    name = "paragraph"

    def split_text(self, text: str) -> list[str]:
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        return self.pack(paragraphs)  # packs to chunk_size with overlap; hard-splits oversized pieces


@CHUNKERS.register("paragraph", description="One chunk per paragraph, packed to chunk_size")
def build(*, config, length_fn, dense_embedder=None):
    return ParagraphChunker(config, length_fn)
```

Load it with either:

```yaml
plugins: [my_company_rag.paragraph]          # config file
```

or a packaged entry point (auto-discovered):

```toml
[project.entry-points."rag_service.plugins"]
paragraph = "my_company_rag.paragraph"   # importing the module registers it; a callable target
                                         # is also called with the dict of all registries
```

Then `chunking: {strategy: paragraph}`.

## Example: an external reranker via HTTP

No code needed for Cohere-compatible APIs:

```yaml
retrieval:
  rerank:
    enabled: true
    provider: http
    model: rerank-v3.5
    options: {url: https://api.cohere.com/v2/rerank, api_key: "${COHERE_API_KEY}"}
```

## Checklist for a new component

1. Implement the Protocol; keep I/O async (wrap CPU-bound work in `asyncio.to_thread`).
2. Import optional third-party libraries **inside** the constructor and raise
   `DependencyMissingError.for_extra(package, extra)` when missing, so the service still starts
   without the extra.
3. Raise only `rag_service.core.errors` exceptions (`ProviderError`, `ProviderUnavailableError`,
   `ValidationFailedError`, …) — the API maps them to problem responses.
4. Register with a stable lowercase name and a one-line description (shown in `/system/info`).
5. Add tests: unit tests for behaviour; for vector stores add the store to the contract suite.
6. Document the new option in `doc/02-configuration.md`.

The `add-rag-component` agent skill (`.github/skills/add-rag-component`) walks an AI assistant
through exactly these steps.

## Replacing infrastructure ports

| Port | Default | Replace when |
|---|---|---|
| `JobRunner` | `InProcessJobRunner` | work must survive restarts or be shared across replicas (queue-backed runner) |
| `SourceRepository` / `CollectionRepository` / `JobRepository` | SQL (SQLite / PostgreSQL) | you need another system of record |
| Authentication | `ApiKeyAuthenticator` | OIDC/JWT: map a verified token to a `Principal(name, scopes, collections)` |

Swap implementations in `container.py` (the composition root) only.
