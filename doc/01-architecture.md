# 01 · Architecture

## Layers (hexagonal / ports & adapters)

```mermaid
flowchart TB
  subgraph API["API layer — rag_service.api"]
    R[Routers /api/v1 + /v1] --> D[Deps: auth, scopes, ACL]
    M[Middleware: request-id, access log, metrics,<br/>security headers, body limit, CORS, gzip]
    E[Errors → RFC 9457 problem+json]
  end
  subgraph APP["Application layer — rag_service.application"]
    CS[CollectionService] --- PF[PipelineFactory]
    IS[IngestionService] --- JS[JobService]
    RS[RetrievalService] --- CH[ChatService]
    SS[SourceService] --- EV[EvaluationService]
  end
  subgraph DOM["Domain — rag_service.domain"]
    MO[Models] --- FI[FilterSpec DSL] --- PO[Ports / Protocols]
  end
  subgraph ADP["Adapters — rag_service.adapters"]
    PA[Parsers] --- CK[Chunkers] --- EN[Enrichers]
    EM[Embedders] --- VS[Vector stores] --- FU[Fusion / MMR]
    RR[Rerankers] --- QT[Query transforms] --- LL[LLM clients]
    RG[SQL registry] --- JR[Job runner]
  end
  API --> APP --> DOM
  ADP -. implements .-> PO
  C[container.py<br/>composition root] --> API
  C --> APP
  C --> ADP
```

| Layer | Package | Depends on | Never depends on |
|-------|---------|-----------|------------------|
| Core | `rag_service.core` | stdlib, pydantic | anything else in the service |
| Domain | `rag_service.domain` | core | adapters, application, api |
| Adapters | `rag_service.adapters` | core, domain | application, api |
| Application | `rag_service.application` | core, domain, adapters (only via `PipelineFactory`) | api |
| API | `rag_service.api` | everything above | — |

The "no API imports below the API layer" rule is enforced by ruff (`TID251` banned-api).

## SOLID in this code base

| Principle | Where |
|-----------|-------|
| **S**ingle responsibility | Ingestion (write side) vs `SourceService` (read side); chunkers only split; enrichers only change embedded text; `ChatService` orchestrates but delegates retrieval. |
| **O**pen/closed | `core/registry.py`: every strategy (parser, chunker, embedder, store, fusion, reranker, query transform, LLM) is looked up by name. New ones register themselves — no edits to existing code. Entry-point plugins (`rag_service.plugins`) and the `plugins:` config key load external components. |
| **L**iskov substitution | One behavioural contract suite (`tests/integration/test_store_contract.py`) runs unchanged against memory, embedded Qdrant, Qdrant server and pgvector. |
| **I**nterface segregation | `VectorStore` = `CollectionAdmin` + `VectorWriter` + `VectorReader`. Ingestion depends on the writer, retrieval on the reader, collections on the admin port. |
| **D**ependency inversion | Services receive ports in their constructors; only `container.py` knows concrete classes. Tests swap in a counting embedder, a fake LLM, an in-memory store. |

Other patterns: **Factory** + component cache (`PipelineFactory`), **Decorator** (`CachedDenseEmbedder`), **Template method** (`BaseChunker.split`), **Strategy** (fusion, rerank, query transform), **Unit of work** (registry `save` replaces a source and its chunk refs atomically), **Optimistic concurrency** (source `version`).

## Request flows

### Ingestion (sync mode)

```mermaid
sequenceDiagram
  participant C as Client
  participant API
  participant J as JobService
  participant I as IngestionService
  participant R as Registry (SQL)
  participant E as Embedders
  participant V as Vector store
  C->>API: POST /collections/{c}/documents
  API->>J: ingest(docs, mode=sync)
  J->>R: create job (queued→running)
  loop each document (bounded concurrency)
    J->>I: sync_source(doc)
    I->>R: get source + chunk refs
    alt unchanged hashes
      I-->>J: unchanged
    else metadata only
      I->>V: update_payloads
      I->>R: save(version+1)
    else content / config changed
      I->>I: parse → chunk → identify (hash IDs) → diff
      I->>E: embed only new chunks
      I->>V: upsert new, patch moved
      I->>R: save(version+1, chunk refs)
      I->>V: delete stale chunk IDs
    end
  end
  J->>R: finish job (succeeded / partial / failed)
  API-->>C: 200 / 207 / 422 + per-source results
```

### Retrieval

```mermaid
flowchart LR
  Q[query] --> T[query transform<br/>none·multi_query·hyde·rewrite]
  T --> DN[dense ANN<br/>HNSW]
  T --> SP[sparse<br/>inverted index]
  DN --> F[fusion<br/>rrf·dbsf·rsf, weighted]
  SP --> F
  F --> DD[dedupe] --> RR[rerank<br/>cross-encoder] --> MM[MMR] --> TH[min_score] --> K[top_k] --> CE[context expansion]
```

### Chat

`prepare` (validate → condense follow-up → retrieve → pack `<source id="n">` blocks within the
token budget) runs **before** a stream is opened so request errors keep proper HTTP statuses;
`stream` then emits `retrieval → delta* → done` events. No sources ⇒ `no_context_answer`
without an LLM call.

## Data model

```mermaid
erDiagram
  rag_collections ||--o{ rag_sources : contains
  rag_sources ||--o{ rag_chunks : "current version"
  rag_collections ||--o{ rag_jobs : runs
  rag_collections {
    string name PK
    json pipeline_overrides
    string embedding_fingerprint
    bool managed
  }
  rag_sources {
    bigint id PK
    string collection
    string source_id "client processing ID"
    string content_hash
    string metadata_hash
    string pipeline_fingerprint
    int version "optimistic lock"
    string status "active|failed"
    text content "for reindex"
  }
  rag_chunks {
    string chunk_id PK "uuid5"
    string chunk_hash
    int chunk_index
    json chunk_metadata
    json metadata_overrides "PATCH /chunks/{id}/metadata"
  }
  rag_jobs {
    string id PK
    string type "ingest|delete|reindex"
    string status
    json results
  }
```

Vector point payload (every store):

```json
{
  "text": "chunk text returned to clients",
  "source_id": "kb:fans:fv-20vq3",
  "chunk_index": 3,
  "chunk_hash": "…",
  "title": "Panasonic FV-20VQ3",
  "mime_type": "text/markdown",
  "metadata": {"brand": "Panasonic", "page": 2, "section": "Specs > Noise", "…": "…"}
}
```

## Package map

```
src/rag_service/
├── core/          config.py (settings schema + loader) · errors.py · logging.py · registry.py · utils.py
├── domain/        models.py · filters.py (DSL + reference semantics) · ports.py (Protocols)
├── adapters/
│   ├── parsers/       text, markdown, html, json, csv, pdf
│   ├── chunking/      length functions; fixed/recursive/sentence/markdown/semantic
│   ├── enrichment/    contextual headers, LLM contextual retrieval
│   ├── embedding/     hash, lexical (offline) · fastembed dense/sparse · OpenAI-compatible · LRU cache
│   ├── vectorstores/  memory · qdrant · pgvector
│   ├── fusion/        rrf · dbsf · rsf · mmr
│   ├── rerank/        fastembed cross-encoder · Cohere-compatible HTTP
│   ├── query/         multi_query · hyde · rewrite
│   ├── llm/           anthropic · openai-compatible · gemini · claude-lite-llm (subscription CLI) · extractive
│   ├── persistence/   SQL registry (SQLAlchemy async)
│   └── jobs/          in-process runner, keyed locks
├── application/   pipeline · collections · ingestion · retrieval · chat · jobs · sources · evaluation
├── api/           app factory · middleware · security · errors · deps · SSE · routers · OpenAI facade
└── container.py   composition root
```
