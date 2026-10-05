# Implementation Plan — Advanced RAG Service

> Status: **executed** (2026-09-27). This document is the plan the service was built against. It is kept as
> the record of scope and design intent; the other documents in `doc/` describe the system as built.
>
> Verification: unit, API, store-contract (memory, embedded Qdrant, Qdrant server, pgvector) and real-model
> e2e tests (BGE + SPLADE + cross-encoder); `ruff` and `mypy --strict` clean; Docker image smoke-tested
> against Qdrant + PostgreSQL.

## 1. Goal

A self-contained, configurable **Retrieval-Augmented Generation micro-service** that any team can deploy
and call over HTTP. Every stage of the RAG pipeline (parsing, chunking, enrichment, embedding,
indexing, retrieval, fusion, re-ranking, generation) is a pluggable component selected by configuration,
not by code changes.

### Functional requirements

| # | Requirement | How it is met |
|---|-------------|---------------|
| F1 | Configurable chunking | `chunking.strategy`: `fixed`, `recursive`, `sentence`, `markdown`, `semantic`; size unit `chars` / `words` / `tokens` |
| F2 | Configurable embedding | Dense: `fastembed`, `openai` (any OpenAI-compatible endpoint), `hash` (offline). Sparse: `fastembed` (SPLADE, BM25, BM42), `lexical` (offline BM25-style) |
| F3 | Configurable indexing | Qdrant (HNSW + sparse inverted index, IDF modifier, quantization, payload indexes), pgvector (HNSW on `vector` and `sparsevec`), in-memory |
| F4 | Configurable retrieval | `dense` / `sparse` / `hybrid`; fusion `rrf` / `dbsf` / `rsf` with per-modality weights; query transforms (`multi_query`, `hyde`, `rewrite`); cross-encoder re-rank; MMR; context expansion; score thresholds; metadata filters |
| F5 | Multiple databases | `vector_store.provider`: `qdrant`, `pgvector`, `memory` behind one `VectorStore` port |
| F6 | On-demand ingestion over HTTP | `POST /api/v1/collections/{c}/documents` (sync or async job), raw text, base64 files (PDF, HTML, Markdown, JSON, CSV, text) or pre-chunked input |
| F7 | Chat-friendly standard responses | Native `/chat` (JSON + SSE) and an **OpenAI-compatible** `/v1/chat/completions` with citations |
| F8 | Raw RAG results API | `POST /api/v1/collections/{c}/retrieve` returns scored chunks, score breakdown and optional per-stage debug |
| F9 | Chunk metadata | Document-level metadata on every chunk; client-supplied per-chunk metadata (pre-chunked input); `PATCH` metadata on a single chunk; metadata-only updates without re-embedding |
| F10 | Update / delete by processing ID | The client's `source_id` is the stable processing ID. Re-sending it re-syncs: unchanged content is skipped, changed chunks are re-embedded, stale chunks are deleted, deleting the source deletes every chunk |
| F11 | Tuning and evaluation | `POST /collections/{c}/evaluate` computes Recall@k, MRR, nDCG, Hit-rate across retrieval variants (dense vs sparse vs hybrid vs rerank) |

### Non-functional requirements (enterprise standards)

- **SOLID / hexagonal architecture**: domain ports (`typing.Protocol`), adapters registered in plugin registries, a single composition root.
- **12-factor configuration**: YAML profiles + `${ENV}` interpolation + `RAG__SECTION__KEY` environment overrides, validated by Pydantic.
- **Security**: API keys (stored as SHA-256 hashes) with scopes (`query`, `ingest`, `admin`) and per-key collection allow-lists; request size limits; security headers; CORS; rate limiting; SSRF-safe callbacks.
- **Observability**: structured JSON logs with request correlation IDs, Prometheus metrics, optional OpenTelemetry tracing, liveness/readiness probes.
- **Reliability**: idempotent ingestion (content hashing and deterministic chunk IDs), per-source locking, optimistic concurrency on the registry, durable job records, graceful shutdown.
- **API standards**: versioned routes (`/api/v1`), a consistent success envelope, RFC 9457 `application/problem+json` errors, pagination metadata, OpenAPI docs with examples.
- **Quality gates**: pytest (unit + API + integration), ruff, mypy, pre-commit, GitHub Actions CI, multi-stage non-root Docker image.
- **AI-assisted development**: Copilot instructions, path-scoped instructions, reusable prompts, custom agents and agent skills (shared with Claude Code).

## 2. Architecture (summary)

```
              ┌──────────────────────── API layer (FastAPI) ────────────────────────┐
              │ routers · schemas · auth · middleware · problem+json · SSE           │
              └───────────────┬──────────────────────────────────────┬──────────────┘
                              │ depends on (services only)           │
              ┌───────────────▼────────── Application services ──────▼──────────────┐
              │ IngestionService · RetrievalService · ChatService · CollectionService│
              │ SourceService · JobService · EvaluationService · PipelineFactory     │
              └───────────────┬──────────────────────────────────────────────────────┘
                              │ depends on (ports only)
              ┌───────────────▼────────────── Domain ────────────────────────────────┐
              │ models (Chunk, SourceRecord, RetrievedChunk…) · FilterSpec · ports    │
              └───────────────▲──────────────────────────────────────────────────────┘
                              │ implemented by
              ┌───────────────┴────────────── Adapters ──────────────────────────────┐
              │ parsers · chunkers · enrichers · embedders · vector stores · fusion  │
              │ rerankers · query transforms · LLM clients · SQL registry · job runner│
              └──────────────────────────────────────────────────────────────────────┘
```

Full detail: [01-architecture.md](01-architecture.md).

## 3. Key design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Extensibility | Named plugin registries plus Python entry points (`rag_service.plugins`) | Open/Closed: new stores or chunkers need no core edits |
| Fusion location | Client-side fusion by default, Qdrant native (`prefetch` + RRF/DBSF) optional | Identical, debuggable semantics on every store; native path for latency |
| Source tracking | SQL **source registry** (SQLite for dev, Postgres for prod) | Enables diff-based re-sync, metadata-only updates, reindex from stored content, job history |
| Chunk identity | `uuid5(collection/source_id/chunk_hash/occurrence)` | Deterministic, so upserts are idempotent and unchanged chunks are never re-embedded |
| Offline-first defaults | `hash` dense + `lexical` sparse + `memory` store + `extractive` LLM | Tests and local development need no model downloads or API keys |
| Response format | `{data, meta}` envelope on success, RFC 9457 problem details on error | Predictable for clients; standards-based errors |
| Chat compatibility | OpenAI-compatible endpoint in addition to the native one | Any OpenAI client or UI (Open WebUI, LibreChat, SDKs) can use the service unchanged |
| Background work | `JobRunner` port with a bounded in-process asyncio runner; jobs persisted in SQL | Simple to operate; can be swapped for a distributed queue |

Architecture Decision Records live in [`doc/adr/`](adr/).

## 4. Work breakdown (execution order)

1. **Scaffold**: `pyproject.toml` (uv, extras), ruff/mypy/pytest config, `.gitignore`, `.env.example`, Makefile.
2. **Core**: settings and YAML loader with env interpolation, structured logging, error hierarchy, generic registry, hashing and ID utilities.
3. **Domain**: dataclass models, `FilterSpec` DSL, ports (`Protocol`s) with segregated interfaces.
4. **Adapters**:
   - parsers (text, markdown, HTML, JSON, CSV, PDF)
   - length functions and chunkers (fixed, recursive, sentence, markdown, semantic)
   - enrichers (contextual header, LLM contextual retrieval)
   - embedders (hash, lexical, fastembed dense/sparse, OpenAI-compatible)
   - vector stores (memory, Qdrant, pgvector) with filter translators
   - fusion (RRF, DBSF, RSF), rerankers (fastembed cross-encoder, HTTP/Cohere-compatible), MMR
   - query transforms (multi-query, HyDE, rewrite), LLM clients (Anthropic, OpenAI-compatible, extractive)
   - SQL registry repositories, in-process job runner, keyed locks
5. **Application services**: pipeline factory with component cache, ingestion sync algorithm, retrieval pipeline, chat (sync + streaming with citations), collections, sources, jobs, evaluation.
6. **API**: app factory with lifespan, middleware (request ID, access log, security headers, body limit, rate limit), auth and scopes, routers, OpenAI-compatible router, health and metrics.
7. **Tests**: unit (chunkers, fusion, filters, hashing, sync diff), API tests on the offline stack, integration tests against Qdrant and pgvector in Docker (opt-in).
8. **Delivery**: Dockerfile, docker-compose (API + Qdrant + Postgres/pgvector), CI workflow, sample data and evaluation set.
9. **Documentation**: this `doc/` folder, root README, ADRs.
10. **AI tooling**: `.github/copilot-instructions.md`, path-scoped instructions, prompts, custom agents, skills; `CLAUDE.md` and `.claude/` equivalents.

## 5. Out of scope (documented extension points)

- Distributed job queue (Redis/Arq/Celery): implement the `JobRunner` port.
- Zero-downtime re-embedding via collection aliases (blue/green): see [06-vector-stores.md](06-vector-stores.md).
- OIDC/JWT auth: implement the `Authenticator` in `api/security.py`.
- Remote URL ingestion: deliberately omitted to avoid SSRF; clients send content.
- Multi-collection federated search: fusion is generic; add a service method when needed.

## 6. Addendum — 2026-10-02: additional LLM providers

* `provider: gemini` — Google Gemini through the official `google-genai` SDK (extra `gemini`),
  Developer API or Vertex AI.
* `provider: claude_lite_llm` — Claude through the Claude Code CLI on the operator's own
  subscription (extra `claude-lite`, package `claude-lite-llm`), for personal and development use
  without an API key. Hardening and the terms that limit its use: [ADR 0006](adr/0006-subscription-claude-via-claude-code-cli.md).
* LLM clients are now created at startup so provider misconfiguration fails fast.

## 7. Addendum — 2026-10-02: cookbook

* `cookbook/` holds six runnable recipes, each a lesson README plus a configuration overlay, data and
  a checked scenario. `cookbook/run.py` layers base → stack → recipe → LLM and checks every step;
  `tests/cookbook` runs them in CI. Every recipe passes on the offline, models, Qdrant and pgvector
  stacks.
* Product changes the use cases needed:
  - `retrieval.group_by` / `group_size` (product variants, chunks per page);
  - `payload_indexes[].tenant` (Qdrant `is_tenant` index for multi-site collections);
  - LLM-only features (query transforms, contextual enrichment) are skipped with a warning when the
    LLM cannot write;
  - unknown top-level configuration keys now fail (except `x-` anchor keys).
