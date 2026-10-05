# Advanced RAG Service

A configurable, enterprise-grade **Retrieval-Augmented Generation micro-service** (FastAPI).
Every stage — parsing, chunking, enrichment, embedding, indexing, retrieval, fusion, re-ranking and
generation — is a pluggable component selected by configuration.

* **On-demand ingestion over HTTP** — text, Markdown, HTML, CSV, JSON, PDF (base64) or your own
  pre-chunked content; sync or async jobs.
* **Update / delete by processing ID** — re-send a `source_id` and only changed chunks are
  re-embedded, stale chunks are removed, metadata-only changes patch payloads without re-embedding;
  delete removes every chunk.
* **Metadata everywhere** — document, per-chunk (client supplied or `PATCH`ed) and system metadata
  (page, section); a store-agnostic filter DSL.
* **Hybrid retrieval** — dense (BGE, OpenAI-compatible…) + sparse (SPLADE, BM25) with RRF / DBSF /
  RSF fusion, weights, query transforms (multi-query, HyDE, rewrite), cross-encoder re-ranking, MMR,
  context expansion, contextual chunk headers and LLM contextual retrieval.
* **Multiple databases** — Qdrant, PostgreSQL/pgvector, in-memory; one contract test suite.
* **Catalog and multi-tenant search** — result grouping (one hit per product family or page),
  facets as filters, and tenant-partitioned collections (Qdrant `is_tenant`) for many sites.
* **Pluggable LLMs** — Anthropic Claude, OpenAI-compatible servers, Google Gemini, an offline
  extractive mode, and `claude_lite_llm` for local development on your own Claude subscription
  without an API key ([ADR 0006](doc/adr/0006-subscription-claude-via-claude-code-cli.md)).
* **Chat-ready responses** — grounded answers with citations as JSON or Server-Sent Events, plus an
  **OpenAI-compatible** `/v1/chat/completions` (collections appear as models).
* **Raw RAG results API** with per-retriever debug, and an **evaluation API** (Recall, MRR, nDCG).
* **Enterprise standards** — scoped API keys, collection ACLs, rate limiting, RFC 9457 errors,
  request IDs, JSON logs, Prometheus metrics, OpenTelemetry, health probes, Alembic migrations,
  strict typing, CI, non-root container.

## Quick start

```bash
make install                      # uv sync --all-extras --group dev + pre-commit
make run                          # offline stack on http://localhost:8080  (docs: /docs)
make smoke                        # ingest sample catalog → retrieve → chat → re-sync → evaluate
```

Production-like stack (Qdrant + Postgres + BGE + SPLADE + reranker):

```bash
make up                           # docker compose: API :8080, Qdrant :6333, Postgres :5432
RAG_URL=http://localhost:8080 make smoke
```

```bash
curl -X POST localhost:8080/api/v1/collections/products/documents \
  -H 'Content-Type: application/json' -d @examples/ingest_products.json
curl -X POST localhost:8080/api/v1/collections/products/retrieve \
  -H 'Content-Type: application/json' -d '{"query": "revent 80 cfm", "top_k": 3}'
curl -X POST localhost:8080/api/v1/collections/products/chat \
  -H 'Content-Type: application/json' -d '{"messages": [{"role": "user", "content": "Which quiet fan prevents mold?"}]}'
```

## Configuration in one minute

```bash
RAG_CONFIG_FILE=config/rag.yaml,config/profiles/qdrant-hybrid.yaml   # layered YAML profiles
RAG__DEFAULTS__RETRIEVAL__TOP_K=8                                   # any key via env
```

```yaml
collections:
  products:
    pipeline:
      chunking:  {strategy: markdown, chunk_size: 256, chunk_overlap: 32}
      embedding: {dense: {provider: fastembed, model: BAAI/bge-small-en-v1.5},
                  sparse: {provider: fastembed, model: prithivida/Splade_PP_en_v1}}
      retrieval: {mode: hybrid, fusion: {method: rrf, weights: {dense: 1, sparse: 1.2}},
                  rerank: {enabled: true}}
```

Retrieval settings can also be overridden per request. See [doc/02-configuration.md](doc/02-configuration.md).

## Documentation

Everything is in [`doc/`](doc/README.md): architecture, configuration, ingestion & sync semantics,
retrieval tuning, API reference, vector stores, operations, extending, testing and ADRs.

## Cookbook: configurations for real use cases

[`cookbook/`](cookbook/README.md) holds runnable lessons. Each has its configuration, data and a
scenario whose every promise is checked:

* an Amazon-style catalog with near-identical product names, variants and facets;
* a CMS hosting many sites in one collection;
* a support helpdesk;
* technical documentation;
* long documents.

```bash
uv run python cookbook/run.py --list
uv run python cookbook/run.py ecommerce -v                  # offline, seconds
uv run python cookbook/run.py cms --stack models --llm anthropic
```

## Development

```bash
make lint          # ruff format --check, ruff check, mypy --strict
make test          # unit + API + store contract (offline, seconds)
make it-up test-it # contract suite against real Qdrant + pgvector (Docker)
make test-models   # end-to-end with real FastEmbed models
```

AI-assisted development is set up for GitHub Copilot (`.github/copilot-instructions.md`,
path-scoped instructions, prompts, custom agents, skills) and Claude Code (`CLAUDE.md`,
`.claude/`).

## Project layout

```
config/            base config + profiles (qdrant-hybrid, pgvector, production, llm-*)
cookbook/          runnable use-case recipes (config + data + checked scenario) and their runner
doc/               technical documentation and ADRs
examples/          sample catalog, retrieval, chat and evaluation payloads (synthetic data)
migrations/        Alembic migrations for the source registry
scripts/smoke.py   end-to-end HTTP smoke / demo client
src/rag_service/   core · domain · adapters · application · api · container.py
tests/             unit · api · integration (store contract) · e2e (real models)
```
