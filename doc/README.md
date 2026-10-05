# Advanced RAG Service — technical documentation

| # | Document | For |
|---|----------|-----|
| 00 | [Implementation plan](00-implementation-plan.md) | Scope, requirements → design mapping, work breakdown |
| 01 | [Architecture](01-architecture.md) | Layers, SOLID mapping, request flows, data model |
| 02 | [Configuration reference](02-configuration.md) | Every setting, layering, profiles, overrides |
| 03 | [Ingestion & source sync](03-ingestion-and-sync.md) | Processing IDs, update/delete semantics, metadata, reindex |
| 04 | [Retrieval techniques & tuning](04-retrieval-and-tuning.md) | Hybrid search, fusion, rerank, MMR, HyDE…, evaluation workflow |
| 05 | [API reference](05-api-reference.md) | Endpoints, envelopes, errors, filters, streaming, OpenAI facade |
| 06 | [Vector stores](06-vector-stores.md) | Qdrant, pgvector, memory; sizing; model migration |
| 07 | [Operations](07-operations.md) | Deployment, scaling, observability, security, runbook |
| 08 | [Extending](08-extending.md) | Adding chunkers, embedders, stores, rerankers, LLMs, plugins |
| 09 | [Testing](09-testing.md) | Suites, quality gates, how to write tests |
| ADR | [Architecture decisions](adr/) | Why the key choices were made |
| 🍳 | [Cookbook](../cookbook/README.md) | Runnable, checked configurations for real use cases (e-commerce, multi-site CMS, helpdesk, docs, long documents) |

Background material from the Hybrid Search & Retrieval class (slides, notebook, video summary) is
kept at the repository root; `04-retrieval-and-tuning.md` maps its concepts (dense vs sparse,
HNSW, inverted index, SPLADE, RRF) to configuration switches in this service.
