---
name: rag-pipeline-config
description: Design and write configuration for the Advanced RAG Service — YAML profiles, per-collection pipelines (chunking, enrichment, embedding, hybrid retrieval, rerank, generation), vector store and LLM provider selection, security and environment overrides. Use when setting up a new deployment, a new collection, or changing retrieval behaviour through config.
argument-hint: "[use case / collection]"
---

# Configure a RAG pipeline

Reference: `doc/02-configuration.md` (every key). Profiles: `config/rag.yaml` +
`config/profiles/*.yaml`, layered with `RAG_CONFIG_FILE=a.yaml,b.yaml`; env overrides
`RAG__SECTION__KEY`.

## Decide by use case

| Content / queries | Chunking | Embedding | Retrieval |
|---|---|---|---|
| Product catalog, SKUs, specs | `markdown`, one chunk per SKU, 256 tok | BGE-small + SPLADE | hybrid (1:1 measured best in cookbook 02), rerank on, `group_by: family`, facets as payload indexes |
| Many sites / tenants (CMS) | `markdown`, 200 tok | BGE (multilingual model for non-English) | one shared collection, `{field: site, tenant: true}`, always filter by site, `group_by: source_id` + `group_size: 2` |
| Docs / wiki with headings | `markdown`, 384 tok, overlap 48 | BGE-small/large + SPLADE | hybrid, rerank, `context_expansion` window 1 |
| Long reports / PDFs | `recursive` or `semantic`, 512 tok | BGE-large + SPLADE; `llm_context` if budget allows | hybrid, rerank, MMR |
| FAQ / Q&A pairs | pre-chunked (one pair per chunk) | BGE-small + BM25 | hybrid, rerank |
| Code, logs, error messages | `fixed` or `recursive` with small chunks | sparse-heavy (BM25/SPLADE) | `mode: sparse` or hybrid with `weights.sparse` 2 |
| Conversational support content | `sentence`, 256 tok | strong dense | hybrid + `query_transform: rewrite`, chat `condense_history` |

Every row above has a runnable, measured example in `cookbook/recipes/` (02 catalog, 03 CMS, 04
helpdesk/FAQ, 05 docs, 06 long documents). Start from the closest recipe's `config.yaml` and run
`uv run python cookbook/run.py <recipe> -v` to see its behaviour.

## Template: new collection

```yaml
collections:
  <name>:
    description: <what it holds>
    payload_indexes: [{field: <filter field>, type: keyword}]
    pipeline:
      chunking: {strategy: markdown, chunk_size: 384, chunk_overlap: 48}
      embedding:
        dense: {provider: fastembed, model: BAAI/bge-small-en-v1.5,
                query_prefix: "Represent this sentence for searching relevant passages: "}
        sparse: {provider: fastembed, model: prithivida/Splade_PP_en_v1}
      retrieval:
        mode: hybrid
        top_k: 6
        candidate_k: 50
        fusion: {method: rrf, k: 60, weights: {dense: 1.0, sparse: 1.0}}
        rerank: {enabled: true, provider: fastembed, model: Xenova/ms-marco-MiniLM-L-6-v2, candidates: 30}
      generation: {max_context_tokens: 6000}
```

Or create it at runtime: `POST /api/v1/collections {"name", "pipeline": {...}}` (admin).

## Rules

- Embedding changes invalidate vectors → the collection becomes stale until
  `POST /collections/{c}/reindex?recreate=true`. Chunking/enrichment changes need a plain reindex.
- `pgvector` cannot apply IDF → use SPLADE, not `Qdrant/bm25`/`lexical`, for its sparse side.
- LLM: `provider: anthropic` (default model `claude-opus-5`, secrets via `${ANTHROPIC_API_KEY}`),
  `provider: openai` with `base_url` for any OpenAI-compatible server, or `provider: gemini`
  (`model: gemini-3.8-flash`, `${GEMINI_API_KEY}`; Vertex AI via `options`). Leave `temperature`
  unset for current Claude models.
- `provider: claude_lite_llm` (overlay `config/profiles/llm-claude-lite.yaml`) uses the person's own
  Claude subscription through the Claude Code CLI — offer it only for local/personal setups, never
  for deployments that serve other users (ADR 0006).
- Secrets only through `${ENV}` references; API keys as `key_sha256`.
- Validate: start the service (config errors fail fast with the offending key), then check
  `GET /api/v1/collections/<name>` and run the `rag-retrieval-tuning` skill.
