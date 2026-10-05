# Cookbook — the service configured for real use cases

Each recipe is a **lesson you can run**:

* a `README.md` that explains the problem and why each setting is there;
* a `config.yaml` overlay — the configuration you would deploy;
* `data/` — synthetic but realistic content;
* a `scenario.yaml` — ingest, search, update, delete, evaluate and chat steps, each with the
  result it promises.

`run.py` composes the configuration, starts the service in process, plays the scenario and checks
every promise. The same recipes pass in CI (`tests/cookbook/`), so they cannot silently drift from
the product.

| # | Recipe | What it teaches |
|---|---|---|
| 01 | [Quickstart](recipes/01-quickstart/) | Collection definition, ingest → search → chat → update → delete, incremental re-sync |
| 02 | [E-commerce: similar product names](recipes/02-ecommerce-similar-products/) | "X9 Pro" vs "X9 Pro Max", model numbers, accessories that name the product, variants, facets, measuring dense vs sparse vs hybrid |
| 03 | [CMS with many sites](recipes/03-cms-multi-site/) | Multi-tenancy (one collection, tenant index), locales, HTML pages, publish/update/unpublish/remove-site workflow, site-scoped chat |
| 04 | [Support helpdesk](recipes/04-support-helpdesk/) | Pre-chunked Q&A with chunk metadata, retiring answers, conversations, OpenAI-compatible endpoint, off-topic fallback |
| 05 | [Technical docs](recipes/05-technical-docs/) | Error codes and config keys, code blocks kept whole, heading paths, changelog filtering |
| 06 | [Long documents](recipes/06-long-documents/) | Choosing a chunker (fixed vs recursive vs semantic), small-to-big context, LLM contextual retrieval |

## Run it

```bash
make install                                         # once: all extras + dev tools

uv run python cookbook/run.py --list                 # recipes, stacks, LLM overlays
uv run python cookbook/run.py                        # every recipe, offline stack (seconds, no downloads)
uv run python cookbook/run.py ecommerce -v           # one recipe; -v prints the ranked results of every search
uv run python cookbook/run.py ecommerce --stack models               # real models (BGE + SPLADE + cross-encoder)
uv run python cookbook/run.py cms --stack qdrant --llm anthropic     # Qdrant server + Claude
uv run python cookbook/run.py cms --serve            # serve that configuration on :8080 — explore /docs
uv run python cookbook/run.py cms --base-url http://localhost:8080   # check a service you started yourself
make cookbook                                        # = every recipe on the offline stack
```

Recipe names match by fragment (`cms`, `02`, `long`). The exit code is non-zero when a check fails,
so the runner also works as a smoke test for a deployment started with the same configuration
(`--base-url … --api-key …`).

## How configuration is layered

Every run is a stack of YAML files, lowest priority first, then environment variables, then
per-request overrides:

```
config/rag.yaml                      base: defaults for every collection
  + stack                            where vectors live, which embedding and reranking models
  + recipes/<recipe>/config.yaml     the use case: collections, chunking, retrieval, prompts
  + LLM overlay                      who writes the answers
  + RAG__SECTION__KEY=value          environment overrides (12-factor)
  + request body                     `retrieval` / `generation` overrides, evaluate `variants`
```

`run.py` prints the exact `RAG_CONFIG_FILE` it uses (`--serve`), so any combination you check here
is one environment variable away from a deployment.

| `--stack` | Layers | Needs |
|---|---|---|
| `offline` (default) | — | nothing: in-memory store, hashing dense + BM25-style sparse embedders |
| `models` | `config/profiles/qdrant-hybrid.yaml` + `stacks/memory-store.yaml` | the models (~0.7 GB, downloaded once to `data/models`) |
| `qdrant` | `config/profiles/qdrant-hybrid.yaml` | a Qdrant server at `$QDRANT_URL` (`make it-up` starts one on :6399) |
| `pgvector` | `config/profiles/pgvector.yaml` + `stacks/reranker.yaml` | PostgreSQL with pgvector at `$PG_DSN` and `$REGISTRY_URL` |

| `--llm` | Layer | Needs |
|---|---|---|
| `extractive` (default) | — | nothing: quotes the best-matching sentences, with citations |
| `anthropic` | `config/profiles/llm-anthropic.yaml` | `ANTHROPIC_API_KEY` |
| `openai` | `config/profiles/llm-openai.yaml` | `OPENAI_API_KEY`, `OPENAI_MODEL` (any OpenAI-compatible server) |
| `gemini` | `config/profiles/llm-gemini.yaml` | `GEMINI_API_KEY` |
| `claude-lite` | `config/profiles/llm-claude-lite.yaml` | Claude Code logged in with your own subscription — personal use only ([ADR 0006](../doc/adr/0006-subscription-claude-via-claude-code-cli.md)) |

Steps marked `when: {llm: generative}` (query rewriting, multi-query, contextual retrieval) run only
with an LLM that writes; with `extractive` they are skipped and the service itself skips those
features with a warning.

In-process runs use a throwaway registry and their own namespace in shared databases (Qdrant collection
prefix `cookbook_`, PostgreSQL schema `cookbook`), and empty those collections first, so runs are
repeatable and never touch a real deployment on the same server. With `--base-url` the runner stops if
a recipe collection already holds documents, unless you pass `--reset`.

## Results

Every recipe passes on every stack (2026-10-02). Highlights — details and the full tables are in each
recipe's README:

| Measurement | offline | models (BGE + SPLADE + MiniLM) |
|---|---|---|
| 02 · "nimbus x9 pro max" rank 1 | the phone's **case** (lexical trap) | the phone (cross-encoder) |
| 02 · shopper-language queries, nDCG@5 hybrid → + cross-encoder | 0.68 | 0.87 → 0.93 |
| 03 · site-scoped queries, nDCG@5 hybrid | 0.91 | 0.99 |
| 05 · identifiers + how-to, nDCG@5 dense / sparse / hybrid | 0.78 / 0.96 / 0.93 | 0.94 / 0.99 / 0.99 |
| 06 · answer inside the top chunk: fixed / recursive / semantic | 12 / 14 / 15 of 15 | 15 / 15 / 15 of 15 |

The offline stack is for development and CI; its numbers show *mechanics*, not production quality.

## Write your own recipe

1. Create `recipes/NN-name/` with `config.yaml` (your collections), `data/*.json` (ingest payloads,
   the same JSON the API accepts) and `scenario.yaml`.
2. Run `uv run python cookbook/run.py NN -v` until every step passes; it is picked up by
   `tests/cookbook` automatically.

`scenario.yaml` reference:

```yaml
title: "NN · Title"
summary: one line for --list
collection: default-collection-for-steps
steps:
  - name: Human-readable promise            # optional; shown in the report
    collection: other                       # optional per-step collection
    when: {stacks: [models, qdrant], llm: generative}   # optional; otherwise skipped
    <one action>: …
    expect: {…}
```

| Action | Value | `expect` keys |
|---|---|---|
| `ingest` | `file` or `{file, mode: sync\|async}` | `chunks_min` |
| `upsert` | `{source_id, file}` (one document body, `PUT`) | `action`, any result field, `<field>_min` / `<field>_max` |
| `patch_metadata` | `{source_id, metadata}` | — (fails if anything was re-embedded) |
| `patch_chunk` | `{source_id, chunk_index, metadata}` | — |
| `delete` | `[source_id, …]` or `{file}` (every document in a payload file) | — |
| `search` | a `/retrieve` body: `{query, filter, top_k, retrieval}` | `first`, `includes` + `within`, `excludes`, `every`, `distinct`, `distinct_min`, `max_per`, `count`, `first_text_contains`, `any_text_contains`, `first_metadata`, `balanced`, `has_context` |
| `facts` | `file` or `{file, within, use_context, retrieval}` — `[{query, contains}]` | `found_min` |
| `evaluate` | `file` or `{file, variants}` — an `/evaluate` body | `variant` + `ndcg_min` / `mrr_min` / `recall_min` / `hit_rate_min`, `not_worse_than`, `better_than` |
| `chat` | a `/chat` body, or `{question, filter, retrieval}` | `citations_min`, `cited_every`, `finish_reason`, `answer_contains` |
| `openai_chat` | `{messages}` → `/v1/chat/completions` with `model: rag:<collection>` | — |
| `create_collection` / `drop_collection` | a `POST /collections` body / a name | — |

Rules in `every`, `cited_every` and `first_metadata` use the same shapes as metadata filters: a value
(equals), a list (one of), or operators `{eq, ne, lt, lte, gt, gte, in, exists, contains}`. Keys are
metadata fields, or `source_id` / `title` / `text`.
