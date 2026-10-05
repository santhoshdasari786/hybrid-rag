# 02 · Configuration reference

## How configuration is resolved

Highest precedence first:

1. **Environment variables** `RAG__<SECTION>__<KEY>` — e.g. `RAG__VECTOR_STORE__PROVIDER=qdrant`,
   `RAG__DEFAULTS__RETRIEVAL__TOP_K=8`.
2. **`.env`** file (same names).
3. **YAML files** listed in `RAG_CONFIG_FILE`, comma-separated and deep-merged left → right:
   `RAG_CONFIG_FILE=config/rag.yaml,config/profiles/qdrant-hybrid.yaml,config/profiles/production.yaml`.
   Any YAML value may reference `${ENV_VAR}` or `${ENV_VAR:-default}`; a required variable that is
   unset stops startup with a clear error.
4. **Built-in defaults** (offline-capable: memory store, hash + lexical embedders, extractive LLM).

Unknown keys are rejected at every level, including the top level, so typos fail fast. Top-level keys
starting with `x-` are the one exception: they hold YAML anchors shared within a file (as in
docker-compose), e.g. `x-retrieval: &retrieval {top_k: 4}` reused with `retrieval: {<<: *retrieval}`.
`GET /api/v1/system/config` (admin) shows the effective configuration with secrets masked.

### Pipeline configuration levels

```
defaults (service-wide)  →  collections.<name>.pipeline (partial)  →  request overrides (retrieval / generation / chunking)
```

| Can be overridden per… | chunking | enrichment | embedding | retrieval | generation |
|---|---|---|---|---|---|
| collection | ✅ | ✅ | ✅ (requires reindex `recreate=true`) | ✅ | ✅ |
| document (ingest request `chunking`) | ✅ | – | – | – | – |
| request (retrieve / chat) | – | – | – | ✅ | ✅ (chat) |

## Profiles shipped in `config/`

| File | Purpose |
|------|---------|
| `rag.yaml` | Base: offline defaults, `default` and `products` collections |
| `profiles/qdrant-hybrid.yaml` | Qdrant + BGE-small dense + SPLADE++ sparse + MiniLM cross-encoder rerank |
| `profiles/pgvector.yaml` | PostgreSQL for vectors (HNSW + sparsevec) and the registry |
| `profiles/production.yaml` | Auth + scopes, rate limit, JSON logs, Postgres registry, Claude (`claude-opus-5`) |
| `profiles/llm-anthropic.yaml` · `llm-openai.yaml` · `llm-gemini.yaml` · `llm-claude-lite.yaml` | LLM overlays to layer on any stack |

Use-case configurations (e-commerce catalog, multi-site CMS, helpdesk, technical docs, long documents),
each runnable and checked, live in [`cookbook/`](../cookbook/README.md).

## Sections

### `app`
| Key | Default | Notes |
|-----|---------|-------|
| `name` | `advanced-rag-service` | Service name in logs / OpenAPI |
| `environment` | `local` | `local`, `dev`, `test`, `staging`, `prod` |
| `api_prefix` | `/api/v1` | |
| `docs_enabled` | `true` | Swagger `/docs`, ReDoc `/redoc`, `/openapi.json` |
| `openai_compat_enabled` | `true` | `/v1/models`, `/v1/chat/completions` |
| `root_path` | `""` | When served behind a path-prefixing proxy |
| `warmup_models` | `true` | Load models at startup; readiness waits for it |

### `server`
`host` (`0.0.0.0`), `port` (`8080`), `workers` (`1`), `graceful_shutdown_s` (`30` — in-flight jobs
are drained for this long), `gzip_min_bytes` (`1024`; SSE is never compressed).

### `logging` / `observability`
`logging.level`, `logging.format` (`json` | `console`), `logging.access_log`.
`observability.metrics_enabled`, `metrics_path` (`/metrics`), `tracing_enabled`, `otlp_endpoint`
(OpenTelemetry — install the `otel` extra).

### `security`
| Key | Default | Notes |
|-----|---------|-------|
| `auth_enabled` | `false` | When true, at least one API key is required |
| `api_key_header` | `X-API-Key` | `Authorization: Bearer <key>` is always accepted too |
| `api_keys[]` | `[]` | `name`, `key` **or** `key_sha256`, `scopes` (`query`, `ingest`, `admin`), `collections` (`["*"]`) |
| `rate_limit` | disabled | `requests_per_minute`, `burst` — per key (or per IP when anonymous) |
| `cors_allow_origins` | `[]` | CORS disabled when empty |
| `trusted_hosts` | `["*"]` | Host header allow-list |
| `max_request_body_mb` | `50` | Enforced before the body is read |
| `security_headers` | `true` | nosniff, frame deny, no-referrer, `Cache-Control: no-store` on API routes |

### `vector_store`
`provider`: `memory` | `qdrant` | `pgvector` | *plugin*.

`qdrant`: `url`, `api_key`, `location` (`:memory:` = embedded), `path` (embedded on-disk),
`prefer_grpc`, `timeout_s`, `collection_prefix` (`rag_`), `on_disk_payload`, `on_disk_vectors`,
`hnsw.{m, ef_construct, ef_search}`, `quantization` (`none`|`scalar`|`binary`), `shard_number`,
`replication_factor`.

`pgvector`: `dsn`, `db_schema` (`rag`), `table_prefix` (`chunks_`), `pool_min_size`, `pool_max_size`,
`hnsw_m`, `hnsw_ef_construction`, `ef_search`, `sparse_dimension` (must exceed the largest sparse
index — 1,048,576 covers SPLADE's 30,522 and the lexical hash space).

### `registry`
`url` (SQLAlchemy async URL: `sqlite+aiosqlite:///./data/rag_registry.db` or
`postgresql+psycopg://…`), `auto_create_schema` (use Alembic in production), `store_content`
(keep normalized text so collections can be re-indexed — disable only if you always re-send
documents), `pool_size`, `echo`.

### `llm` / `utility_llm`

Providers (all behind the same `LLMClient` port, selected by `provider`):

| `provider` | Extra | Auth | Use it for |
|---|---|---|---|
| `extractive` | – | none | Offline tests and development (quotes retrieved sentences) |
| `anthropic` (alias `claude`) | `anthropic` | API key | Claude in production; streaming, refusal fallbacks, prompt caching |
| `openai` | `openai` | API key | OpenAI and any OpenAI-compatible server (vLLM, Ollama, Azure, LM Studio, LiteLLM) |
| `gemini` (alias `google`) | `gemini` | API key, or Vertex AI credentials | Google Gemini through the `google-genai` SDK |
| `claude_lite_llm` (aliases `claude-lite-llm`, `claude_lite`) | `claude-lite` | your own Claude subscription (no API key) | **Personal / development use only** — see below |

| Key | Default | Notes |
|-----|---------|-------|
| `provider` | `extractive` | See the table above, or a plugin name |
| `model` | `extractive` | Anthropic default when empty: `claude-opus-5`. Required for `openai` and `gemini` (e.g. `gemini-3.8-flash`). `claude_lite_llm`: a CLI alias (`opus`, `sonnet`) or full ID; empty uses the CLI's default for your plan |
| `api_key` | – | Falls back to the SDK's env var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`). Not used by `claude_lite_llm` |
| `base_url` | – | OpenAI-compatible servers; Gemini API endpoint override |
| `timeout_s` / `max_retries` | `120` / `2` | Per request. `claude_lite_llm` has no retries (one CLI run per call) |
| `max_tokens` | `16000` | Per-pipeline `generation.max_tokens` overrides it (ignored by `claude_lite_llm`: the CLI has no such flag) |
| `temperature` | unset | Only sent when set (current Claude models reject sampling parameters; ignored by `claude_lite_llm`) |
| `anthropic.refusal_fallback` | `default` | Server-side fallback when a request is declined |
| `anthropic.effort` | unset | `low`…`max` |
| `anthropic.cache_system_prompt` | `true` | Prompt caching for repeated system prompts (contextual enrichment) |
| `claude_lite.oauth_token` | unset | Token from `claude setup-token`, given to the CLI as `CLAUDE_CODE_OAUTH_TOKEN`. Unset: the CLI's own login (`claude login`), or `CLAUDE_CODE_OAUTH_TOKEN` already in the environment |
| `claude_lite.cli_path` | PATH lookup | Path to the `claude` executable |
| `claude_lite.workdir` | private temp dir | Where the CLI runs. Keep it outside any project so no `CLAUDE.md`, hooks or `.mcp.json` are loaded |
| `claude_lite.max_concurrency` | `2` | CLI processes at once (plan usage limits apply to all of them together) |
| `claude_lite.max_arg_bytes` | `120000` | Largest prompt / system prompt; the CLI receives them as command-line arguments |
| `claude_lite.extra_args` | `[]` | Extra CLI flags, e.g. `["--effort", "low"]`, `["--fallback-model", "sonnet"]` |
| `options` | `{}` | OpenAI: `max_tokens_param`, `include_usage`, `extra_body`. Gemini: `thinking_level` (`minimal`…`high`), `thinking_budget`, `vertexai`, `project`, `location` |

`utility_llm` (optional) is used for query rewriting, multi-query, HyDE, history condensation and
contextual enrichment; it defaults to `llm`.

LLM clients are created at startup, so a missing SDK, CLI or model name stops the service instead of
failing the first chat request.

#### Claude through your subscription (`claude_lite_llm`)

[`claude-lite-llm`](https://pypi.org/project/claude-lite-llm/) runs the Claude Code CLI
(`claude -p`) with your own Claude subscription, so answers need no API key and are not billed per
token; they count against your plan's usage limits. Overlay: `config/profiles/llm-claude-lite.yaml`.

```bash
make install                         # all extras (or: uv sync --inexact --extra claude-lite); plus Claude Code: https://code.claude.com
claude login                         # or run `claude setup-token` and export CLAUDE_CODE_OAUTH_TOKEN=<token>
RAG_CONFIG_FILE=config/rag.yaml,config/profiles/llm-claude-lite.yaml make run
```

* **Who may use it.** Anthropic's terms reserve subscription (OAuth) credentials for the
  subscriber's own use of Claude Code and do not allow routing other people's requests through
  them. Use this provider on your own machine for your own work; anything shared or deployed for
  others uses `provider: anthropic` with an API key. The service logs a warning at startup as a
  reminder. Reasoning: [ADR 0006](adr/0006-subscription-claude-via-claude-code-cli.md).
* **What the adapter locks down.** The CLI is an agent, and retrieved text is untrusted, so every
  call runs with no tools (built-in or MCP), no claude.ai connectors, no skills or slash commands,
  in an empty working directory. `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`
  and the cloud-provider switches are removed from the CLI's environment — in `-p` mode an API key
  would otherwise take precedence and bill per token. Settings in your user-level `~/.claude`
  (for example hooks or `apiKeyHelper`) still apply.
* **Trade-offs.** Each call starts a CLI process (a few seconds of overhead), answers arrive in one piece
  (SSE sends a single `delta`), prompts larger than `max_arg_bytes` are rejected, and prompts are
  visible in the host's process list while the CLI runs.
* **Codex.** `claude-lite-llm` can also drive the OpenAI Codex CLI; that backend is not wired in
  because `codex exec` always runs as an agent with shell access (read-only sandbox), which
  retrieved text could steer into reading local files.

### `ingestion`
| Key | Default | Notes |
|-----|---------|-------|
| `default_mode` | `async` | Per-request `options.mode` overrides |
| `sync_max_documents` | `20` | Larger "sync" batches run async (202) |
| `max_documents_per_request` | `100` | |
| `max_document_bytes` | 20 MB | Per document after base64 decoding |
| `max_chunks_per_document` | `20000` | |
| `worker_concurrency` | `4` | Sources processed in parallel |
| `embed_concurrency` | `2` | Concurrent embedding batches (CPU/GPU bound) |
| `normalize_unicode` / `collapse_whitespace` | `true` | Text normalisation before hashing/chunking |
| `allowed_mime_types` | text, markdown, html, csv, json, pdf | |
| `callbacks` | disabled | `enabled`, `allowed_hosts` (exact host allow-list), `timeout_s` |

### `defaults` — the pipeline

**`chunking`**

| Key | Default | Notes |
|-----|---------|-------|
| `strategy` | `recursive` | `fixed`, `recursive`, `sentence`, `markdown`, `semantic`, *plugin* |
| `chunk_size` | `512` | In `length_unit` |
| `chunk_overlap` | `64` | Must be < `chunk_size` |
| `length_unit` | `tokens` | `chars`, `words`, `tokens` |
| `tokenizer` | `approx` | `approx` (chars/4) or `tiktoken:cl100k_base` (extra `tokenizers`) |
| `min_chunk_size` | `0` | Smaller chunks are merged into a neighbour in the same section |
| `separators` | paragraph → line → sentence → clause → word | Recursive splitter priority list |
| `markdown_header_levels` | `[1, 2, 3]` | Header levels that start a section |
| `semantic.breakpoint_percentile` | `90` | Higher → fewer, larger chunks |
| `semantic.buffer_size` | `1` | Sentences combined around each sentence before embedding |

**`enrichment`** — `contextual_header.enabled` (`true`), `contextual_header.template`
(`"{title}\n{section}\n\n{text}"`), `llm_context.enabled` (`false`), `llm_context.max_document_chars`,
`llm_context.max_concurrency`, `llm_context.max_tokens`.

**`embedding`**

| Key | Default | Notes |
|-----|---------|-------|
| `dense.provider` / `dense.model` | `hash` / `hash-384` | `fastembed` (e.g. `BAAI/bge-small-en-v1.5`, `BAAI/bge-large-en-v1.5`), `openai` (e.g. `text-embedding-3-large`) |
| `dense.dimension` | inferred | Required only for unknown OpenAI-compatible models (otherwise probed) |
| `dense.distance` | `cosine` | `cosine`, `dot`, `euclid` |
| `dense.query_prefix` / `document_prefix` | `""` | BGE v1.5: `"Represent this sentence for searching relevant passages: "`; E5: `"query: "` / `"passage: "` |
| `dense.batch_size` | `32` | |
| `dense.options` | `{}` | fastembed: `cache_dir`, `threads`, `providers`, `cuda`; openai: `api_key`, `base_url`, `send_dimensions` |
| `sparse.provider` / `sparse.model` | `lexical` / `lexical-bm25` | `fastembed`: `prithivida/Splade_PP_en_v1`, `Qdrant/bm25`, `Qdrant/bm42-all-minilm-l6-v2-attentions` |
| `sparse.options` | `{}` | lexical: `k1`, `b`, `avg_doc_len`, `hash_space`, `drop_stopwords` |
| `query_cache_size` | `2048` | LRU of query embeddings (0 disables) |

Set `dense: null` or `sparse: null` for a single-modality collection.

**`retrieval`**

| Key | Default | Notes |
|-----|---------|-------|
| `mode` | `hybrid` | `dense`, `sparse`, `hybrid` |
| `top_k` | `5` | Results returned |
| `candidate_k` | `50` | Candidates per retriever before fusion (raise for better recall) |
| `min_score` | – | Final-score cut-off (score scale depends on fusion / rerank) |
| `dense_score_threshold` | – | Similarity cut-off inside dense search |
| `dedupe` | `true` | Collapse identical texts |
| `group_by` | – | Collapse results that share a field: `source_id` (one hit per document) or a metadata key such as `family` (product variants). `""` disables a configured grouping per request |
| `group_size` | `1` | Results kept per group (e.g. `2` = at most two chunks per page) |
| `fusion.method` | `rrf` | `rrf`, `dbsf`, `rsf` |
| `fusion.k` | `60` | RRF smoothing constant |
| `fusion.weights` | `{dense: 1, sparse: 1}` | Per-retriever weights |
| `fusion.execution` | `client` | `native` → store-side fusion (Qdrant RRF/DBSF) |
| `query_transform.strategy` | `none` | `multi_query`, `hyde`, `rewrite` |
| `query_transform.num_queries` / `include_original` | `3` / `true` | |
| `rerank.enabled` | `false` | |
| `rerank.provider` / `model` | `fastembed` / `Xenova/ms-marco-MiniLM-L-6-v2` | `http` for Cohere/Jina/TEI (`options.url`, `options.api_key`) |
| `rerank.candidates` | `30` | Fused candidates passed to the cross-encoder |
| `rerank.min_score` | – | Drop low rerank scores |
| `mmr.enabled` / `lambda_mult` | `false` / `0.5` | Diversity (1 = relevance only) |
| `context_expansion.enabled` / `window` | `false` / `1` | Attach neighbouring chunks as `context` |

**`generation`** — `system_prompt`, `max_context_tokens` (`6000`), `max_tokens`, `temperature`,
`condense_history` (`true`), `history_turns` (`6`), `no_context_answer`.

### `collections`
```yaml
collections:
  products:
    description: Product catalog
    payload_indexes: [{field: brand, type: keyword}, {field: cfm, type: integer}]
    pipeline:                      # partial PipelineConfig
      chunking: {strategy: markdown, chunk_size: 256}
      retrieval: {top_k: 8, fusion: {weights: {sparse: 1.2}}}
```
`payload_indexes[]`: `field`, `type` (`keyword`, `integer`, `float`, `bool`, `datetime`, `text`) and
`tenant` (`false`). `tenant: true` (keyword only) marks the field that partitions a shared collection,
e.g. `site` in a multi-site CMS. Qdrant then builds a tenant-optimised index (`is_tenant`); other
stores ignore the flag. Indexes are created with the collection; recreate it to add one later.
`allow_dynamic_collections` (`true`) permits `POST /collections`. `plugins: [my_pkg.rag_ext]`
imports extra modules that register components.

## Worked examples

```bash
# Local, offline
RAG_CONFIG_FILE=config/rag.yaml uv run rag-service

# Hybrid with Qdrant, bigger dense model, stronger sparse weight
RAG_CONFIG_FILE=config/rag.yaml,config/profiles/qdrant-hybrid.yaml \
RAG__DEFAULTS__EMBEDDING__DENSE__MODEL=BAAI/bge-large-en-v1.5 \
RAG__DEFAULTS__RETRIEVAL__FUSION__WEIGHTS='{"dense": 1.0, "sparse": 1.5}' \
uv run rag-service

# OpenAI-compatible local model server for generation (e.g. Ollama)
RAG__LLM__PROVIDER=openai RAG__LLM__MODEL=llama3.1 RAG__LLM__BASE_URL=http://localhost:11434/v1 RAG__LLM__API_KEY=ollama

# Google Gemini
RAG__LLM__PROVIDER=gemini RAG__LLM__MODEL=gemini-3.8-flash RAG__LLM__API_KEY=$GEMINI_API_KEY

# Claude through your own subscription (personal use; Claude Code installed and logged in)
RAG__LLM__PROVIDER=claude_lite_llm RAG__LLM__MODEL=opus
```
