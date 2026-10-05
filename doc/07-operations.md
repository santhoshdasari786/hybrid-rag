# 07 · Operations

## Running

| Where | Command |
|---|---|
| Local, offline | `make run` (memory store, SQLite registry, hash/lexical embedders) |
| Local, hybrid | `docker run -p 6333:6333 qdrant/qdrant` then `make run-hybrid` |
| Local, Claude on your subscription | `make run-claude-lite` (Claude Code installed and logged in; personal use — [ADR 0006](adr/0006-subscription-claude-via-claude-code-cli.md)) |
| Full stack | `make up` → API :8080, Qdrant :6333, Postgres :5432 (`docker-compose.yml`) |
| Image | `make docker-build` → `advanced-rag-service:local` (multi-stage, non-root, healthcheck) |

The container reads `RAG_CONFIG_FILE` (default `config/rag.yaml`), serves on `$PORT` (8080) and
keeps FastEmbed models in `/app/data/models` (mount a volume so they are downloaded once).
The image installs the `anthropic`, `openai` and `gemini` SDKs (build arg `EXTRAS`). It does not
include the Claude Code CLI: `claude_lite_llm` is for running on your own machine (ADR 0006).

### Production checklist

- [ ] `config/profiles/production.yaml` layered on: `auth_enabled`, keys as `key_sha256`, scopes and
      collection allow-lists per client.
- [ ] Registry on PostgreSQL; `auto_create_schema: false`; `alembic upgrade head` in the release
      pipeline (`REGISTRY_URL=… make migrate`).
- [ ] Qdrant with persistence/replication (or managed Qdrant Cloud), API key set.
- [ ] Secrets from a secret manager as environment variables (`ANTHROPIC_API_KEY`, `QDRANT_API_KEY`, …).
- [ ] LLM provider uses an API key or cloud credentials (`anthropic`, `openai`, `gemini`) — never
      `claude_lite_llm`, whose subscription credentials are for one person's own use (ADR 0006).
- [ ] `docs_enabled: false` on public ingress (or protect `/docs`).
- [ ] TLS terminated at the ingress / gateway; set `trusted_hosts` and CORS origins explicitly.
- [ ] Rate limit at the gateway for global limits (the built-in limiter is per replica).
- [ ] Model cache volume; `warmup_models: true` so readiness waits for models.
- [ ] Resource requests: CPU-bound embedding — size `embed_concurrency` ≈ cores / 2 per replica.
- [ ] Dashboards/alerts on the metrics below; log shipping of JSON logs.

### Kubernetes probes

```yaml
livenessProbe:  {httpGet: {path: /health/live, port: 8080}, periodSeconds: 10}
readinessProbe: {httpGet: {path: /health/ready, port: 8080}, periodSeconds: 5, failureThreshold: 3}
startupProbe:   {httpGet: {path: /health/ready, port: 8080}, periodSeconds: 5, failureThreshold: 60}  # model warm-up
terminationGracePeriodSeconds: 45   # > server.graceful_shutdown_s so jobs drain
```

## Scaling

* The API is stateless apart from **in-process async jobs**. Replicas scale horizontally for
  retrieval/chat. For heavy ingestion either run a dedicated ingestion deployment, or implement the
  `JobRunner` port on a queue (Redis/Arq, Celery, SQS) so work survives restarts and is shared.
* Jobs interrupted by a restart are marked `failed` at startup ("Interrupted by service restart");
  re-submitting is safe because ingestion is idempotent.
* Concurrent updates of one source across replicas are protected by optimistic locking
  (`409 CONCURRENT_MODIFICATION` — retry).
* GPU: pass `providers: ["CUDAExecutionProvider"]` in the FastEmbed `options` (install
  `fastembed-gpu`), or serve models through TEI / vLLM with the `openai` embedder and `http`
  reranker.

## Observability

**Logs** — JSON (structlog) with `request_id`, `principal`, route, status, duration; secrets are
redacted. Key events: `ingest.source`, `ingest.source_deleted`, `retrieval.done` (per-stage
`ms_*` timings), `job.finished`, `collection.stale`, `request.error`.

**Metrics** (`/metrics`)

| Metric | Type | Labels |
|---|---|---|
| `rag_http_requests_total` | counter | method, route (template), status |
| `rag_http_request_duration_seconds` | histogram | method, route |
| `rag_http_requests_in_flight` | gauge | |
| `rag_retrievals_total` | counter | collection, mode |
| `rag_retrieval_results` | histogram | collection |
| `rag_llm_tokens_total` | counter | provider, kind (input/output) |
| `rag_jobs_active` | gauge | |

Suggested alerts: 5xx ratio > 1 %; p95 `/retrieve` latency; `rag_retrieval_results` bucket 0 rising
(empty results); `PROVIDER_UNAVAILABLE` in logs; readiness flapping.

**Tracing** — install the `otel` extra, set `observability.tracing_enabled: true` and
`otlp_endpoint`; FastAPI spans are exported via OTLP/gRPC.

## Security

* Scoped API keys, SHA-256 stored, constant-time-safe lookup by digest; per-key collection
  allow-lists for tenant isolation.
* Body-size limit before parsing; per-document byte and chunk limits; MIME allow-list.
* No server-side URL fetching (no SSRF surface); job callbacks only to allow-listed hosts, no
  redirects.
* SQL is parameterised everywhere; pgvector filters are bound JSONPath.
* Security headers + `Cache-Control: no-store` on API responses; containers run as non-root.
* **Prompt injection**: retrieved text is untrusted. It is wrapped in `<source>` tags and the system
  prompt instructs the model to answer only from sources. Do not grant the chat endpoint any
  tool/action capabilities without an additional review of that risk.
* Per-request `generation.system_prompt` overrides are available to any `query` key — restrict keys
  accordingly if that is not desired.

## Runbook

| Symptom | Check | Fix |
|---|---|---|
| `409 EMBEDDING_CONFIG_CHANGED` | `GET /collections/{c}` → `stale: true` | `POST /collections/{c}/reindex?recreate=true` or revert the embedding config |
| Ready probe `503` | body `checks` | registry / vector store reachability, credentials |
| Sources `status: failed` | `GET …/documents?status=failed`, `error` field, job results | fix input (MIME, base64, empty text) and re-send |
| Duplicates in results | `GET …/documents/{id}/chunks` vs point count | re-send the source (sync removes orphans), or `DELETE` + re-send |
| Slow ingestion | `ingest.source` durations | raise `embed_concurrency`, batch size; GPU; disable `llm_context` |
| Slow retrieval | `retrieval.done` `ms_*` fields | rerank candidates, `ef_search`, dense model size, native fusion |
| LLM errors | `PROVIDER_ERROR` / `PROVIDER_UNAVAILABLE` | key/quota/rate limits; `finish_reason: refusal` indicates a declined request. `claude_lite_llm`: "usage limit reached" (wait for the plan's reset), "not signed in" (`claude login`), or prompt larger than `claude_lite.max_arg_bytes` |
