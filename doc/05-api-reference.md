# 05 · API reference

Interactive docs: `/docs` (Swagger UI), `/redoc`, machine-readable `/openapi.json`.

## Conventions

**Authentication** (when `security.auth_enabled`): `X-API-Key: <key>` or `Authorization: Bearer <key>`.

| Scope | Grants |
|---|---|
| `query` | read collections/sources/chunks/jobs, retrieve, chat, evaluate, OpenAI facade |
| `ingest` | create/update/delete documents and metadata |
| `admin` | everything, incl. collection management, reindex, `/system/*` |

A key may be limited to named collections (`collections: [products]`); other collections return `403`.

**Success envelope** — every `/api/v1` success response:

```json
{
  "data": { "…": "…" },
  "meta": {
    "request_id": "f9a44b408da240999fdfa0325c659330",
    "timestamp": "2026-09-27T18:17:56.533071Z",
    "duration_ms": 47.8,
    "api_version": "v1",
    "pagination": {"limit": 50, "offset": 0, "total": 132, "has_more": true}
  }
}
```

**Errors** — RFC 9457 `application/problem+json`, with a stable machine-readable `code`:

```json
{
  "type": "urn:rag-service:problem:collection-not-found",
  "title": "Collection not found",
  "status": 404,
  "detail": "Collection 'nope' does not exist",
  "instance": "/api/v1/collections/nope/retrieve",
  "code": "COLLECTION_NOT_FOUND",
  "request_id": "40f74117af684b54aa8f3ffbaa35ba85"
}
```

Validation errors add `errors: [{"loc": ["body", "query"], "msg": "…", "type": "string_too_short"}]`.

| Status | Codes |
|---|---|
| 401 | `UNAUTHENTICATED` |
| 403 | `FORBIDDEN` |
| 404 | `COLLECTION_NOT_FOUND`, `SOURCE_NOT_FOUND`, `CHUNK_NOT_FOUND`, `JOB_NOT_FOUND` |
| 409 | `COLLECTION_EXISTS`, `CONFLICT` (managed collection), `CONCURRENT_MODIFICATION`, `EMBEDDING_CONFIG_CHANGED` |
| 413 | `PAYLOAD_TOO_LARGE` |
| 415 | `UNSUPPORTED_MEDIA_TYPE` |
| 422 | `VALIDATION_FAILED`, `INVALID_FILTER` |
| 429 | `RATE_LIMITED` (+ `Retry-After`) |
| 500 | `INTERNAL_ERROR`, `CONFIGURATION_ERROR`, `DEPENDENCY_MISSING` |
| 501 | `NOT_SUPPORTED` |
| 502 / 503 | `PROVIDER_ERROR` / `PROVIDER_UNAVAILABLE` (vector DB, model or LLM upstream) |

**Tracing** — send `X-Request-ID` (8–128 chars `[A-Za-z0-9._-]`) to correlate; it is echoed in the
response header, `meta.request_id`, problem bodies and every log line.

## Endpoints

| Method & path | Scope | Purpose |
|---|---|---|
| `GET /api/v1/collections` | query | List collections visible to the key |
| `POST /api/v1/collections` | admin | Create a collection (`201`, `Location`) |
| `GET /api/v1/collections/{c}` | query | Effective pipeline + stats (`sources`, `chunks`, `points`, `failed_sources`) |
| `PATCH /api/v1/collections/{c}` | admin | Update description / pipeline overrides (dynamic collections) |
| `DELETE /api/v1/collections/{c}` | admin | Delete index + registry data (`204`) |
| `POST /api/v1/collections/{c}/reindex[?recreate=true]` | admin | Re-process stored sources (`202` job) |
| `POST /api/v1/collections/{c}/documents` | ingest | Batch upsert (sync `200/207/422`, async `202`) |
| `GET /api/v1/collections/{c}/documents?status&limit&offset` | query | List sources |
| `PUT /api/v1/collections/{c}/documents/{source_id}` | ingest | Idempotent single upsert (sync by default) |
| `GET /api/v1/collections/{c}/documents/{source_id}` | query | Source record (version, hashes, status, error) |
| `DELETE /api/v1/collections/{c}/documents/{source_id}` | ingest | Delete source + all chunks |
| `POST /api/v1/collections/{c}/documents:delete` | ingest | Bulk delete by `source_ids` |
| `PATCH /api/v1/collections/{c}/documents/{source_id}/metadata` | ingest | Metadata-only update, no re-embedding |
| `GET /api/v1/collections/{c}/documents/{source_id}/chunks` | query | Chunks of a source (paginated) |
| `GET /api/v1/collections/{c}/chunks/{chunk_id}` | query | One chunk |
| `PATCH /api/v1/collections/{c}/chunks/{chunk_id}/metadata` | ingest | Chunk-level metadata (survives re-sync) |
| `POST /api/v1/collections/{c}/retrieve` | query | Raw RAG results |
| `POST /api/v1/collections/{c}/chat` | query | Grounded answer + citations (JSON or SSE) |
| `POST /api/v1/collections/{c}/evaluate` | query | Retrieval metrics per variant |
| `GET /api/v1/jobs?collection&status&limit&offset` | query | Jobs, newest first |
| `GET /api/v1/jobs/{job_id}` | query | Job progress + per-source results |
| `GET /api/v1/system/info` | admin | Providers, capabilities, registered components |
| `GET /api/v1/system/config` | admin | Effective config, secrets masked |
| `GET /v1/models` | query | Collections as OpenAI "models" (`rag:<collection>`) |
| `POST /v1/chat/completions` | query | OpenAI-compatible chat (JSON or SSE) |
| `GET /health/live`, `GET /health/ready` | none | Probes (`ready` → `503` until started and dependencies healthy) |
| `GET /metrics` | none | Prometheus |

Ingestion payloads and sync semantics: [03-ingestion-and-sync.md](03-ingestion-and-sync.md).

## Retrieve

```http
POST /api/v1/collections/products/retrieve
{
  "query": "revent 80 cfm",
  "top_k": 5,                          // shortcut for retrieval.top_k
  "mode": "hybrid",                    // shortcut for retrieval.mode
  "filter": {"category": "bathroom-fan", "cfm": {"lte": 110}},
  "retrieval": {                       // any retrieval setting, per request
    "fusion": {"method": "rrf", "weights": {"dense": 1.0, "sparse": 1.5}},
    "rerank": {"enabled": true, "candidates": 30},
    "mmr": {"enabled": false},
    "context_expansion": {"enabled": true, "window": 1},
    "query_transform": {"strategy": "none"},
    "group_by": "family", "group_size": 1   // one hit per product family; "" turns a configured grouping off
  },
  "include_debug": false
}
```

```json
{
  "data": {
    "collection": "products",
    "query": "revent 80 cfm",
    "mode": "hybrid",
    "results": [
      {
        "rank": 1,
        "chunk_id": "95b1d56f-eedf-5b21-bb06-df3aa01bd0e6",
        "source_id": "fan-revent-rg80",
        "chunk_index": 0,
        "score": 0.032787,
        "scores": {"fused": 0.032787, "dense_rank": 1.0, "dense_score": 0.4258, "sparse_rank": 1.0, "sparse_score": 2.8585},
        "text": "# ReVent RG80 Bathroom Exhaust Fan 80 CFM\n\nThe ReVent RG80 is a ceiling mounted …",
        "context": null,
        "title": "ReVent RG80 Bathroom Exhaust Fan 80 CFM",
        "metadata": {"brand": "ReVent", "category": "bathroom-fan", "cfm": 80, "tags": ["quiet", "ceiling"]}
      }
    ],
    "variants": [{"text": "revent 80 cfm", "kind": "original"}],
    "timings_ms": {"query_transform": 0.01, "search": 0.9},
    "config": null,
    "debug": null
  },
  "meta": {"request_id": "…", "timestamp": "…", "duration_ms": 3.1, "api_version": "v1"}
}
```

`scores` contains what applies: `fused`, `<retriever>_rank`, `<retriever>_score`, `rerank`, `mmr`.
With `include_debug: true`, `debug.lists` holds each retriever's ranked list, `debug.fused` the
fused list, `debug.reranked` the reranked list, and `config` the effective retrieval config.

### Filters

| Operator | Meaning | Example |
|---|---|---|
| `eq` / `ne` | equals / not equals (array fields match if any element matches) | `{"field": "brand", "op": "eq", "value": "Delta"}` |
| `in` / `nin` | any of / none of | `{"field": "tags", "op": "in", "value": ["quiet", "led"]}` |
| `gt` `gte` `lt` `lte` | numeric or ISO-8601 string ranges | `{"field": "released", "op": "gte", "value": "2025-01-01"}` |
| `exists` | field present and not empty | `{"field": "warranty", "op": "exists", "value": true}` |

Groups: `must` (AND), `should` (OR), `must_not` (NOR) — nestable up to 5 levels, 64 conditions.
Shorthand: `{"brand": "Delta", "tags": ["quiet"], "cfm": {"gte": 50, "lte": 100}}` (AND of eq / in /
range). Field names address user metadata; `source_id`, `chunk_index`, `chunk_hash`, `title`,
`mime_type` address the point itself. Semantics are identical on every vector store (verified by
the store contract tests).

## Chat

```http
POST /api/v1/collections/products/chat
{
  "messages": [
    {"role": "user", "content": "Tell me about the Delta RAD80"},
    {"role": "assistant", "content": "It is an 80 CFM fan with a heater [1]."},
    {"role": "user", "content": "How loud is it?"}
  ],
  "conversation_id": "c-42",
  "stream": false,
  "filter": {"brand": "Delta"},
  "retrieval": {"top_k": 6, "rerank": {"enabled": true}},
  "generation": {"max_tokens": 2000, "system_prompt": "Answer in one paragraph and cite sources."},
  "include_sources": true
}
```

```json
{
  "data": {
    "id": "chat_f8fc05873ee34a38a2e8eba4de466dbb",
    "conversation_id": "c-42",
    "collection": "products",
    "model": "claude-opus-5",
    "message": {"role": "assistant", "content": "The Delta BreezRadiance RAD80 is rated at 1.5 sones [1]."},
    "citations": [
      {"index": 1, "chunk_id": "…", "source_id": "fan-delta-rad80", "title": "Delta BreezRadiance RAD80 …",
       "snippet": "…", "score": 0.0328, "metadata": {"brand": "Delta", "cfm": 80}}
    ],
    "sources": ["… every source placed in the prompt, same shape …"],
    "finish_reason": "stop",
    "usage": {"input_tokens": 1830, "output_tokens": 41, "total_tokens": 1871},
    "retrieval": {"standalone_query": "Delta RAD80 noise level in sones", "mode": "hybrid", "results": 6, "timings_ms": {"search": 8.1}}
  },
  "meta": {"…": "…"}
}
```

* The service is stateless: send the conversation history in `messages` each turn.
* `finish_reason`: `stop`, `length`, `refusal` (provider declined), `no_context` (nothing relevant
  retrieved — the configured `no_context_answer` is returned without calling the LLM).
* `citations` are the sources referenced as `[n]` in the answer; `sources` are all sources given to
  the model.

### Streaming (`"stream": true`) — `text/event-stream`

```
event: retrieval
data: {"id":"chat_…","conversation_id":"c-42","standalone_query":"…","sources":[{…}]}

event: delta
data: {"content":"The Delta BreezRadiance RAD80 "}

event: delta
data: {"content":"is rated at 1.5 sones [1]."}

event: done
data: {"id":"chat_…","citations":[{…}],"finish_reason":"stop","model":"claude-opus-5","usage":{…}}
```

Validation, auth and not-found errors are returned as normal HTTP problems *before* the stream
starts; failures after it started arrive as `event: error` with a problem body. Browser `EventSource`
cannot POST — use `fetch` with a streaming reader (or `@microsoft/fetch-event-source`).

## OpenAI-compatible facade

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="<rag api key>")
reply = client.chat.completions.create(
    model="rag:products",
    messages=[{"role": "user", "content": "Which 80 CFM fans are quiet?"}],
    extra_body={"rag": {"filter": {"category": "bathroom-fan"}, "retrieval": {"top_k": 6}}},
)
print(reply.choices[0].message.content)
print(reply.model_extra["citations"])
```

`stream=True` yields standard `chat.completion.chunk` events ending with `data: [DONE]`; the final
chunk carries `finish_reason`, `usage` and `citations`. Errors use OpenAI's `{"error": {…}}` shape.
`system` / `developer` messages are appended to the collection's system prompt.

## Evaluate

```http
POST /api/v1/collections/products/evaluate
{
  "k": 5,
  "level": "source",
  "cases": [
    {"query": "revent 80 cfm", "relevant": {"fan-revent-rg80": 2, "fan-revent-rg110": 1}},
    {"query": "bathroom fan with heater", "relevant": ["fan-delta-rad80"]}
  ],
  "variants": {"hybrid": {"mode": "hybrid"}, "hybrid+rerank": {"mode": "hybrid", "rerank": {"enabled": true}}},
  "include_cases": false
}
```

Returns per variant `metrics`: `hit_rate`, `recall`, `precision`, `mrr`, `ndcg`, `latency_ms_p50`,
`latency_ms_max`, plus `best` (highest nDCG, MRR breaks ties; `null` when the top variants tie). Defaults to dense / sparse / hybrid variants.

## curl quick start

```bash
BASE=http://localhost:8080/api/v1/collections/products
curl -s -X POST $BASE/documents -H 'Content-Type: application/json' -d @examples/ingest_products.json | jq '.data.status'
curl -s -X POST $BASE/retrieve  -H 'Content-Type: application/json' -d @examples/retrieve.json | jq '.data.results[] | {rank, source_id, score}'
curl -s -X POST $BASE/chat      -H 'Content-Type: application/json' -d @examples/chat.json | jq '.data.message.content'
curl -N -X POST $BASE/chat      -H 'Content-Type: application/json' -d '{"stream": true, "messages": [{"role": "user", "content": "quiet 80 cfm fan?"}]}'
curl -s -X DELETE $BASE/documents/fan-revent-rg80 | jq '.data.action'
```
