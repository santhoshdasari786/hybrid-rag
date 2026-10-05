---
applyTo: "src/rag_service/api/**"
description: HTTP layer standards (FastAPI)
---

- Routers are thin: parse request model → call one application service → map to a response schema
  with `api/v1/mappers.py` → return `envelope(request, data)`.
- Every route declares `response_model=Envelope[...]`, `summary`, and uses a scope dependency from
  `api/deps.py`. Routes with a `{collection}` path parameter must use the collection-aware variants
  (`QueryPrincipal`, `IngestPrincipal`, `AdminCollectionPrincipal`).
- Request/response models live in `api/schemas/` and extend `ApiModel` (`extra="forbid"`), with
  `Field` constraints, descriptions and `examples`.
- Never raise `HTTPException` for domain errors — raise `core.errors` exceptions; handlers in
  `api/errors.py` render RFC 9457 problems (and OpenAI-shaped errors under `/v1`).
- Status codes: 201+Location (create), 202+Location (async job), 204 (delete, no body), 207
  (partial batch), 409 (conflict/stale), 422 (validation).
- Streaming: validate and prepare before returning `sse_response(...)`; errors after the stream
  starts are sent with `error_event`.
- Middleware must be pure ASGI (no response buffering) — see `api/middleware.py`.
- Add an API test in `tests/api/` for every new route: happy path, auth/scope, and error shape.
