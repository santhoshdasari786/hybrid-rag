# ADR 0004 — Response envelope, RFC 9457 errors and an OpenAI-compatible facade

* Status: accepted · Date: 2026-09-27

## Decision
* Success: `{"data": …, "meta": {request_id, timestamp, duration_ms, api_version, pagination}}`.
* Errors: RFC 9457 `application/problem+json` with a stable `code` and `request_id`.
* Streaming chat: SSE events `retrieval` → `delta`* → `done` (or `error`); errors detectable before
  streaming are returned as normal HTTP problems.
* An OpenAI-compatible `/v1/chat/completions` + `/v1/models` facade (collections as models,
  citations as an extension field) with OpenAI-shaped errors.

## Consequences
+ Predictable client code, standards-based errors, drop-in use from OpenAI SDKs and chat UIs.
− Two chat surfaces to maintain; both delegate to the same `ChatService`.
