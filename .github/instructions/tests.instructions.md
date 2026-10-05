---
applyTo: "tests/**"
description: Testing conventions
---

- Default tests must run offline: build settings with `tests.conftest.make_settings(...)` (memory
  store, in-memory SQLite registry, hash/lexical embedders, extractive LLM).
- HTTP tests use `app_client` or `async with client_for(settings) as client:` (runs the lifespan).
- Assert on behaviour and contracts (status codes, envelope/problem shape, actions like
  `created/updated/unchanged`), not on log output or private attributes.
- Use `CountingEmbedder.calls` to assert what was (re-)embedded and `FakeLLM` (see
  `tests/unit/test_llm_features.py`) for LLM-driven behaviour.
- Real infrastructure tests are opt-in via env vars (`RAG_IT_QDRANT_URL`, `RAG_IT_PG_DSN`,
  `RAG_MODELS=1`) and must skip cleanly otherwise.
- Keep tests fast (< 1 s each) and independent; no sleeps except bounded polling of async jobs.
