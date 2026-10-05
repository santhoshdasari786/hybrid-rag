# 09 · Testing & quality gates

| Suite | Path | Needs | Command |
|---|---|---|---|
| Unit | `tests/unit` | nothing | `make test` |
| API (full HTTP stack, lifespan, auth, SSE, OpenAI facade) | `tests/api` | nothing | `make test` |
| Store contract — memory + embedded Qdrant | `tests/integration` | nothing | `make test` |
| Store contract — Qdrant server + pgvector | `tests/integration` | Docker | `make it-up test-it it-down` |
| End-to-end with real models (BGE, SPLADE, cross-encoder) | `tests/e2e` | ~700 MB model download; optional Qdrant/Postgres | `make test-models` |

The default suite is fully offline: memory store, in-memory SQLite registry, `hash`/`lexical`
embedders and the `extractive` LLM, so it runs in seconds in CI with no secrets.

## What is covered

* **Configuration** — layering, env interpolation, env overrides, fail-fast validation, secret masking.
* **Chunkers** — size limits, overlap, character offsets, Markdown header paths (code-fence aware),
  tiny-chunk merging, semantic breakpoints.
* **Filters** — reference semantics for every operator, invalid input, SQL-injection safety of the
  pgvector compiler, Qdrant translation.
* **Fusion / MMR / embedders** — RRF reproduces the class-notebook example, weighting, DBSF/RSF
  scale handling, BM25 weighting, diversity selection.
* **Source sync** — created → unchanged → metadata-only → partial edit (asserts exactly one chunk
  re-embedded) → delete; chunk overrides survive re-sync; force; pre-chunked input; failure
  recording and retry; re-chunking on config change.
* **LLM features** — multi-query, HyDE, graceful fallback, contextual enrichment, history
  condensation, Anthropic/OpenAI request construction.
* **API** — envelopes, problem details, 200/202/207/422 ingestion semantics, pagination, auth, scopes,
  collection ACLs, rate limiting, body limits, request IDs, security headers, metrics labels,
  staleness + `reindex?recreate=true`, SSE event order, OpenAI JSON/stream shapes, evaluation.
* **Store contract** — identical behaviour of memory, embedded Qdrant, Qdrant server, pgvector.
* **Cookbook** (`tests/cookbook/`) — every recipe in `cookbook/recipes/` runs end to end on the
  offline stack. The steps that need a generative LLM run with a scripted one. Run them on real
  stacks with `make cookbook STACK=models|qdrant|pgvector LLM=…`.

## Static analysis

* `ruff format` + `ruff check` (pycodestyle, pyflakes, bugbear, bandit subset, pylint subset,
  import sorting, banned API-layer imports below the API).
* `mypy --strict` over `src/`.
* `pre-commit` runs all of the above plus hygiene hooks; CI (`.github/workflows/ci.yml`) runs
  lint, tests against real Qdrant + pgvector service containers, and the Docker build.

## Writing tests

* Use `tests/conftest.py::make_settings(**overrides)` for a configured offline stack and
  `client_for(settings)` for an HTTP client with lifespan.
* To assert "what was embedded", use the `counting` dense embedder (`CountingEmbedder.calls`).
* For LLM behaviour, register a scripted fake (see `tests/unit/test_llm_features.py::FakeLLM`).
* Provider adapters are tested without network: request construction and response mapping with the
  SDK's own types (`test_gemini_llm.py`), or a fake executable when the provider is a CLI
  (`test_claude_lite_llm.py` runs the real `claude-lite-llm` package against a fake `claude`).
* New vector store → add it to `_params()` in the contract suite; do not write store-specific
  behaviour tests that the contract does not cover.
