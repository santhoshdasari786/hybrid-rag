# Copilot instructions — Advanced RAG Service

This repository is a configurable Retrieval-Augmented Generation micro-service (Python 3.11+,
FastAPI, Pydantic v2, SQLAlchemy 2 async, uv). Read `doc/01-architecture.md` before structural
changes and `doc/02-configuration.md` before touching configuration.

## Architecture rules (enforced by review, lint and tests)

- Layers: `core` → `domain` → `adapters` → `application` → `api`; `container.py` is the only
  composition root. Nothing below `rag_service.api` may import it (ruff `TID251`).
- The domain (`src/rag_service/domain`) has no I/O and no framework code except Pydantic for the
  filter DSL. Ports are `typing.Protocol`s in `domain/ports.py`.
- Services depend on the **narrowest port** they need (`VectorWriter`, `VectorReader`,
  `CollectionAdmin`), never on concrete adapters.
- New strategies (parser, chunker, embedder, vector store, fusion, reranker, query transform, LLM)
  are **registered by name** in `core/registry.py` registries — never add `if provider == …`
  branches to services. Provider-specific settings go in the config section's `options` dict.
- Optional third-party libraries are imported inside constructors and raise
  `DependencyMissingError.for_extra(package, extra)` when missing.
- Raise only exceptions from `core/errors.py`; the API maps them to RFC 9457 problems. Never
  raise `HTTPException` from services.
- Ingestion must stay idempotent: chunk IDs are `uuid5(collection/source_id/chunk_hash/occurrence)`,
  writes happen in the order upsert → registry save → delete stale. Do not change hashing or ID
  derivation without an ADR and a migration plan.
- Every vector store must satisfy `tests/integration/test_store_contract.py`, including filter
  semantics identical to `domain.filters.matches`.

## API standards

- Routes live under `settings.app.api_prefix` (`/api/v1`); resource-oriented paths
  (`/collections/{collection}/documents/{source_id}`).
- Success bodies use `envelope(request, data)` → `{"data", "meta"}`; list endpoints pass
  `pagination=paginate(limit, offset, total)`.
- Declare `response_model=Envelope[...]`, a `summary`, and precise request models (`ApiModel`,
  `extra="forbid"`, field constraints, examples) in `api/schemas/`.
- Protect every route with a scope dependency from `api/deps.py` (`QueryPrincipal`,
  `IngestPrincipal`, `AdminPrincipal`, …); routes with `{collection}` must use the
  collection-aware variants.
- Keep routers thin: validate → call one application service → map with `api/v1/mappers.py`.
- Streaming uses SSE (`api/sse.py`); do request validation *before* opening the stream.
- Status codes: 201 + `Location` for creation, 202 + `Location` for async jobs, 204 for deletes
  without body, 207 for partial batch success, 409 for conflicts/staleness.

## Coding standards

- Type everything; `mypy --strict` must pass. Prefer `@dataclass(slots=True)` for domain objects,
  Pydantic for config and API schemas.
- Async all the way; wrap CPU-bound work (embedding, reranking) in `asyncio.to_thread`.
- Structured logging via `core.logging.get_logger(__name__)`: `log.info("event.name", key=value)`.
  Never log secrets, document text or full prompts.
- Line length 120; ruff format; docstrings explain *why* and module intent, not the obvious.
- Configuration: add new settings to the Pydantic models in `core/config.py` with defaults,
  constraints and a `description`, then document them in `doc/02-configuration.md`.
- Security: parameterise all SQL; never fetch arbitrary URLs server-side; treat retrieved text as
  untrusted input to LLM prompts.

## LLM providers

- Anthropic via the official `anthropic` SDK (default model `claude-opus-5`, streaming,
  server-side refusal fallbacks, no sampling params unless configured). OpenAI-compatible via the
  `openai` SDK. Gemini via the official `google-genai` SDK. The `extractive` provider keeps
  everything runnable offline. Keep provider code in its own adapter file.
- `claude_lite_llm` runs the Claude Code CLI on the operator's own subscription (`claude-lite-llm`).
  It is for personal/development use only — never put it in production profiles or examples for
  shared deployments. Keep its hardening (no tools/MCP, scrubbed API-key env, private workdir,
  prompt after `--`) intact; see `doc/adr/0006-subscription-claude-via-claude-code-cli.md`.

## Cookbook

- `cookbook/recipes/*` are runnable lessons (config overlay + data + `scenario.yaml` with checked
  expectations); `tests/cookbook` runs every recipe on the offline stack in CI.
- When a change alters retrieval behaviour, run `make cookbook` (and `STACK=models` when models are
  available) and update a recipe's numbers or expectations only with a measured reason.
- New use-case guidance belongs in a recipe (README + config + scenario), not only in prose.

## Commands

```bash
make install        # uv sync --all-extras --group dev
make lint           # ruff format --check · ruff check · mypy --strict
make test           # offline unit + API + store-contract tests
make it-up test-it  # contract tests vs real Qdrant + pgvector (Docker)
make run            # local server on :8080 (offline config)
make cookbook       # every cookbook recipe, offline (STACK=models|qdrant|pgvector, LLM=anthropic|…)
```

## Definition of done

1. Tests added/updated (see `doc/09-testing.md`); `make lint test` green.
2. Public behaviour or config changes documented in `doc/` (and an ADR for architectural decisions).
3. No new mandatory dependency for an optional feature — use an extra.
4. OpenAPI stays accurate (models, examples, status codes).
