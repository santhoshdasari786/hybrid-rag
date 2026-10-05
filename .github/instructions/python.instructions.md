---
applyTo: "**/*.py"
description: Python conventions for the Advanced RAG Service
---

- Target Python 3.11+, `from __future__ import annotations` at the top of every module.
- Full type hints; code must pass `mypy --strict` (config in `pyproject.toml`). Avoid `Any` except at
  I/O boundaries (SDK responses, JSON payloads).
- Prefer `@dataclass(slots=True)` for domain values, Pydantic `BaseModel` for config/API schemas.
- Async I/O everywhere. CPU-bound work (embedding, cross-encoding) goes through `asyncio.to_thread`.
- Logging: `log = get_logger(__name__)`; `log.info("area.event", key=value)` — snake_case event
  names, structured fields, no f-strings in the event name, no secrets or document text.
- Errors: raise subclasses of `rag_service.core.errors.RAGError` with a helpful `detail`.
- Imports of optional extras (qdrant_client, fastembed, anthropic, openai, psycopg, pypdf, tiktoken)
  happen inside functions/constructors with `# noqa: PLC0415`.
- Keep functions small; one responsibility per class; no module-level mutable state except registries.
- Format with `ruff format` (line length 120); `ruff check` must be clean.
