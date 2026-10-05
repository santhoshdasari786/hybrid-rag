---
applyTo: "src/rag_service/adapters/**"
description: Rules for pipeline component adapters
---

- An adapter implements exactly one Protocol from `rag_service/domain/ports.py` and registers a
  factory with the matching registry in `rag_service/core/registry.py` using a lowercase name and a
  one-line `description`.
- Factory keyword arguments per registry are fixed (see `doc/08-extending.md`); read extra settings
  from the config section's `options` dict instead of adding schema fields.
- Heavy models load lazily and thread-safely; expose `async def warmup()` when loading is expensive.
- Map third-party exceptions to `ProviderError` / `ProviderUnavailableError` (retryable: timeouts,
  429, 5xx) and never leak raw SDK exceptions.
- Vector stores: translate `FilterSpec` so results match `domain.filters.matches` exactly; payload
  keys come from `PayloadKeys`; add the store to `tests/integration/test_store_contract.py`.
- Keep adapters free of application logic (no diffing, no job handling, no HTTP concerns).
