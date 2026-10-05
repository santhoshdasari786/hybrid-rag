---
description: Write tests for a module following the project's test conventions
agent: agent
argument-hint: "file or feature to test"
---

Write tests for `${input:target:file or feature}`.

- Follow `.github/instructions/tests.instructions.md`: offline by default via `make_settings`,
  `client_for` for HTTP, `CountingEmbedder` / `FakeLLM` for embeddings and LLM behaviour.
- Cover the happy path, boundaries, error mapping (problem+json codes) and idempotency where relevant.
- Prefer behavioural assertions; avoid mocking internals of the unit under test.
- Run `make test` and report coverage for the touched module (`uv run pytest --cov=<module>`).
