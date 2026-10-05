---
description: Add an API endpoint that follows the service's HTTP standards
agent: agent
argument-hint: "METHOD /path and purpose"
---

Add the endpoint `${input:route:METHOD /path}` that ${input:purpose:what it does}.

Checklist (see `.github/instructions/api.instructions.md`):
- Request/response models in `src/rag_service/api/schemas/v1.py` (`ApiModel`, constraints, examples).
- Business logic in an application service (`src/rag_service/application/`), not in the router.
- Router function: scope dependency from `api/deps.py`, `response_model=Envelope[...]`, `summary`,
  `description`; return `envelope(request, data)`; correct status code and `Location` header.
- Errors via `core/errors.py` exceptions only.
- Tests in `tests/api/`: success, validation (problem+json with `errors`), auth/scope (401/403),
  not-found.
- Update `doc/05-api-reference.md` (endpoint table + example). Run `make lint test`.
