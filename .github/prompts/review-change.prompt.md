---
description: Enterprise review of the current change set
agent: api-reviewer
---

Review the current changes (`git diff` against the main branch) against the project's standards:
layering and SOLID rules, API conventions, error handling, security (SQL, SSRF, secrets, prompt
injection), idempotency of ingestion, test coverage, documentation and configuration updates.
Report findings ordered by severity with file:line references and concrete fixes.
