---
applyTo: "doc/**/*.md"
description: Documentation conventions
---

- Audience: engineers integrating, operating or extending the service. Lead with what to do,
  then why.
- Keep documents in their lane: configuration keys → `02-configuration.md`; endpoints → 
  `05-api-reference.md`; operational procedures → `07-operations.md`.
- Every config key, endpoint and error code mentioned must exist in the code — verify before
  writing. Use real request/response examples.
- Architectural decisions get an ADR in `doc/adr/NNNN-title.md` (Context, Decision, Consequences).
- Mermaid diagrams are welcome for flows; keep them small.
