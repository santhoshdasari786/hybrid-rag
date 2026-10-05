---
description: Record an architecture decision (ADR)
agent: agent
argument-hint: "decision title"
---

Write an ADR for: ${input:title:decision title}.

Create `doc/adr/NNNN-<kebab-title>.md` with the next number, using the existing ADRs as the format:
`Status` / `Date`, **Context** (forces, constraints), **Decision** (what and where in the code),
**Consequences** (+ / −). Keep it under one page, reference code paths, and link it from any
document it changes.
