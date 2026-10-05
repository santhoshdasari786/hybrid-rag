# CLAUDE.md

Project instructions are shared with GitHub Copilot — the single source of truth is:

@.github/copilot-instructions.md

## Claude Code specifics

- Path-scoped rules live in `.github/instructions/*.instructions.md` (the `applyTo` glob says when
  each applies) — read the matching file before editing Python, API, adapter, test or doc files.
- Skills: `.claude/skills` → `.github/skills` (add-rag-component, ingest-and-sync,
  rag-pipeline-config, rag-retrieval-tuning).
- Subagents: `.claude/agents/` (rag-architect, retrieval-tuner, api-reviewer, test-engineer) mirror
  `.github/agents/`; keep them in sync with `scripts/sync_agents.py`.
- Reusable task prompts for Copilot are in `.github/prompts/`; in Claude Code simply ask for the task
  (e.g. "add a Weaviate vector store") and the `add-rag-component` skill applies.
- Verify changes with `make lint test` before reporting completion.
