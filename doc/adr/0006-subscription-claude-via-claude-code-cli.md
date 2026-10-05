# ADR 0006 — Subscription-backed Claude through the Claude Code CLI

* Status: accepted · Date: 2026-10-02

## Context
Developers want to run the service locally with a real Claude model without an API key or
per-token billing, using the Claude subscription they already have. `claude-lite-llm` wraps the
Claude Code CLI (`claude -p --output-format json`) for this and reports token usage.

Three facts shape how it can be used:

1. **Terms.** Anthropic's Claude Code legal and compliance page reserves OAuth (subscription)
   credentials for subscribers' ordinary use of Claude Code. Products and services built on Claude
   should use API keys, and developers may not route other people's requests through Free, Pro or
   Max plan credentials.
2. **The CLI is an agent.** Without hardening, `claude -p` loads `CLAUDE.md`, hooks and `.mcp.json`
   from its working directory, connects MCP servers (including claude.ai connectors), and can use
   tools. In a RAG service the prompt contains retrieved documents, which are untrusted.
3. **Credential precedence.** In `-p` mode an `ANTHROPIC_API_KEY` in the environment always wins over
   the subscription login, and `ANTHROPIC_BASE_URL` redirects the request (and the token).
   `claude-lite-llm` 0.2 copies the whole parent environment and exports the token as
   `CLAUDE_CODE_TOKEN`, a variable the CLI does not read (it reads `CLAUDE_CODE_OAUTH_TOKEN`).

## Decision
Add `provider: claude_lite_llm` as an LLM adapter beside `anthropic`, `openai` and `gemini`, behind
the optional `claude-lite` extra, scoped to **personal and development use**:

* A thin subclass of the package's `ClaudeClient` (`adapters/llm/_claude_lite_client.py`):
  - adds `--strict-mcp-config --disallowedTools "*" --disable-slash-commands` to the package's
    `--tools ""`;
  - puts the prompt after `--`, so text starting with `-` is never parsed as a flag;
  - removes API-key, auth-token, base-URL and cloud-provider variables from the CLI environment,
    sets `ENABLE_CLAUDEAI_MCP_SERVERS=false` and `DISABLE_AUTOUPDATER=1`, and exports a configured
    token as `CLAUDE_CODE_OAUTH_TOKEN`.
* The CLI runs in a private, empty temporary directory, removed on shutdown. No `.env` file is read
  by the package (`env_file=os.devnull`).
* Prompts above `max_arg_bytes` are rejected before a process starts, and a semaphore bounds the
  number of concurrent CLI processes.
* The adapter logs a warning at startup that subscription credentials are for the subscriber's own
  use. It is not part of `production.yaml`, and the Docker image does not bundle the CLI.
* The package's Codex backend is not exposed: `codex exec` always runs with a shell tool, which
  retrieved text could steer into reading local files.

## Consequences
+ Real Claude answers locally with no API key and no per-token cost. Usage still counts against
  the plan's limits, and usage-limit errors surface as `PROVIDER_UNAVAILABLE`.
+ Same port, registry and configuration as every other provider; switching to `anthropic` for a
  shared deployment is a one-line configuration change.
− A CLI process per call adds a few seconds of latency, and there is no token streaming (SSE
  delivers one `delta`). `max_tokens` and `temperature` are not supported.
− The adapter depends on two private methods of `claude-lite-llm` (`_build_env`,
  `_build_command`). The extra is pinned to `>=0.2,<0.3`, and tests run the real package against a
  fake `claude` executable to detect layout changes.
− User-level `~/.claude` settings (hooks, `apiKeyHelper`) still apply to the CLI; prompts are
  visible in the process list while a call runs. Both are acceptable on a personal machine and
  are further reasons not to run this provider on shared hosts.
