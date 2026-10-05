"""Hardened ``claude-lite-llm`` client: pins the Claude Code CLI to plain text generation.

Imported lazily by :mod:`rag_service.adapters.llm.claude_lite` (the ``claude-lite`` extra is optional).
``claude -p`` is a full agent that would otherwise pick up tools, MCP servers, hooks and API keys from
its environment; retrieved text is untrusted, so none of that may be reachable from a RAG prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from claude_lite_llm import (
    ClaudeAuthError,
    ClaudeClient,
    ClaudeCLINotFoundError,
    ClaudeError,
    ClaudeExecutionError,
    ClaudeRateLimitError,
)

__all__ = [
    "HARDENING_ARGS",
    "ClaudeAuthError",
    "ClaudeCLINotFoundError",
    "ClaudeError",
    "ClaudeExecutionError",
    "ClaudeRateLimitError",
    "SubscriptionClaudeClient",
]

# Credentials and routing that would move the call off the subscription: in `-p` mode an
# ANTHROPIC_API_KEY always wins (per-token billing), and a custom base URL would receive the OAuth
# token. CLAUDE_CODE_TOKEN is the package's own variable, which the CLI does not read.
SCRUBBED_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_TOKEN",
)
FIXED_ENV = {"ENABLE_CLAUDEAI_MCP_SERVERS": "false", "DISABLE_AUTOUPDATER": "1"}
# The package already sends `--tools ""` (no built-in tools); these remove MCP tools and servers,
# skills and slash commands as well.
HARDENING_ARGS = ("--strict-mcp-config", "--disallowedTools", "*", "--disable-slash-commands")


class SubscriptionClaudeClient(ClaudeClient):
    def __init__(self, *, extra_args: Sequence[str] = (), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.extra_args = tuple(extra_args)

    def _build_env(self) -> dict[str, str]:
        env = super()._build_env()
        for name in SCRUBBED_ENV:
            env.pop(name, None)
        env.update(FIXED_ENV)
        if self.token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.token
        return env

    def _build_command(self, prompt: str, *args: Any, **kwargs: Any) -> list[str]:
        cmd = super()._build_command(prompt, *args, **kwargs)
        if cmd[1:3] != ["-p", prompt]:
            raise ClaudeExecutionError("Unsupported claude-lite-llm version: unexpected CLI command layout")
        # The prompt moves behind `--` so text starting with '-' can never be parsed as a flag.
        return [cmd[0], "-p", *cmd[3:], *HARDENING_ARGS, *self.extra_args, "--", prompt]
