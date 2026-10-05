"""Subscription-backed Claude through the Claude Code CLI (extra ``claude-lite``, ``claude-lite-llm``).

``provider: claude_lite_llm`` runs ``claude -p`` as a subprocess authenticated with the operator's own
Claude subscription (``claude login`` or a ``claude setup-token`` token) instead of an API key, so
calls count against the plan's usage limits rather than per-token billing.

Anthropic's terms reserve subscription credentials for the subscriber's own use of Claude Code: they
must not serve other people's requests. Use this provider for personal and development work, and
``provider: anthropic`` with an API key for anything shared (doc/adr/0006).

What the adapter adds on top of the package (see ``_claude_lite_client``): no tools, MCP servers,
skills or project files reach the CLI; API-key variables are removed from its environment; prompts
are bounded by ``max_arg_bytes``; at most ``max_concurrency`` CLI processes run at once.

Limits: a process start per call (seconds of overhead), no token streaming (``stream`` emits the
answer as one delta), and ``max_tokens`` / ``temperature`` are not supported by the CLI.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from rag_service.core.config import LLMConfig
from rag_service.core.errors import (
    ConfigurationError,
    DependencyMissingError,
    ProviderError,
    ProviderUnavailableError,
)
from rag_service.core.logging import get_logger
from rag_service.core.registry import LLMS
from rag_service.domain.models import ChatMessage, LLMRequest, LLMResponse, LLMStreamEvent, TokenUsage

log = get_logger(__name__)

PROVIDER = "claude_lite_llm"
_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "refusal": "refusal"}
_ROLE_LABEL = {"user": "User", "assistant": "Assistant"}
_MAX_ERROR_CHARS = 300


def render_prompt(messages: list[ChatMessage]) -> str:
    """One user turn is sent as-is; a conversation uses the package's ``[User: …]`` transcript format."""
    turns = [m for m in messages if m.role in _ROLE_LABEL]
    if len(turns) == 1 and turns[0].role == "user":
        return turns[0].content
    return "\n\n".join(f"[{_ROLE_LABEL[m.role]}: {m.content}]" for m in turns)


class ClaudeLiteLLM:
    provider = PROVIDER

    def __init__(self, config: LLMConfig) -> None:
        try:
            from rag_service.adapters.llm import _claude_lite_client as cli  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise DependencyMissingError.for_extra("claude-lite-llm", "claude-lite") from exc
        self._cli = cli
        self.config = config
        opts = config.claude_lite
        # Empty → the CLI's default model for the signed-in plan; aliases (opus, sonnet) or full IDs work.
        self.model = config.model if config.model not in ("", "extractive") else ""
        self._owns_workdir = opts.workdir is None
        workdir = Path(opts.workdir) if opts.workdir else Path(tempfile.mkdtemp(prefix="rag-claude-cli-"))
        workdir.mkdir(parents=True, exist_ok=True)
        self._workdir = workdir
        try:
            self._client = cli.SubscriptionClaudeClient(
                token=opts.oauth_token.get_secret_value() if opts.oauth_token else None,
                claude_path=opts.cli_path,
                env_file=os.devnull,  # the service has its own configuration; never read a stray .env
                default_model=self.model or None,
                cwd=workdir,
                extra_args=opts.extra_args,
            )
        except cli.ClaudeCLINotFoundError as exc:
            self._remove_workdir()
            raise ConfigurationError(
                f"{exc} Install Claude Code (https://code.claude.com) or set llm.claude_lite.cli_path."
            ) from exc
        self._slots = asyncio.Semaphore(opts.max_concurrency)
        log.warning(
            "llm.subscription_credentials",
            provider=PROVIDER,
            hint="Claude subscription credentials are for the subscriber's own use; "
            "shared deployments must use provider=anthropic with an API key (doc/adr/0006).",
        )

    def _split(self, request: LLMRequest) -> tuple[str | None, str]:
        system_parts = [request.system] if request.system else []
        system_parts += [m.content for m in request.messages if m.role == "system"]
        system = "\n\n".join(system_parts) or None
        prompt = render_prompt(request.messages)
        limit = self.config.claude_lite.max_arg_bytes
        for name, text in (("system prompt", system), ("prompt", prompt)):
            size = len(text.encode()) if text else 0
            if size > limit:
                raise ProviderError(
                    f"The {name} is {size} bytes; the Claude CLI accepts at most {limit} "
                    "(llm.claude_lite.max_arg_bytes). Lower retrieval.top_k or "
                    "enrichment.llm_context.max_document_chars, or use provider=anthropic."
                )
        return system, prompt

    def _map_error(self, exc: Exception) -> Exception:
        cli = self._cli
        detail = str(exc)[:_MAX_ERROR_CHARS]
        if isinstance(exc, cli.ClaudeRateLimitError):
            return ProviderUnavailableError(f"Claude subscription usage limit reached: {detail}")
        if isinstance(exc, cli.ClaudeAuthError):
            return ProviderError(
                "The Claude CLI is not signed in: run `claude login`, or set llm.claude_lite.oauth_token "
                f"to a token from `claude setup-token`. ({detail})"
            )
        if "timed out" in detail:
            return ProviderUnavailableError(f"Claude CLI timed out after {self.config.timeout_s:g}s")
        return ProviderError(f"Claude CLI failed: {detail}")

    def _answer_model(self, raw: dict[str, Any]) -> str:
        """The CLI may call more than one model (e.g. a small helper call); the answer's model is the one
        whose token counts match the result's ``usage``."""
        usage = raw.get("usage") or {}
        for name, stats in (raw.get("modelUsage") or {}).items():
            if stats.get("inputTokens") == usage.get("input_tokens") and stats.get("outputTokens") == usage.get(
                "output_tokens"
            ):
                return str(name)
        return self.model or "claude-code"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        system, prompt = self._split(request)
        async with self._slots:
            try:
                result = await self._client.acompletion(
                    prompt=prompt,
                    system_prompt=system,
                    model=self.model or None,
                    timeout=self.config.timeout_s,
                )
            except self._cli.ClaudeError as exc:
                raise self._map_error(exc) from exc
        usage = result.usage
        return LLMResponse(
            text=result.content,
            model=self._answer_model(result.raw),
            finish_reason=_FINISH.get(str(result.raw.get("stop_reason")), "stop"),
            usage=TokenUsage(
                input_tokens=usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens,
                output_tokens=usage.output_tokens,
            ),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        response = await self.generate(request)
        if response.text:
            yield LLMStreamEvent(delta=response.text)
        yield LLMStreamEvent(
            done=True, finish_reason=response.finish_reason, usage=response.usage, model=response.model
        )

    def _remove_workdir(self) -> None:
        if self._owns_workdir:
            shutil.rmtree(self._workdir, ignore_errors=True)

    async def close(self) -> None:
        self._remove_workdir()


@LLMS.register(
    PROVIDER,
    aliases=("claude-lite-llm", "claude_lite"),
    description="Claude via the Claude Code CLI and your own subscription (claude-lite-llm; personal use)",
)
def _claude_lite(*, config: LLMConfig) -> ClaudeLiteLLM:
    return ClaudeLiteLLM(config)
