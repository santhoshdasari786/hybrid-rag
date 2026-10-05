"""Anthropic Claude adapter (extra ``anthropic``), built on the official ``anthropic`` SDK.

* Always streams internally (``messages.stream``) — long answers never hit HTTP timeouts.
* ``llm.anthropic.refusal_fallback: default`` (the default) enables server-side fallbacks: if the
  requested model's safety classifiers decline a request, the API re-runs it on Anthropic's
  recommended fallback model inside the same call. Set it to ``off`` to disable.
* A ``refusal`` stop reason is surfaced as ``finish_reason="refusal"``.
* The system prompt is marked cacheable when ``cache_system`` is requested (contextual enrichment
  sends the same document for every chunk, so caching cuts cost and latency).
* Sampling parameters are only sent when explicitly configured — current Claude models reject them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from rag_service.core.config import LLMConfig
from rag_service.core.errors import DependencyMissingError, ProviderError, ProviderUnavailableError
from rag_service.core.registry import LLMS
from rag_service.domain.models import LLMRequest, LLMResponse, LLMStreamEvent, TokenUsage

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "refusal",
    "tool_use": "tool_calls",
}


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, config: LLMConfig) -> None:
        try:
            import anthropic  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise DependencyMissingError.for_extra("anthropic", "anthropic") from exc
        self._sdk = anthropic
        self.config = config
        self.model = config.model if config.model not in ("", "extractive") else DEFAULT_MODEL
        self._client = anthropic.AsyncAnthropic(
            api_key=config.api_key.get_secret_value() if config.api_key else None,
            base_url=config.base_url,
            timeout=config.timeout_s,
            max_retries=config.max_retries,
        )

    def _params(self, request: LLMRequest) -> dict[str, Any]:
        system_parts = [request.system] if request.system else []
        system_parts += [m.content for m in request.messages if m.role == "system"]
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "messages": [
                {"role": m.role, "content": m.content} for m in request.messages if m.role in ("user", "assistant")
            ],
        }
        if system_parts:
            block: dict[str, Any] = {"type": "text", "text": "\n\n".join(system_parts)}
            if request.cache_system and self.config.anthropic.cache_system_prompt:
                block["cache_control"] = {"type": "ephemeral"}
            params["system"] = [block]
        temperature = request.temperature if request.temperature is not None else self.config.temperature
        if temperature is not None:
            params["temperature"] = temperature
        if self.config.anthropic.effort:
            params["output_config"] = {"effort": self.config.anthropic.effort}
        return params

    def _stream_ctx(self, request: LLMRequest) -> Any:
        params = self._params(request)
        if self.config.anthropic.refusal_fallback == "default":
            return self._client.beta.messages.stream(**params, betas=[FALLBACK_BETA], fallbacks="default")
        return self._client.messages.stream(**params)

    def _map_error(self, exc: Exception) -> Exception:
        sdk = self._sdk
        if isinstance(exc, sdk.RateLimitError):
            return ProviderUnavailableError(f"Anthropic rate limit: {exc}")
        if isinstance(exc, sdk.APIStatusError):
            if exc.status_code >= 500:
                return ProviderUnavailableError(f"Anthropic server error ({exc.status_code}): {exc}")
            return ProviderError(f"Anthropic request rejected ({exc.status_code}): {exc}")
        if isinstance(exc, sdk.APIConnectionError):  # includes APITimeoutError
            return ProviderUnavailableError(f"Anthropic unreachable: {exc}")
        return ProviderError(f"Anthropic error: {exc}")

    @staticmethod
    def _usage(message: Any) -> TokenUsage:
        usage = getattr(message, "usage", None)
        if usage is None:
            return TokenUsage()
        cached = (getattr(usage, "cache_read_input_tokens", 0) or 0) + (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )
        return TokenUsage(input_tokens=(usage.input_tokens or 0) + cached, output_tokens=usage.output_tokens or 0)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            async with self._stream_ctx(request) as stream:
                message = await stream.get_final_message()
        except self._sdk.AnthropicError as exc:
            raise self._map_error(exc) from exc
        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        return LLMResponse(
            text=text,
            model=message.model,
            finish_reason=_FINISH.get(str(message.stop_reason), str(message.stop_reason)),
            usage=self._usage(message),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        try:
            async with self._stream_ctx(request) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield LLMStreamEvent(delta=text)
                message = await stream.get_final_message()
        except self._sdk.AnthropicError as exc:
            raise self._map_error(exc) from exc
        yield LLMStreamEvent(
            done=True,
            finish_reason=_FINISH.get(str(message.stop_reason), str(message.stop_reason)),
            usage=self._usage(message),
            model=message.model,
        )

    async def close(self) -> None:
        await self._client.close()


@LLMS.register(
    "anthropic",
    aliases=("claude",),
    description="Anthropic Claude (Messages API, streaming, refusal fallbacks)",
)
def _anthropic(*, config: LLMConfig) -> AnthropicLLM:
    return AnthropicLLM(config)
