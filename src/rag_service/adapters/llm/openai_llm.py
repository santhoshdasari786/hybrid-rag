"""OpenAI-compatible chat adapter (extra ``openai``).

Targets any ``/v1/chat/completions`` endpoint: OpenAI, Azure OpenAI, vLLM, Ollama, LM Studio,
LiteLLM, Together, Groq…

Options (``llm.options``):
* ``max_tokens_param`` — ``max_completion_tokens`` (default for api.openai.com) or ``max_tokens``
  (default when ``base_url`` points elsewhere — widest compatibility).
* ``include_usage`` — request usage in streams (``stream_options``); defaults to true for OpenAI.
* ``extra_body`` — dict merged into every request (provider-specific parameters).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from rag_service.core.config import LLMConfig
from rag_service.core.errors import (
    ConfigurationError,
    DependencyMissingError,
    ProviderError,
    ProviderUnavailableError,
)
from rag_service.core.registry import LLMS
from rag_service.domain.models import LLMRequest, LLMResponse, LLMStreamEvent, TokenUsage


class OpenAICompatibleLLM:
    provider = "openai"

    def __init__(self, config: LLMConfig) -> None:
        try:
            import openai  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise DependencyMissingError.for_extra("openai", "openai") from exc
        if not config.model or config.model == "extractive":
            raise ConfigurationError("llm.model must name the chat model when provider=openai")
        self._sdk = openai
        self.config = config
        self.model = config.model
        self._client = openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value() if config.api_key else None,
            base_url=config.base_url,
            timeout=config.timeout_s,
            max_retries=config.max_retries,
        )
        official = config.base_url is None
        self._max_tokens_param = str(
            config.options.get("max_tokens_param", "max_completion_tokens" if official else "max_tokens")
        )
        self._include_usage = bool(config.options.get("include_usage", official))
        self._extra_body = dict(config.options.get("extra_body", {}))

    def _params(self, request: LLMRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self._max_tokens_param: request.max_tokens or self.config.max_tokens,
        }
        temperature = request.temperature if request.temperature is not None else self.config.temperature
        if temperature is not None:
            params["temperature"] = temperature
        if self._extra_body:
            params["extra_body"] = self._extra_body
        return params

    def _map_error(self, exc: Exception) -> Exception:
        sdk = self._sdk
        if isinstance(exc, sdk.RateLimitError):
            return ProviderUnavailableError(f"LLM rate limit: {exc}")
        if isinstance(exc, sdk.APIStatusError):
            if exc.status_code >= 500:
                return ProviderUnavailableError(f"LLM server error ({exc.status_code}): {exc}")
            return ProviderError(f"LLM request rejected ({exc.status_code}): {exc}")
        if isinstance(exc, sdk.APIConnectionError):
            return ProviderUnavailableError(f"LLM unreachable: {exc}")
        return ProviderError(f"LLM error: {exc}")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self._client.chat.completions.create(**self._params(request))
        except self._sdk.OpenAIError as exc:
            raise self._map_error(exc) from exc
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            usage=TokenUsage(usage.prompt_tokens, usage.completion_tokens) if usage else TokenUsage(),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        params = self._params(request)
        params["stream"] = True
        if self._include_usage:
            params["stream_options"] = {"include_usage": True}
        finish: str | None = None
        usage = TokenUsage()
        model = self.model
        try:
            stream = await self._client.chat.completions.create(**params)
            async for chunk in stream:
                model = chunk.model or model
                if chunk.usage:
                    usage = TokenUsage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta and choice.delta.content:
                    yield LLMStreamEvent(delta=choice.delta.content)
                if choice.finish_reason:
                    finish = choice.finish_reason
        except self._sdk.OpenAIError as exc:
            raise self._map_error(exc) from exc
        yield LLMStreamEvent(done=True, finish_reason=finish or "stop", usage=usage, model=model)

    async def close(self) -> None:
        await self._client.close()


@LLMS.register(
    "openai",
    aliases=("openai-compatible", "azure-openai", "vllm", "ollama"),
    description="OpenAI-compatible chat completions",
)
def _openai(*, config: LLMConfig) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(config)
