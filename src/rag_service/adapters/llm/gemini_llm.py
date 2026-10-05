"""Google Gemini adapter (extra ``gemini``), built on the official ``google-genai`` SDK.

* Gemini Developer API with ``llm.api_key`` (falls back to ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``),
  or Vertex AI with ``llm.options: {vertexai: true, project: …, location: …}`` and Google
  application-default credentials.
* Conversation roles map ``assistant`` → ``model``; system text goes to ``system_instruction``.
* ``llm.options.thinking_level`` / ``thinking_budget`` are passed to ``ThinkingConfig``; thought
  parts are never returned as answer text, but their tokens count as output.
* Sampling parameters are only sent when configured; safety blocks surface as ``finish_reason="refusal"``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from rag_service.core.config import LLMConfig
from rag_service.core.errors import (
    ConfigurationError,
    DependencyMissingError,
    ProviderError,
    ProviderUnavailableError,
)
from rag_service.core.registry import LLMS
from rag_service.domain.models import LLMRequest, LLMResponse, LLMStreamEvent, TokenUsage

_FINISH = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "refusal",
    "RECITATION": "refusal",
    "BLOCKLIST": "refusal",
    "PROHIBITED_CONTENT": "refusal",
    "SPII": "refusal",
}


def _text(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    content = getattr(candidates[0], "content", None) if candidates else None
    parts = getattr(content, "parts", None) or []
    return "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))


def _finish(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    if reason is None:
        feedback = getattr(response, "prompt_feedback", None)
        return "refusal" if getattr(feedback, "block_reason", None) else None
    name = getattr(reason, "name", str(reason))
    return _FINISH.get(name, "stop")


def _usage(response: Any) -> TokenUsage | None:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    return TokenUsage(
        input_tokens=meta.prompt_token_count or 0,
        output_tokens=(meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0),
    )


class GeminiLLM:
    provider = "gemini"

    def __init__(self, config: LLMConfig) -> None:
        try:
            from google import genai  # noqa: PLC0415 - optional dependency
            from google.genai import errors, types  # noqa: PLC0415
        except ImportError as exc:
            raise DependencyMissingError.for_extra("google-genai", "gemini") from exc
        if not config.model or config.model == "extractive":
            raise ConfigurationError(
                "llm.model must name the Gemini model when provider=gemini (e.g. gemini-3.8-flash)"
            )
        self._errors = errors
        self._types = types
        self.config = config
        self.model = config.model
        opts = config.options
        http = types.HttpOptions(
            timeout=int(config.timeout_s * 1000),  # milliseconds
            retry_options=types.HttpRetryOptions(attempts=config.max_retries + 1),
            **({"base_url": config.base_url} if config.base_url else {}),
        )
        client_args: dict[str, Any] = {"http_options": http}
        if opts.get("vertexai"):
            client_args.update(vertexai=True, project=opts.get("project"), location=opts.get("location"))
        else:
            client_args["api_key"] = config.api_key.get_secret_value() if config.api_key else None
        try:
            self._client = genai.Client(**client_args)
        except ValueError as exc:  # raised for missing credentials / project
            raise ConfigurationError(f"Gemini client configuration: {exc}") from exc

    def _params(self, request: LLMRequest) -> dict[str, Any]:
        types = self._types
        system_parts = [request.system] if request.system else []
        system_parts += [m.content for m in request.messages if m.role == "system"]
        contents = [
            types.Content(role="model" if m.role == "assistant" else "user", parts=[types.Part(text=m.content)])
            for m in request.messages
            if m.role in ("user", "assistant")
        ]
        settings: dict[str, Any] = {"max_output_tokens": request.max_tokens or self.config.max_tokens}
        if system_parts:
            settings["system_instruction"] = "\n\n".join(system_parts)
        temperature = request.temperature if request.temperature is not None else self.config.temperature
        if temperature is not None:
            settings["temperature"] = temperature
        thinking = {
            k: self.config.options[k] for k in ("thinking_level", "thinking_budget") if k in self.config.options
        }
        if thinking:
            settings["thinking_config"] = types.ThinkingConfig(**thinking)
        return {"model": self.model, "contents": contents, "config": types.GenerateContentConfig(**settings)}

    def _map_error(self, exc: Exception) -> Exception:
        if isinstance(exc, self._errors.APIError):
            if exc.code == 429 or (exc.code or 0) >= 500:
                return ProviderUnavailableError(f"Gemini unavailable ({exc.code}): {exc.message}")
            return ProviderError(f"Gemini request rejected ({exc.code}): {exc.message}")
        return ProviderUnavailableError(f"Gemini unreachable: {exc}")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self._client.aio.models.generate_content(**self._params(request))
        except (self._errors.APIError, httpx.HTTPError) as exc:
            raise self._map_error(exc) from exc
        return LLMResponse(
            text=_text(response),
            model=getattr(response, "model_version", None) or self.model,
            finish_reason=_finish(response) or "stop",
            usage=_usage(response) or TokenUsage(),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        finish: str | None = None
        usage: TokenUsage | None = None
        model = self.model
        try:
            async for chunk in await self._client.aio.models.generate_content_stream(**self._params(request)):
                text = _text(chunk)
                if text:
                    yield LLMStreamEvent(delta=text)
                finish = _finish(chunk) or finish
                usage = _usage(chunk) or usage
                model = getattr(chunk, "model_version", None) or model
        except (self._errors.APIError, httpx.HTTPError) as exc:
            raise self._map_error(exc) from exc
        yield LLMStreamEvent(done=True, finish_reason=finish or "stop", usage=usage or TokenUsage(), model=model)

    async def close(self) -> None:
        await self._client.aio.aclose()


@LLMS.register(
    "gemini", aliases=("google",), description="Google Gemini (google-genai SDK; Developer API or Vertex AI)"
)
def _gemini(*, config: LLMConfig) -> GeminiLLM:
    return GeminiLLM(config)
