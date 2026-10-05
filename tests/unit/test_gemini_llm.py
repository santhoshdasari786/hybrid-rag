"""``provider: gemini`` request construction and response mapping with real ``google-genai`` types and
a fake transport (no network)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from rag_service.core.config import LLMConfig
from rag_service.core.errors import ConfigurationError, ProviderError, ProviderUnavailableError
from rag_service.domain.models import ChatMessage, LLMRequest

pytest.importorskip("google.genai")

from google.genai import errors, types

from rag_service.adapters.llm.gemini_llm import GeminiLLM


def response(*parts: types.Part, finish: str | None = "STOP", usage: bool = True) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)),
                finish_reason=types.FinishReason(finish) if finish else None,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=3, thoughts_token_count=4
        )
        if usage
        else None,
        model_version="gemini-3.8-flash-001",
    )


class FakeModels:
    def __init__(self, result: Any = None, chunks: list[Any] | None = None, error: Exception | None = None) -> None:
        self.result, self.chunks, self.error = result, chunks or [], error
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **params: Any) -> Any:
        self.calls.append(params)
        if self.error:
            raise self.error
        return self.result

    async def generate_content_stream(self, **params: Any) -> AsyncIterator[Any]:
        self.calls.append(params)

        async def chunks() -> AsyncIterator[Any]:
            for chunk in self.chunks:
                yield chunk

        return chunks()


def make_llm(models: FakeModels | None = None, **config: Any) -> GeminiLLM:
    llm = GeminiLLM(LLMConfig(provider="gemini", model="gemini-3.8-flash", api_key="k", **config))
    if models is not None:
        llm._client = SimpleNamespace(aio=SimpleNamespace(models=models))  # type: ignore[assignment]
    return llm


def test_request_shape() -> None:
    llm = make_llm(options={"thinking_level": "low"})
    params = llm._params(
        LLMRequest(
            system="Be precise.",
            messages=[
                ChatMessage("system", "Cite sources."),
                ChatMessage("user", "hi"),
                ChatMessage("assistant", "hello"),
                ChatMessage("user", "q"),
            ],
        )
    )
    assert params["model"] == "gemini-3.8-flash"
    assert [c.role for c in params["contents"]] == ["user", "model", "user"]
    config: types.GenerateContentConfig = params["config"]
    assert config.system_instruction == "Be precise.\n\nCite sources."
    assert config.max_output_tokens == 16_000
    assert config.temperature is None  # never sent unless configured
    assert config.thinking_config is not None and config.thinking_config.thinking_level == types.ThinkingLevel.LOW


async def test_generate_skips_thoughts_and_counts_their_tokens() -> None:
    models = FakeModels(result=response(types.Part(text="planning…", thought=True), types.Part(text="Answer [1].")))
    result = await make_llm(models).generate(LLMRequest(messages=[ChatMessage("user", "q")]))
    assert result.text == "Answer [1]."
    assert (result.usage.input_tokens, result.usage.output_tokens) == (10, 7)
    assert result.finish_reason == "stop" and result.model == "gemini-3.8-flash-001"


async def test_stream_yields_deltas_then_usage() -> None:
    chunks = [
        response(types.Part(text="Hel"), finish=None, usage=False),
        response(types.Part(text="lo"), finish="MAX_TOKENS"),
    ]
    events = [
        e async for e in make_llm(FakeModels(chunks=chunks)).stream(LLMRequest(messages=[ChatMessage("user", "q")]))
    ]
    assert [e.delta for e in events if not e.done] == ["Hel", "lo"]
    done = events[-1]
    assert done.done and done.finish_reason == "length" and done.usage is not None and done.usage.output_tokens == 7


async def test_safety_block_is_a_refusal() -> None:
    blocked = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY")
    )
    result = await make_llm(FakeModels(result=blocked)).generate(LLMRequest(messages=[ChatMessage("user", "q")]))
    assert result.text == "" and result.finish_reason == "refusal"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
            ProviderUnavailableError,
        ),
        (errors.ServerError(503, {"error": {"message": "busy", "status": "UNAVAILABLE"}}), ProviderUnavailableError),
        (errors.ClientError(400, {"error": {"message": "bad", "status": "INVALID_ARGUMENT"}}), ProviderError),
    ],
)
async def test_api_errors_are_mapped(error: Exception, expected: type[Exception]) -> None:
    with pytest.raises(expected):
        await make_llm(FakeModels(error=error)).generate(LLMRequest(messages=[ChatMessage("user", "q")]))


def test_model_is_required() -> None:
    with pytest.raises(ConfigurationError, match="gemini"):
        GeminiLLM(LLMConfig(provider="gemini", model="", api_key="k"))
