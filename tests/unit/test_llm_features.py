"""LLM-driven features with a scripted fake LLM: query transforms, contextual enrichment,
history condensation — plus request construction of the real provider adapters (no network)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import pytest

from rag_service.application.ingestion import DocumentInput
from rag_service.container import Container
from rag_service.core.config import LLMConfig, QueryTransformConfig
from rag_service.core.registry import LLMS, QUERY_TRANSFORMS
from rag_service.domain.models import ChatMessage, LLMRequest, LLMResponse, LLMStreamEvent, TokenUsage
from tests.conftest import make_settings


class FakeLLM:
    """Answers from a script keyed by prompt substrings; records every request."""

    provider = "fake"
    requests: ClassVar[list[LLMRequest]] = []
    script: ClassVar[dict[str, str]] = {}

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.model = "fake-1"

    def _reply(self, request: LLMRequest) -> str:
        FakeLLM.requests.append(request)
        prompt = "\n".join([request.system or "", *(m.content for m in request.messages)])
        for needle, answer in FakeLLM.script.items():
            if needle in prompt:
                return answer
        return "Grounded answer [1]."

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text=self._reply(request), model=self.model, usage=TokenUsage(10, 5))

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        for word in self._reply(request).split(" "):
            yield LLMStreamEvent(delta=word + " ")
        yield LLMStreamEvent(done=True, finish_reason="stop", usage=TokenUsage(10, 5), model=self.model)

    async def close(self) -> None:
        return None


@LLMS.register("fake", description="test-only")
def _fake(*, config: LLMConfig) -> FakeLLM:
    return FakeLLM(config)


@pytest.fixture(autouse=True)
def _reset() -> None:
    FakeLLM.requests.clear()
    FakeLLM.script.clear()


async def test_multi_query_generates_variants_and_keeps_original() -> None:
    FakeLLM.script["different search queries"] = "1. quiet bathroom fan\n2. low sone exhaust fan\n- quiet bathroom fan"
    transformer = QUERY_TRANSFORMS.create("multi_query", config=QueryTransformConfig(num_queries=3), llm=FakeLLM())
    variants = await transformer.transform("silent exhaust for small room")
    assert [v.text for v in variants] == ["silent exhaust for small room", "quiet bathroom fan", "low sone exhaust fan"]


async def test_hyde_embeds_passage_densely_and_query_sparsely() -> None:
    FakeLLM.script["short, factual passage"] = "This ceiling fan runs at 0.3 sones and moves 80 CFM."
    variants = await QUERY_TRANSFORMS.create(
        "hyde", config=QueryTransformConfig(include_original=False), llm=FakeLLM()
    ).transform("quiet fan?")
    hyde = next(v for v in variants if v.kind == "hyde")
    literal = next(v for v in variants if v.kind == "original")
    assert hyde.embed_dense and not hyde.embed_sparse
    assert literal.embed_sparse and not literal.embed_dense


async def test_transform_failure_falls_back_to_original() -> None:
    class Broken(FakeLLM):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            raise RuntimeError("provider down")

    variants = await QUERY_TRANSFORMS.create("rewrite", config=QueryTransformConfig(), llm=Broken()).transform("q")
    assert [v.text for v in variants] == ["q"]


async def fake_container(**pipeline: Any) -> Container:
    settings = make_settings(
        llm={"provider": "fake", "model": "fake-1"}, collections={"products": {"pipeline": pipeline}}
    )
    c = Container.build(settings)
    await c.start()
    return c


async def test_llm_contextual_enrichment_changes_embedded_text_only() -> None:
    FakeLLM.script["situate this excerpt"] = "This excerpt describes the Delta fan's noise rating."
    c = await fake_container(enrichment={"llm_context": {"enabled": True}})
    try:
        await c.ingestion.ingest(
            "products", [DocumentInput(source_id="d", content="Rated at 1.5 sones.", title="Delta RAD80")]
        )
        points, _ = await c.store.scroll("products", limit=10)
        assert points[0].payload["text"] == "Rated at 1.5 sones."  # returned text is untouched
        assert points[0].payload["metadata"]["llm_context"].startswith("This excerpt describes")
        enrichment_calls = [r for r in FakeLLM.requests if r.cache_system]
        assert enrichment_calls and "<document>" in (enrichment_calls[0].system or "")
    finally:
        await c.stop()


async def test_chat_condenses_follow_up_questions() -> None:
    FakeLLM.script["standalone search query"] = "Delta RAD80 noise level in sones"
    c = await fake_container()
    try:
        await c.ingestion.ingest(
            "products", [DocumentInput(source_id="d", content="The Delta RAD80 is rated at 1.5 sones.")]
        )
        result = await c.chat.answer(
            "products",
            [
                ChatMessage("user", "Tell me about the Delta RAD80"),
                ChatMessage("assistant", "It is an 80 CFM fan [1]."),
                ChatMessage("user", "How loud is it?"),
            ],
        )
        assert result.standalone_query == "Delta RAD80 noise level in sones"
        generation = FakeLLM.requests[-1]
        assert [m.role for m in generation.messages] == ["user", "assistant", "user"]
        assert '<source id="1"' in generation.messages[-1].content
        assert result.citations and result.citations[0].source_id == "d"
    finally:
        await c.stop()


def test_anthropic_request_shape() -> None:
    pytest.importorskip("anthropic")
    from rag_service.adapters.llm.anthropic_llm import DEFAULT_MODEL, AnthropicLLM

    llm = AnthropicLLM(LLMConfig(provider="anthropic", model="", api_key="test-key"))
    assert llm.model == DEFAULT_MODEL
    params = llm._params(
        LLMRequest(
            system="Be precise.",
            messages=[ChatMessage("system", "Extra rule."), ChatMessage("user", "hi")],
            cache_system=True,
        )
    )
    assert params["messages"] == [{"role": "user", "content": "hi"}]
    assert params["system"][0]["text"] == "Be precise.\n\nExtra rule."
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in params  # never sent unless configured
    assert params["max_tokens"] == 16_000


def test_openai_request_shape() -> None:
    pytest.importorskip("openai")
    from rag_service.adapters.llm.openai_llm import OpenAICompatibleLLM

    official = OpenAICompatibleLLM(LLMConfig(provider="openai", model="gpt-4o", api_key="k"))
    params = official._params(
        LLMRequest(system="S", messages=[ChatMessage("user", "hi")], max_tokens=50, temperature=0.2)
    )
    assert params["messages"][0] == {"role": "system", "content": "S"}
    assert params["max_completion_tokens"] == 50 and params["temperature"] == 0.2

    local = OpenAICompatibleLLM(
        LLMConfig(provider="openai", model="llama3", base_url="http://localhost:11434/v1", api_key="k")
    )
    assert "max_tokens" in local._params(LLMRequest(messages=[ChatMessage("user", "hi")]))


async def test_llm_features_are_skipped_without_a_generative_llm() -> None:
    settings = make_settings(
        llm={"provider": "extractive"},
        collections={"products": {"pipeline": {"enrichment": {"llm_context": {"enabled": True}}}}},
    )
    c = Container.build(settings)
    await c.start()
    try:
        transformer = c.factory.query_transformer(QueryTransformConfig(strategy="multi_query"))
        assert [v.text for v in await transformer.transform("quiet fan")] == ["quiet fan"]
        pipeline = await c.collections.pipeline("products")
        assert c.factory.llm_context_enricher(pipeline.config) is None
    finally:
        await c.stop()
