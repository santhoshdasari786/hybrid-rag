"""Retrieval-augmented chat.

1. **Condense** — a follow-up ("what about the quieter one?") is rewritten into a standalone query
   using the conversation history (utility LLM; skipped for the extractive provider).
2. **Retrieve** — the full retrieval pipeline with request-level overrides and filters.
3. **Ground** — numbered ``<source>`` blocks are packed into the prompt within a token budget.
   When nothing relevant is retrieved the configured ``no_context_answer`` is returned without
   calling the LLM (cheaper, and it prevents ungrounded answers).
4. **Generate** — blocking or streaming.
5. **Cite** — ``[n]`` markers in the answer are resolved to chunk/source citations.
"""

from __future__ import annotations

import html
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from rag_service.adapters.chunking.length import ApproxTokenLength
from rag_service.application.pipeline import PipelineFactory
from rag_service.application.retrieval import RetrievalResult, RetrievalService
from rag_service.core.config import GenerationConfig, merge_pipeline
from rag_service.core.errors import ValidationFailedError
from rag_service.core.logging import get_logger
from rag_service.domain.filters import FilterSpec
from rag_service.domain.models import ChatMessage, LLMRequest, RetrievedChunk, TokenUsage

log = get_logger(__name__)

_CITATION = re.compile(r"\[(\d{1,3})\]")
_approx_tokens = ApproxTokenLength()

CONDENSE_PROMPT = (
    "Given the conversation below, rewrite the user's last message as a standalone search query that "
    "can be understood without the conversation. Keep product names, numbers and constraints. "
    "Return only the query.\n\n<conversation>\n{history}\n</conversation>\n\nQuestion: {question}"
)


@dataclass(slots=True)
class Citation:
    index: int
    chunk_id: str
    source_id: str
    title: str | None
    snippet: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatResult:
    answer: str
    citations: list[Citation]
    sources: list[Citation]
    retrieval: RetrievalResult
    standalone_query: str
    model: str
    finish_reason: str
    usage: TokenUsage


@dataclass(slots=True)
class ChatStreamEvent:
    """``type`` is one of ``retrieval``, ``delta``, ``done``."""

    type: str
    data: dict[str, Any]


@dataclass(slots=True)
class PreparedChat:
    question: str
    standalone: str
    history: list[ChatMessage]
    retrieval: RetrievalResult
    sources: list[Citation]
    request: LLMRequest | None
    generation: GenerationConfig


class ChatService:
    def __init__(self, factory: PipelineFactory, retrieval: RetrievalService) -> None:
        self.factory = factory
        self.retrieval = retrieval

    # ------------------------------------------------------------------ public API

    async def answer(
        self,
        collection: str,
        messages: list[ChatMessage],
        *,
        filter_: FilterSpec | None = None,
        retrieval_overrides: dict[str, Any] | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ) -> ChatResult:
        prepared = await self._prepare(collection, messages, filter_, retrieval_overrides, generation_overrides)
        if prepared.request is None:
            return self._no_context(prepared)
        llm = self.factory.llm()
        response = await llm.generate(prepared.request)
        citations = self._cited(response.text, prepared.sources)
        return ChatResult(
            answer=response.text,
            citations=citations,
            sources=prepared.sources,
            retrieval=prepared.retrieval,
            standalone_query=prepared.standalone,
            model=response.model,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

    async def prepare(
        self,
        collection: str,
        messages: list[ChatMessage],
        *,
        filter_: FilterSpec | None = None,
        retrieval_overrides: dict[str, Any] | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ) -> PreparedChat:
        """Validate, condense and retrieve. Call before opening a stream so errors map to HTTP statuses."""
        return await self._prepare(collection, messages, filter_, retrieval_overrides, generation_overrides)

    async def stream(self, prepared: PreparedChat) -> AsyncIterator[ChatStreamEvent]:
        yield ChatStreamEvent("retrieval", {"standalone_query": prepared.standalone, "sources": prepared.sources})
        if prepared.request is None:
            result = self._no_context(prepared)
            yield ChatStreamEvent("delta", {"content": result.answer})
            yield ChatStreamEvent(
                "done",
                {
                    "citations": [],
                    "finish_reason": result.finish_reason,
                    "usage": result.usage,
                    "model": result.model,
                },
            )
            return
        llm = self.factory.llm()
        parts: list[str] = []
        final_usage, finish, model = TokenUsage(), "stop", llm.model
        async for event in llm.stream(prepared.request):
            if event.done:
                final_usage = event.usage or final_usage
                finish = event.finish_reason or finish
                model = event.model or model
            elif event.delta:
                parts.append(event.delta)
                yield ChatStreamEvent("delta", {"content": event.delta})
        answer = "".join(parts)
        yield ChatStreamEvent(
            "done",
            {
                "citations": self._cited(answer, prepared.sources),
                "finish_reason": finish,
                "usage": final_usage,
                "model": model,
                "answer": answer,
            },
        )

    # ------------------------------------------------------------------ steps

    async def _prepare(
        self,
        collection: str,
        messages: list[ChatMessage],
        filter_: FilterSpec | None,
        retrieval_overrides: dict[str, Any] | None,
        generation_overrides: dict[str, Any] | None,
    ) -> PreparedChat:
        if not messages or messages[-1].role != "user" or not messages[-1].content.strip():
            raise ValidationFailedError("The last message must be a non-empty user message")
        pipeline = await self.retrieval.collections.pipeline(collection)
        generation = (
            merge_pipeline(pipeline.config, {"generation": generation_overrides}).generation
            if generation_overrides
            else pipeline.config.generation
        )
        question = messages[-1].content.strip()
        history = [m for m in messages[:-1] if m.role in ("user", "assistant")]
        history = history[-generation.history_turns * 2 :] if generation.history_turns else []

        standalone = await self._condense(question, history) if generation.condense_history and history else question
        retrieval = await self.retrieval.retrieve(
            collection, standalone, filter_=filter_, overrides=retrieval_overrides
        )
        sources, context = self._build_context(retrieval.chunks, generation.max_context_tokens)
        if not sources:
            return PreparedChat(question, standalone, history, retrieval, [], None, generation)

        user_prompt = f"<sources>\n{context}\n</sources>\n\nQuestion: {question}"
        request = LLMRequest(
            system=generation.system_prompt,
            messages=[*history, ChatMessage(role="user", content=user_prompt)],
            max_tokens=generation.max_tokens,
            temperature=generation.temperature,
        )
        return PreparedChat(question, standalone, history, retrieval, sources, request, generation)

    async def _condense(self, question: str, history: list[ChatMessage]) -> str:
        llm = self.factory.utility_llm()
        if not self.factory.is_generative(llm):
            return question
        transcript = "\n".join(f"{m.role}: {m.content}" for m in history)
        try:
            response = await llm.generate(
                LLMRequest(
                    messages=[
                        ChatMessage(role="user", content=CONDENSE_PROMPT.format(history=transcript, question=question))
                    ],
                    max_tokens=200,
                )
            )
            rewritten = response.text.strip().splitlines()[0].strip() if response.text.strip() else ""
            return rewritten or question
        except Exception as exc:  # fall back to the raw question
            log.warning("chat.condense_failed", error=str(exc))
            return question

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk], budget_tokens: int) -> tuple[list[Citation], str]:
        sources: list[Citation] = []
        blocks: list[str] = []
        used = 0
        for chunk in chunks:
            body = chunk.context or chunk.text
            cost = _approx_tokens(body) + 20
            if sources and used + cost > budget_tokens:
                break
            index = len(sources) + 1
            title_attr = f' title="{html.escape(chunk.title, quote=True)}"' if chunk.title else ""
            source_attr = html.escape(chunk.source_id, quote=True)
            blocks.append(f'<source id="{index}"{title_attr} source_id="{source_attr}">\n{body}\n</source>')
            sources.append(
                Citation(
                    index=index,
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    title=chunk.title,
                    snippet=chunk.text[:400],
                    score=round(chunk.score, 6),
                    metadata=chunk.metadata,
                )
            )
            used += cost
        return sources, "\n\n".join(blocks)

    @staticmethod
    def _cited(answer: str, sources: list[Citation]) -> list[Citation]:
        by_index = {s.index: s for s in sources}
        seen: list[int] = []
        for match in _CITATION.finditer(answer):
            n = int(match.group(1))
            if n in by_index and n not in seen:
                seen.append(n)
        return [by_index[n] for n in seen]

    def _no_context(self, prepared: PreparedChat) -> ChatResult:
        return ChatResult(
            answer=prepared.generation.no_context_answer,
            citations=[],
            sources=[],
            retrieval=prepared.retrieval,
            standalone_query=prepared.standalone,
            model="none",
            finish_reason="no_context",
            usage=TokenUsage(),
        )
