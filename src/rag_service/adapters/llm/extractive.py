"""Offline "LLM" that answers extractively from the supplied sources.

It needs no model or API key, which makes the full chat flow runnable in tests, CI and air-gapped
environments. It reads the ``<source id="n">`` blocks produced by the prompt builder, scores
sentences by term overlap with the question and returns the best ones with ``[n]`` citations.
Prompts without sources (query rewriting, multi-query, HyDE) are answered by echoing the question,
so those transforms degrade to "no-op" rather than failing.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from rag_service.adapters.chunking.text_splitters import split_sentences
from rag_service.adapters.embedding.local import tokenize
from rag_service.core.config import LLMConfig
from rag_service.core.registry import LLMS
from rag_service.domain.models import LLMRequest, LLMResponse, LLMStreamEvent, TokenUsage

_SOURCE = re.compile(r'<source id="(\d+)"[^>]*>(.*?)</source>', re.DOTALL)
_QUESTION = re.compile(r"(?:^|\n)(?:Question|Query):\s*(.+?)\s*$", re.DOTALL | re.IGNORECASE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s.*$", re.MULTILINE)
MAX_SENTENCES = 3


class ExtractiveLLM:
    provider = "extractive"

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.model = "extractive"

    @staticmethod
    def _question(prompt: str) -> str:
        match = _QUESTION.search(prompt)
        return (match.group(1) if match else prompt.strip().split("\n")[-1]).strip()

    def _answer(self, request: LLMRequest) -> str:
        prompt = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        question = self._question(prompt)
        sources = _SOURCE.findall(prompt)
        if not sources:
            return question
        q_terms = set(tokenize(question))
        scored: list[tuple[float, int, str, str]] = []
        for order, (sid, text) in enumerate(sources):
            body = _HEADING.sub("", text).strip()  # headings are titles, not answers
            for sentence in split_sentences(body):
                terms = set(tokenize(sentence))
                if not terms:
                    continue
                overlap = len(q_terms & terms) / (len(q_terms) or 1)
                scored.append((overlap, -order, sid, sentence.strip()))
        scored.sort(reverse=True)
        best = [s for s in scored if s[0] > 0][:MAX_SENTENCES] or scored[:1]
        if not best:
            return "The retrieved sources do not contain an answer."
        return " ".join(f"{sentence} [{sid}]" for _, _, sid, sentence in best)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        text = self._answer(request)
        return LLMResponse(text=text, model=self.model, usage=TokenUsage(0, len(text.split())))

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        text = self._answer(request)
        for word in re.findall(r"\S+\s*", text):
            yield LLMStreamEvent(delta=word)
        yield LLMStreamEvent(done=True, finish_reason="stop", usage=TokenUsage(0, len(text.split())), model=self.model)

    async def close(self) -> None:
        return None


@LLMS.register(
    "extractive",
    aliases=("none", "offline"),
    description="Offline extractive answers with citations (no model)",
)
def _extractive(*, config: LLMConfig) -> ExtractiveLLM:
    return ExtractiveLLM(config)
