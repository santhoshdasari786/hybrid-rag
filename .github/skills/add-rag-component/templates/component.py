"""<One line: what this component does and when to use it>.

Options (``<config section>.options``): ``example_option`` (default …).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from rag_service.core.config import RerankConfig  # ← the config model for your registry
from rag_service.core.errors import DependencyMissingError, ProviderError, ProviderUnavailableError
from rag_service.core.registry import RERANKERS  # ← your registry
from rag_service.domain.models import RetrievedChunk


class ExampleReranker:
    """Implements ``rag_service.domain.ports.Reranker``."""

    def __init__(self, config: RerankConfig) -> None:
        try:
            import some_sdk  # noqa: F401, PLC0415 - optional dependency
        except ImportError as exc:
            raise DependencyMissingError.for_extra("some-sdk", "example") from exc
        self.model_id = config.model
        self._option = config.options.get("example_option", 1)
        self._client: Any = None  # create lazily / here

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        try:
            scores = await asyncio.to_thread(self._score, query, [c.text for c in chunks])
        except TimeoutError as exc:
            raise ProviderUnavailableError(f"Example reranker timed out: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"Example reranker failed: {exc}") from exc
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.scores["rerank"] = round(float(score), 6)
            chunk.score = float(score)
        return sorted(chunks, key=lambda c: -c.score)

    def _score(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError


@RERANKERS.register("example", description="<one line shown in /system/info>")
def _example(*, config: RerankConfig) -> ExampleReranker:
    return ExampleReranker(config)
