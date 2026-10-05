"""Dense embeddings from any OpenAI-compatible ``/v1/embeddings`` endpoint.

Works with OpenAI, Azure OpenAI (``base_url``), Hugging Face TEI, vLLM, Ollama, LM Studio and
LiteLLM. Install with the ``openai`` extra.

Options (``embedding.dense.options``): ``api_key``, ``base_url``, ``timeout_s``, ``max_retries``,
``send_dimensions`` (pass ``dimensions`` for Matryoshka models such as text-embedding-3-*).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from rag_service.core.config import DenseEmbeddingConfig
from rag_service.core.errors import DependencyMissingError, ProviderError
from rag_service.core.registry import DENSE_EMBEDDERS
from rag_service.core.utils import batched


class OpenAICompatibleEmbedder:
    def __init__(self, config: DenseEmbeddingConfig) -> None:
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise DependencyMissingError.for_extra("openai", "openai") from exc
        opts: dict[str, Any] = dict(config.options)
        self._client = AsyncOpenAI(
            api_key=opts.get("api_key"),
            base_url=opts.get("base_url"),
            timeout=float(opts.get("timeout_s", 60)),
            max_retries=int(opts.get("max_retries", 2)),
        )
        self.model_id = config.model
        self.distance = config.distance
        self.batch_size = min(config.batch_size, 2048)
        self.query_prefix = config.query_prefix
        self.document_prefix = config.document_prefix
        self._send_dimensions = bool(opts.get("send_dimensions", False))
        self.dimension = config.dimension or 0
        self._probe_lock = asyncio.Lock()

    async def warmup(self) -> None:
        """Discover the output dimension when it is not configured."""
        if self.dimension:
            return
        async with self._probe_lock:
            if not self.dimension:
                self.dimension = len((await self._embed(["dimension probe"]))[0])

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model_id, "input": list(texts)}
        if self._send_dimensions and self.dimension:
            kwargs["dimensions"] = self.dimension
        try:
            response = await self._client.embeddings.create(**kwargs)
        except Exception as exc:
            raise ProviderError(f"Embedding request failed ({self.model_id}): {exc}") from exc
        ordered = sorted(response.data, key=lambda d: d.index)
        return [list(d.embedding) for d in ordered]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for batch in batched([self.document_prefix + t for t in texts], self.batch_size):
            out.extend(await self._embed(batch))
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([self.query_prefix + text]))[0]

    async def close(self) -> None:
        await self._client.close()


@DENSE_EMBEDDERS.register("openai", aliases=("openai-compatible",), description="OpenAI-compatible /v1/embeddings")
def _openai(*, config: DenseEmbeddingConfig) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(config)
