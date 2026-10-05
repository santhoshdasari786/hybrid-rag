"""Second-stage re-rankers.

* ``fastembed`` — local ONNX cross-encoders (ms-marco MiniLM, BGE reranker, Jina reranker).
* ``http``      — any Cohere-compatible ``POST /rerank`` API: Cohere, Jina, Voyage, Hugging Face
  TEI, Infinity. Options: ``url`` (required), ``api_key``, ``timeout_s``, ``headers``.

A cross-encoder reads query and passage together, so it is far more precise than bi-encoder
similarity — but too slow for the whole corpus. Re-rank only the fused top ``candidates``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Any

import httpx

from rag_service.core.config import RerankConfig
from rag_service.core.errors import ConfigurationError, DependencyMissingError, ProviderError
from rag_service.core.registry import RERANKERS
from rag_service.domain.models import RetrievedChunk


def _apply(chunks: Sequence[RetrievedChunk], scores: Sequence[float]) -> list[RetrievedChunk]:
    out = []
    for chunk, score in zip(chunks, scores, strict=True):
        chunk.scores["rerank"] = round(float(score), 6)
        chunk.score = float(score)
        out.append(chunk)
    return sorted(out, key=lambda c: -c.score)


class FastEmbedReranker:
    def __init__(self, config: RerankConfig) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise DependencyMissingError.for_extra("fastembed", "fastembed") from exc
        self._cls = TextCrossEncoder
        self.model_id = config.model
        self.batch_size = config.batch_size
        self._options = {k: v for k, v in config.options.items() if k in ("cache_dir", "threads", "providers", "cuda")}
        self._model: Any = None
        self._lock = threading.Lock()

    def _get(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        self._model = self._cls(model_name=self.model_id, **self._options)
                    except Exception as exc:
                        raise ProviderError(f"Failed to load reranker '{self.model_id}': {exc}") from exc
        return self._model

    async def warmup(self) -> None:
        await asyncio.to_thread(self._get)

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        scores = await asyncio.to_thread(lambda: list(self._get().rerank(query, texts, batch_size=self.batch_size)))
        return _apply(chunks, scores)


class HttpReranker:
    def __init__(self, config: RerankConfig) -> None:
        url = config.options.get("url")
        if not url:
            raise ConfigurationError("rerank.options.url is required for the http reranker")
        self.model_id = config.model
        self._url = str(url)
        headers = dict(config.options.get("headers", {}))
        if config.options.get("api_key"):
            headers["Authorization"] = f"Bearer {config.options['api_key']}"
        self._client = httpx.AsyncClient(timeout=float(config.options.get("timeout_s", 30)), headers=headers)

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        body = {
            "model": self.model_id,
            "query": query,
            "documents": [c.text for c in chunks],
            "top_n": len(chunks),
        }
        try:
            response = await self._client.post(self._url, json=body)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"Rerank request failed: {exc}") from exc
        results = data.get("results", data) if isinstance(data, dict) else data
        scores = [0.0] * len(chunks)
        for item in results:
            scores[int(item["index"])] = float(item.get("relevance_score", item.get("score", 0.0)))
        return _apply(chunks, scores)

    async def close(self) -> None:
        await self._client.aclose()


@RERANKERS.register("fastembed", description="Local ONNX cross-encoder (FastEmbed)")
def _fastembed(*, config: RerankConfig) -> FastEmbedReranker:
    return FastEmbedReranker(config)


@RERANKERS.register("http", aliases=("cohere", "tei", "jina"), description="Cohere-compatible /rerank HTTP API")
def _http(*, config: RerankConfig) -> HttpReranker:
    return HttpReranker(config)
