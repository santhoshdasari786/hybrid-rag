"""Decorators that add query-embedding caching to any embedder (Decorator pattern).

Queries repeat far more often than documents (dashboards, evaluation runs, retries), so the
decorators cache ``embed_query`` in a bounded LRU while delegating everything else unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Generic, TypeVar

from rag_service.core.config import DistanceMetric
from rag_service.domain.models import SparseVector
from rag_service.domain.ports import DenseEmbedder, SparseEmbedder

V = TypeVar("V")


class _LRU(Generic[V]):
    def __init__(self, size: int) -> None:
        self.size = size
        self._data: OrderedDict[str, V] = OrderedDict()

    def get(self, key: str) -> V | None:
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key: str, value: V) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.size:
            self._data.popitem(last=False)


class CachedDenseEmbedder:
    def __init__(self, inner: DenseEmbedder, size: int) -> None:
        self.inner = inner
        self._cache: _LRU[list[float]] = _LRU(size)

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    @property
    def distance(self) -> DistanceMetric:
        return self.inner.distance

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.inner.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is None:
            cached = await self.inner.embed_query(text)
            self._cache.put(text, cached)
        return cached

    async def warmup(self) -> None:
        warm = getattr(self.inner, "warmup", None)
        if warm is not None:
            await warm()


class CachedSparseEmbedder:
    def __init__(self, inner: SparseEmbedder, size: int) -> None:
        self.inner = inner
        self._cache: _LRU[SparseVector] = _LRU(size)

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def requires_idf(self) -> bool:
        return self.inner.requires_idf

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return await self.inner.embed_documents(texts)

    async def embed_query(self, text: str) -> SparseVector:
        cached = self._cache.get(text)
        if cached is None:
            cached = await self.inner.embed_query(text)
            self._cache.put(text, cached)
        return cached

    async def warmup(self) -> None:
        warm = getattr(self.inner, "warmup", None)
        if warm is not None:
            await warm()
