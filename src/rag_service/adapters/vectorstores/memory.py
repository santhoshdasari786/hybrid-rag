"""In-process vector store (NumPy). Exact search, full filter support, BM25 IDF for sparse vectors.

Intended for tests, local development and small single-replica deployments. Data lives only in
process memory.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rag_service.core.config import DistanceMetric, FusionConfig, VectorStoreConfig
from rag_service.core.errors import CapabilityNotSupportedError, CollectionNotFoundError
from rag_service.core.registry import VECTOR_STORES
from rag_service.core.utils import deep_merge
from rag_service.domain.filters import FilterSpec, matches
from rag_service.domain.models import CollectionSpec, PayloadPatch, ScoredPoint, SparseVector, VectorRecord
from rag_service.domain.ports import StoreCapabilities


@dataclass
class _Collection:
    spec: CollectionSpec
    payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    dense: dict[str, np.ndarray] = field(default_factory=dict)
    sparse: dict[str, dict[int, float]] = field(default_factory=dict)
    doc_freq: dict[int, int] = field(default_factory=dict)

    def _add_df(self, vec: dict[int, float], sign: int) -> None:
        for idx in vec:
            self.doc_freq[idx] = self.doc_freq.get(idx, 0) + sign
            if self.doc_freq[idx] <= 0:
                del self.doc_freq[idx]

    def put(self, record: VectorRecord) -> None:
        self.remove(record.id)
        self.payloads[record.id] = dict(record.payload)
        if record.dense is not None:
            self.dense[record.id] = np.asarray(record.dense, dtype=np.float32)
        if record.sparse is not None:
            vec = record.sparse.as_dict()
            self.sparse[record.id] = vec
            self._add_df(vec, +1)

    def remove(self, point_id: str) -> None:
        self.payloads.pop(point_id, None)
        self.dense.pop(point_id, None)
        old = self.sparse.pop(point_id, None)
        if old:
            self._add_df(old, -1)


class InMemoryVectorStore:
    name = "memory"
    capabilities = StoreCapabilities(dense=True, sparse=True, native_hybrid=False, sparse_idf=True)

    def __init__(self) -> None:
        self._collections: dict[str, _Collection] = {}
        self._lock = asyncio.Lock()

    def _get(self, name: str) -> _Collection:
        try:
            return self._collections[name]
        except KeyError:
            raise CollectionNotFoundError(f"Vector collection '{name}' does not exist") from None

    # ------------------------------------------------------------------ admin

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        async with self._lock:
            if spec.name not in self._collections:
                self._collections[spec.name] = _Collection(spec=spec)

    async def drop_collection(self, name: str) -> None:
        async with self._lock:
            self._collections.pop(name, None)

    async def collection_exists(self, name: str) -> bool:
        return name in self._collections

    async def count(self, name: str, filter_: FilterSpec | None = None) -> int:
        col = self._get(name)
        return sum(1 for p in col.payloads.values() if matches(filter_, p))

    # ------------------------------------------------------------------ writes

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        col = self._get(collection)
        async with self._lock:
            for record in records:
                col.put(record)

    async def update_payloads(self, collection: str, patches: Sequence[PayloadPatch]) -> None:
        col = self._get(collection)
        async with self._lock:
            for patch in patches:
                if patch.id in col.payloads:
                    col.payloads[patch.id] = {**col.payloads[patch.id], **patch.payload}

    async def delete_ids(self, collection: str, ids: Sequence[str]) -> None:
        col = self._get(collection)
        async with self._lock:
            for point_id in ids:
                col.remove(point_id)

    async def delete_by_filter(self, collection: str, filter_: FilterSpec) -> None:
        col = self._get(collection)
        async with self._lock:
            for point_id in [pid for pid, p in col.payloads.items() if matches(filter_, p)]:
                col.remove(point_id)

    # ------------------------------------------------------------------ reads

    def _candidates(self, col: _Collection, filter_: FilterSpec | None) -> list[str]:
        return [pid for pid, payload in col.payloads.items() if matches(filter_, payload)]

    def _point(self, col: _Collection, pid: str, score: float, with_vectors: bool) -> ScoredPoint:
        dense = col.dense[pid].tolist() if with_vectors and pid in col.dense else None
        return ScoredPoint(id=pid, score=score, payload=deep_merge({}, col.payloads[pid]), dense=dense)

    async def search_dense(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int,
        filter_: FilterSpec | None = None,
        score_threshold: float | None = None,
        with_vectors: bool = False,
    ) -> list[ScoredPoint]:
        col = self._get(collection)
        ids = [pid for pid in self._candidates(col, filter_) if pid in col.dense]
        if not ids:
            return []
        matrix = np.stack([col.dense[pid] for pid in ids])
        query = np.asarray(vector, dtype=np.float32)
        metric = col.spec.distance or DistanceMetric.COSINE
        if metric == DistanceMetric.COSINE:
            denom = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
            scores = (matrix @ query) / np.where(denom == 0, 1.0, denom)
        elif metric == DistanceMetric.DOT:
            scores = matrix @ query
        else:  # euclid: higher is better → negative distance
            scores = -np.linalg.norm(matrix - query, axis=1)
        order = np.argsort(-scores)[:limit]
        out = []
        for i in order:
            score = float(scores[i])
            if score_threshold is not None and score < score_threshold:
                continue
            out.append(self._point(col, ids[i], score, with_vectors))
        return out

    async def search_sparse(
        self,
        collection: str,
        vector: SparseVector,
        *,
        limit: int,
        filter_: FilterSpec | None = None,
    ) -> list[ScoredPoint]:
        col = self._get(collection)
        query = vector.as_dict()
        if not query:
            return []
        n_docs = max(1, len(col.sparse))
        idf: dict[int, float] = {}
        if col.spec.sparse_idf:
            for idx in query:
                df = col.doc_freq.get(idx, 0)
                idf[idx] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        allowed = set(self._candidates(col, filter_)) if filter_ else None
        scored: list[tuple[float, str]] = []
        for pid, doc in col.sparse.items():
            if allowed is not None and pid not in allowed:
                continue
            score = 0.0
            for idx, qv in query.items():
                dv = doc.get(idx)
                if dv is not None:
                    score += qv * dv * idf.get(idx, 1.0)
            if score > 0:
                scored.append((score, pid))
        scored.sort(key=lambda s: -s[0])
        return [self._point(col, pid, score, False) for score, pid in scored[:limit]]

    async def search_hybrid(
        self,
        collection: str,
        dense: Sequence[float],
        sparse: SparseVector,
        *,
        limit: int,
        candidate_k: int,
        fusion: FusionConfig,
        filter_: FilterSpec | None = None,
    ) -> list[ScoredPoint]:
        raise CapabilityNotSupportedError("The memory store has no native hybrid search; use fusion.execution=client")

    async def fetch(self, collection: str, ids: Sequence[str], *, with_vectors: bool = False) -> list[ScoredPoint]:
        col = self._get(collection)
        return [self._point(col, pid, 0.0, with_vectors) for pid in ids if pid in col.payloads]

    async def scroll(
        self,
        collection: str,
        *,
        filter_: FilterSpec | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[ScoredPoint], str | None]:
        col = self._get(collection)
        ids = sorted(self._candidates(col, filter_))
        start = ids.index(offset) if offset in ids else 0
        page = ids[start : start + limit]
        next_offset = ids[start + limit] if start + limit < len(ids) else None
        return [self._point(col, pid, 0.0, False) for pid in page], next_offset

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


@VECTOR_STORES.register("memory", aliases=("inmemory",), description="In-process NumPy store (tests / local)")
def _memory(*, config: VectorStoreConfig) -> InMemoryVectorStore:
    return InMemoryVectorStore()
