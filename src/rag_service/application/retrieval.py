"""Retrieval pipeline.

::

    query ─► transform (none | multi_query | hyde | rewrite)
          ─► for each variant, in parallel: dense ANN search  ┐
                                            sparse search     ┘ candidate_k each
          ─► fusion (rrf | dbsf | rsf, weighted)  — or store-native hybrid
          ─► dedupe ─► rerank (cross-encoder) ─► group (one hit per product / page) ─► MMR
          ─► min_score ─► top_k
          ─► context expansion (neighbouring chunks)

Every stage is optional and configurable per collection and per request. With ``debug=True`` the
result contains each retriever's ranked list, the fused list and the stage timings, which is what
you need to tune a pipeline (compare dense vs sparse, see what the re-ranker changed…).
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from rag_service.adapters.fusion.mmr import mmr_select
from rag_service.application.collections import CollectionService
from rag_service.application.pipeline import CollectionPipeline, PipelineFactory
from rag_service.core.config import RetrievalConfig, RetrievalMode, merge_pipeline
from rag_service.core.errors import CapabilityNotSupportedError, ValidationFailedError
from rag_service.core.logging import get_logger
from rag_service.core.utils import Stopwatch, canonical_json, get_path, sha256_text
from rag_service.domain.filters import Condition, FilterOp, FilterSpec
from rag_service.domain.models import (
    FusedItem,
    PayloadKeys,
    QueryVariant,
    RankedList,
    RetrievedChunk,
    ScoredPoint,
)
from rag_service.domain.ports import VectorReader

log = get_logger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    config: RetrievalConfig
    mode: RetrievalMode
    variants: list[QueryVariant] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] | None = None


def _summarise(points: list[ScoredPoint], limit: int = 50) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": p.id,
            "source_id": p.payload.get(PayloadKeys.SOURCE_ID),
            "score": round(p.score, 6),
            "title": p.payload.get(PayloadKeys.TITLE),
        }
        for rank, p in enumerate(points[:limit], start=1)
    ]


class RetrievalService:
    def __init__(self, factory: PipelineFactory, collections: CollectionService, reader: VectorReader) -> None:
        self.factory = factory
        self.collections = collections
        self.reader = reader

    def effective_config(self, pipeline: CollectionPipeline, overrides: dict[str, Any] | None) -> RetrievalConfig:
        if not overrides:
            return pipeline.config.retrieval
        return merge_pipeline(pipeline.config, {"retrieval": overrides}).retrieval

    @staticmethod
    def _mode(pipeline: CollectionPipeline, cfg: RetrievalConfig) -> RetrievalMode:
        mode = cfg.mode
        if mode == RetrievalMode.HYBRID:
            if pipeline.dense is None:
                return RetrievalMode.SPARSE
            if pipeline.sparse is None:
                return RetrievalMode.DENSE
        if mode == RetrievalMode.DENSE and pipeline.dense is None:
            raise ValidationFailedError(f"Collection '{pipeline.name}' has no dense embeddings; use mode=sparse")
        if mode == RetrievalMode.SPARSE and pipeline.sparse is None:
            raise ValidationFailedError(f"Collection '{pipeline.name}' has no sparse embeddings; use mode=dense")
        return mode

    async def retrieve(
        self,
        collection: str,
        query: str,
        *,
        filter_: FilterSpec | None = None,
        overrides: dict[str, Any] | None = None,
        debug: bool = False,
    ) -> RetrievalResult:
        if not query.strip():
            raise ValidationFailedError("query must not be empty")
        pipeline = await self.collections.pipeline(collection)
        cfg = self.effective_config(pipeline, overrides)
        mode = self._mode(pipeline, cfg)
        sw = Stopwatch()
        dbg: dict[str, Any] | None = {"lists": []} if debug else None

        with sw.stage("query_transform"):
            variants = await self.factory.query_transformer(cfg.query_transform).transform(query)

        use_dense = mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID)
        use_sparse = mode in (RetrievalMode.SPARSE, RetrievalMode.HYBRID)
        fetch_vectors = cfg.mmr.enabled and use_dense

        native = (
            cfg.fusion.execution == "native"
            and mode == RetrievalMode.HYBRID
            and self.reader.capabilities.native_hybrid
            and len(variants) == 1
        )

        points_by_id: dict[str, ScoredPoint] = {}
        with sw.stage("search"):
            if native:
                fused = await self._native_hybrid(pipeline, variants[0], cfg, filter_, points_by_id)
            else:
                lists = await self._search_all(pipeline, variants, cfg, filter_, use_dense, use_sparse, fetch_vectors)
                for lst in lists:
                    for p in lst.items:
                        known = points_by_id.get(p.id)
                        if known is None or (known.dense is None and p.dense is not None):
                            points_by_id[p.id] = p
                    if dbg is not None:
                        dbg["lists"].append({"retriever": lst.name, "items": _summarise(lst.items)})
                fused = self._fuse(lists, cfg)

        chunks = []
        for item in fused:
            point = points_by_id.get(item.id)
            if point is None:
                continue
            chunk = RetrievedChunk.from_point(point, score=item.score)
            chunk.scores["fused"] = round(item.score, 6)
            for retriever, contribution in item.contributions.items():
                chunk.scores[f"{retriever}_rank"] = contribution["rank"]
                chunk.scores[f"{retriever}_score"] = round(contribution["score"], 6)
            chunks.append(chunk)
        if dbg is not None:
            dbg["fused"] = [
                {"rank": i, "chunk_id": c.chunk_id, "score": round(c.score, 6)} for i, c in enumerate(chunks[:50], 1)
            ]

        if cfg.dedupe:
            chunks = self._dedupe(chunks)

        if cfg.rerank.enabled and chunks:
            with sw.stage("rerank"):
                reranker = self.factory.reranker(cfg.rerank)
                chunks = await reranker.rerank(query, chunks[: cfg.rerank.candidates])
                if cfg.rerank.min_score is not None:
                    chunks = [c for c in chunks if c.scores.get("rerank", c.score) >= cfg.rerank.min_score]
            if dbg is not None:
                dbg["reranked"] = [
                    {"rank": i, "chunk_id": c.chunk_id, "score": c.score} for i, c in enumerate(chunks, 1)
                ]

        if cfg.group_by:
            chunks = self._group(chunks, cfg.group_by, cfg.group_size)

        if cfg.mmr.enabled and chunks and pipeline.dense is not None:
            with sw.stage("mmr"):
                query_vec = await pipeline.dense.embed_query(query)
                chunks = mmr_select(query_vec, chunks, k=cfg.top_k, lambda_mult=cfg.mmr.lambda_mult)

        if cfg.min_score is not None:
            chunks = [c for c in chunks if c.score >= cfg.min_score]
        chunks = chunks[: cfg.top_k]

        if cfg.context_expansion.enabled and chunks:
            with sw.stage("context_expansion"):
                await self._expand(pipeline.name, chunks, cfg.context_expansion.window)

        for chunk in chunks:
            chunk.dense = None  # internal only
        log.info(
            "retrieval.done",
            collection=collection,
            mode=mode.value,
            variants=len(variants),
            results=len(chunks),
            **{f"ms_{k}": v for k, v in sw.timings.items()},
        )
        return RetrievalResult(
            query=query,
            chunks=chunks,
            config=cfg,
            mode=mode,
            variants=variants,
            timings=sw.timings,
            debug=dbg,
        )

    # ------------------------------------------------------------------ stages

    async def _search_all(
        self,
        pipeline: CollectionPipeline,
        variants: list[QueryVariant],
        cfg: RetrievalConfig,
        filter_: FilterSpec | None,
        use_dense: bool,
        use_sparse: bool,
        fetch_vectors: bool,
    ) -> list[RankedList]:
        weights = cfg.fusion.weights

        async def dense(v: QueryVariant) -> RankedList:
            assert pipeline.dense is not None
            vector = await pipeline.dense.embed_query(v.text)
            items = await self.reader.search_dense(
                pipeline.name,
                vector,
                limit=cfg.candidate_k,
                filter_=filter_,
                score_threshold=cfg.dense_score_threshold,
                with_vectors=fetch_vectors,
            )
            return RankedList(name="dense", weight=weights.get("dense", 1.0), items=items)

        async def sparse(v: QueryVariant) -> RankedList:
            assert pipeline.sparse is not None
            vector = await pipeline.sparse.embed_query(v.text)
            items = await self.reader.search_sparse(pipeline.name, vector, limit=cfg.candidate_k, filter_=filter_)
            return RankedList(name="sparse", weight=weights.get("sparse", 1.0), items=items)

        tasks = []
        for v in variants:
            if use_dense and v.embed_dense:
                tasks.append(dense(v))
            if use_sparse and v.embed_sparse:
                tasks.append(sparse(v))
        return list(await asyncio.gather(*tasks))

    async def _native_hybrid(
        self,
        pipeline: CollectionPipeline,
        variant: QueryVariant,
        cfg: RetrievalConfig,
        filter_: FilterSpec | None,
        points_by_id: dict[str, ScoredPoint],
    ) -> list[FusedItem]:
        assert pipeline.dense is not None
        assert pipeline.sparse is not None
        dense_vec, sparse_vec = await asyncio.gather(
            pipeline.dense.embed_query(variant.text), pipeline.sparse.embed_query(variant.text)
        )
        try:
            points = await self.reader.search_hybrid(
                pipeline.name,
                dense_vec,
                sparse_vec,
                limit=cfg.candidate_k,
                candidate_k=cfg.candidate_k,
                fusion=cfg.fusion,
                filter_=filter_,
            )
        except CapabilityNotSupportedError as exc:
            raise ValidationFailedError(exc.detail) from exc
        for p in points:
            points_by_id[p.id] = p
        return [FusedItem(id=p.id, score=p.score, contributions={}) for p in points]

    def _fuse(self, lists: list[RankedList], cfg: RetrievalConfig) -> list[FusedItem]:
        non_empty = [lst for lst in lists if lst.items]
        if len(non_empty) == 1 and len(lists) == 1:
            lst = non_empty[0]
            return [
                FusedItem(id=p.id, score=p.score, contributions={lst.name: {"rank": float(r), "score": p.score}})
                for r, p in enumerate(lst.items, start=1)
            ]
        if not non_empty:
            return []
        return self.factory.fusion(cfg.fusion.method).fuse(non_empty, config=cfg.fusion, limit=cfg.candidate_k)

    @staticmethod
    def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[str] = set()
        out = []
        for chunk in chunks:
            key = sha256_text(" ".join(chunk.text.lower().split()))
            if key not in seen:
                seen.add(key)
                out.append(chunk)
        return out

    @staticmethod
    def _group(chunks: list[RetrievedChunk], field: str, size: int) -> list[RetrievedChunk]:
        """Keep the best ``size`` chunks per value of ``field`` (chunks arrive best-first). Chunks without the
        field are never grouped together."""
        kept: Counter[str] = Counter()
        out = []
        for chunk in chunks:
            if field == PayloadKeys.SOURCE_ID:
                value: Any = chunk.source_id
            else:
                value = get_path(chunk.metadata, field.removeprefix(f"{PayloadKeys.METADATA}."))
            if value is None:
                out.append(chunk)
                continue
            key = value if isinstance(value, str) else canonical_json(value)
            if kept[key] < size:
                kept[key] += 1
                out.append(chunk)
        return out

    async def _expand(self, collection: str, chunks: list[RetrievedChunk], window: int) -> None:
        """Attach neighbouring chunks as ``context`` (sentence-window / small-to-big retrieval)."""
        by_source: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            by_source[chunk.source_id].append(chunk)

        async def one_source(source_id: str, members: list[RetrievedChunk]) -> None:
            lo = max(0, min(c.chunk_index for c in members) - window)
            hi = max(c.chunk_index for c in members) + window
            filter_ = FilterSpec(
                must=[
                    Condition(field=PayloadKeys.SOURCE_ID, op=FilterOp.EQ, value=source_id),
                    Condition(field=PayloadKeys.CHUNK_INDEX, op=FilterOp.GTE, value=lo),
                    Condition(field=PayloadKeys.CHUNK_INDEX, op=FilterOp.LTE, value=hi),
                ]
            )
            neighbours, _ = await self.reader.scroll(collection, filter_=filter_, limit=hi - lo + 1)
            ordered = sorted(neighbours, key=lambda p: int(p.payload.get(PayloadKeys.CHUNK_INDEX, 0)))
            for chunk in members:
                window_points = [
                    p
                    for p in ordered
                    if abs(int(p.payload.get(PayloadKeys.CHUNK_INDEX, 0)) - chunk.chunk_index) <= window
                ]
                chunk.context = _stitch(window_points)

        await asyncio.gather(*(one_source(s, m) for s, m in by_source.items()))


def _stitch(points: list[ScoredPoint]) -> str:
    """Join consecutive chunks, trimming the overlap when character offsets are known."""
    parts: list[str] = []
    prev_end: int | None = None
    prev_section: Any = None
    for p in points:
        text = str(p.payload.get(PayloadKeys.TEXT, ""))
        meta = p.payload.get(PayloadKeys.METADATA) or {}
        start, end = meta.get("start_char"), meta.get("end_char")
        section = (meta.get("page"), meta.get("section"), meta.get("row"))
        if (
            prev_end is not None
            and isinstance(start, int)
            and section == prev_section
            and start < prev_end
            and prev_end - start < len(text)
        ):
            text = text[prev_end - start :].lstrip()
        parts.append(text)
        prev_end = end if isinstance(end, int) else None
        prev_section = section
    return "\n".join(t for t in parts if t)
