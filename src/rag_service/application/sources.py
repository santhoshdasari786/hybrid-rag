"""Read-side queries over sources and chunks (the write side lives in ``ingestion``)."""

from __future__ import annotations

from dataclasses import dataclass

from rag_service.application.collections import CollectionService
from rag_service.core.errors import ChunkNotFoundError, SourceNotFoundError
from rag_service.domain.filters import FilterSpec
from rag_service.domain.models import RetrievedChunk, SourceRecord
from rag_service.domain.ports import SourceRepository, VectorReader


@dataclass(slots=True)
class ChunkView:
    chunk: RetrievedChunk
    chunk_hash: str | None = None


class SourceService:
    def __init__(self, collections: CollectionService, sources: SourceRepository, reader: VectorReader) -> None:
        self.collections = collections
        self.sources = sources
        self.reader = reader

    async def list_page(
        self, collection: str, *, limit: int, offset: int, status: str | None
    ) -> tuple[list[SourceRecord], int]:
        await self.collections.get(collection, with_stats=False)
        return await self.sources.list_page(collection, limit=limit, offset=offset, status=status)

    async def get(self, collection: str, source_id: str) -> SourceRecord:
        record = await self.sources.get(collection, source_id)
        if record is None:
            raise SourceNotFoundError(f"Source '{source_id}' not found in collection '{collection}'")
        return record

    async def chunks(self, collection: str, source_id: str, *, limit: int, offset: int) -> tuple[list[ChunkView], int]:
        await self.get(collection, source_id)
        refs = await self.sources.chunks(collection, source_id)
        page = refs[offset : offset + limit]
        points = {p.id: p for p in await self.reader.fetch(collection, [r.chunk_id for r in page])}
        views = []
        for ref in page:
            point = points.get(ref.chunk_id)
            if point is None:  # registry and index out of sync — surface it rather than hide it
                continue
            views.append(ChunkView(chunk=RetrievedChunk.from_point(point, score=0.0), chunk_hash=ref.chunk_hash))
        return views, len(refs)

    async def chunk(self, collection: str, chunk_id: str) -> RetrievedChunk:
        await self.collections.get(collection, with_stats=False)
        points = await self.reader.fetch(collection, [chunk_id])
        if not points:
            raise ChunkNotFoundError(f"Chunk '{chunk_id}' not found in collection '{collection}'")
        return RetrievedChunk.from_point(points[0], score=0.0)

    async def count_points(self, collection: str, source_id: str) -> int:
        count = getattr(self.reader, "count", None)
        if count is None:
            return -1
        result: int = await count(collection, FilterSpec.for_source(source_id))
        return result
