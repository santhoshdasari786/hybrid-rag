"""Domain → API schema mapping (keeps routers thin and schemas decoupled from domain types)."""

from __future__ import annotations

from rag_service.api.schemas.v1 import (
    ChunkOut,
    CitationOut,
    CollectionOut,
    DocumentBody,
    JobOut,
    QueryVariantOut,
    RetrievedChunkOut,
    RetrieveResponse,
    SourceOut,
    SourceResultOut,
)
from rag_service.application.chat import Citation
from rag_service.application.collections import CollectionInfo
from rag_service.application.ingestion import ChunkInput, DocumentInput
from rag_service.application.retrieval import RetrievalResult
from rag_service.domain.models import JobRecord, RetrievedChunk, SourceRecord


def collection_out(info: CollectionInfo) -> CollectionOut:
    r = info.record
    return CollectionOut(
        name=r.name,
        description=r.description,
        managed=r.managed,
        stale=info.stale,
        embedding_fingerprint=r.embedding_fingerprint,
        pipeline=info.config,
        overrides=r.pipeline_overrides,
        payload_indexes=r.payload_indexes,
        stats=info.stats,
        created_at=r.created_at,
        updated_at=r.updated_at,
    )


def job_out(job: JobRecord, api_prefix: str, *, include_results: bool = True) -> JobOut:
    return JobOut(
        id=job.id,
        collection=job.collection,
        type=job.type.value,
        status=job.status.value,
        total=job.total,
        processed=job.processed,
        succeeded=job.succeeded,
        failed=job.failed,
        error=job.error,
        requested_by=job.requested_by,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        results=[SourceResultOut(**r) for r in job.results] if include_results else None,
        links={"self": f"{api_prefix}/jobs/{job.id}"},
    )


def source_out(record: SourceRecord) -> SourceOut:
    return SourceOut(
        source_id=record.source_id,
        title=record.title,
        mime_type=record.mime_type,
        status=record.status.value,
        version=record.version,
        chunk_count=record.chunk_count,
        metadata=record.metadata,
        content_hash=record.content_hash,
        error=record.error,
        last_job_id=record.last_job_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def chunk_out(chunk: RetrievedChunk, chunk_hash: str | None = None) -> ChunkOut:
    return ChunkOut(
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        title=chunk.title,
        metadata=chunk.metadata,
        chunk_hash=chunk_hash,
    )


def retrieved_out(rank: int, chunk: RetrievedChunk) -> RetrievedChunkOut:
    return RetrievedChunkOut(
        rank=rank,
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        chunk_index=chunk.chunk_index,
        score=round(chunk.score, 6),
        scores=chunk.scores,
        text=chunk.text,
        context=chunk.context,
        title=chunk.title,
        metadata=chunk.metadata,
    )


def retrieve_response(collection: str, result: RetrievalResult, *, debug: bool) -> RetrieveResponse:
    return RetrieveResponse(
        collection=collection,
        query=result.query,
        mode=result.mode.value,
        results=[retrieved_out(i, c) for i, c in enumerate(result.chunks, start=1)],
        variants=[QueryVariantOut(text=v.text, kind=v.kind) for v in result.variants],
        timings_ms=result.timings,
        config=result.config.model_dump(mode="json") if debug else None,
        debug=result.debug,
    )


def citation_out(c: Citation) -> CitationOut:
    return CitationOut(
        index=c.index,
        chunk_id=c.chunk_id,
        source_id=c.source_id,
        title=c.title,
        snippet=c.snippet,
        score=c.score,
        metadata=c.metadata,
    )


def document_input(source_id: str, body: DocumentBody) -> DocumentInput:
    return DocumentInput(
        source_id=source_id,
        mime_type=body.mime_type,
        content=body.content,
        content_base64=body.content_base64,
        chunks=[ChunkInput(text=c.text, metadata=c.metadata) for c in body.chunks] if body.chunks is not None else None,
        title=body.title,
        metadata=dict(body.metadata),
        chunking=body.chunking,
        filename=body.filename,
    )
