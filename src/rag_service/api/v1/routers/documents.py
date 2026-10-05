"""On-demand ingestion and source (document) lifecycle.

The ``source_id`` is the client's stable processing ID: re-sending it re-syncs the source (only
changed chunks are re-embedded), ``DELETE`` removes every chunk of it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, Request, Response, status

from rag_service.api.deps import ContainerDep, IngestPrincipal, QueryPrincipal
from rag_service.api.responses import envelope, paginate
from rag_service.api.schemas.common import Envelope
from rag_service.api.schemas.v1 import (
    SOURCE_ID_PATTERN,
    BulkDeleteRequest,
    ChunkOut,
    IngestRequest,
    JobOut,
    MetadataPatch,
    SourceOut,
    SourceResultOut,
    UpsertDocumentRequest,
)
from rag_service.api.v1.mappers import chunk_out, document_input, job_out, source_out
from rag_service.container import Container
from rag_service.core.errors import SourceNotFoundError, ValidationFailedError
from rag_service.core.utils import new_id
from rag_service.domain.models import JobRecord, JobStatus

router = APIRouter(prefix="/collections/{collection}", tags=["Documents"])

SourceIdPath = Annotated[str, Path(pattern=SOURCE_ID_PATTERN, description="Source (processing) ID")]
Limit = Annotated[int, Query(ge=1, le=500)]
Offset = Annotated[int, Query(ge=0)]


def _mode(container: Container, requested: str | None, count: int) -> str:
    mode = requested or container.settings.ingestion.default_mode
    if mode == "sync" and count > container.settings.ingestion.sync_max_documents:
        return "async"  # large batches never block the request
    return mode


def _job_response(
    request: Request, response: Response, container: Container, job: JobRecord, mode: str
) -> Envelope[JobOut]:
    prefix = container.settings.app.api_prefix
    if mode == "async" and not job.status.is_terminal:
        response.status_code = status.HTTP_202_ACCEPTED
        response.headers["Location"] = f"{prefix}/jobs/{job.id}"
    elif job.status == JobStatus.FAILED:
        response.status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    elif job.status == JobStatus.PARTIALLY_SUCCEEDED:
        response.status_code = status.HTTP_207_MULTI_STATUS
    return envelope(request, job_out(job, prefix))


@router.post(
    "/documents",
    response_model=Envelope[JobOut],
    summary="Ingest or update documents (batch)",
    description=(
        "Upserts a batch of documents. Each document's `source_id` is its stable processing ID; re-sending it "
        "synchronises the index (unchanged → skipped, metadata-only → payload patch, changed → re-chunk and "
        "re-embed only changed chunks, stale chunks deleted).\n\n"
        "`options.mode=sync` returns per-source results (200 / 207 partial / 422 all failed); `async` returns "
        "202 with a job to poll at `GET /jobs/{id}`."
    ),
    responses={202: {"description": "Accepted — poll the job"}, 207: {"description": "Some sources failed"}},
)
async def ingest_documents(
    request: Request,
    response: Response,
    collection: str,
    body: IngestRequest,
    container: ContainerDep,
    principal: IngestPrincipal,
) -> Envelope[JobOut]:
    limit = container.settings.ingestion.max_documents_per_request
    if len(body.documents) > limit:
        raise ValidationFailedError(f"At most {limit} documents per request (got {len(body.documents)})")
    docs = [document_input(d.source_id or new_id("src"), d) for d in body.documents]
    ids = [d.source_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValidationFailedError("Duplicate source_id values in one request")
    mode = _mode(container, body.options.mode, len(docs))
    job = await container.jobs.ingest(
        collection,
        docs,
        mode=mode,
        force=body.options.force,
        requested_by=principal.name,
        callback_url=body.options.callback_url,
    )
    return _job_response(request, response, container, job, mode)


@router.put(
    "/documents/{source_id}",
    response_model=Envelope[JobOut],
    summary="Create or update one document by its source ID (idempotent)",
)
async def upsert_document(
    request: Request,
    response: Response,
    collection: str,
    source_id: SourceIdPath,
    body: UpsertDocumentRequest,
    container: ContainerDep,
    principal: IngestPrincipal,
) -> Envelope[JobOut]:
    mode = _mode(container, body.options.mode or "sync", 1)
    job = await container.jobs.ingest(
        collection,
        [document_input(source_id, body)],
        mode=mode,
        force=body.options.force,
        requested_by=principal.name,
        callback_url=body.options.callback_url,
    )
    return _job_response(request, response, container, job, mode)


@router.get("/documents", response_model=Envelope[list[SourceOut]], summary="List sources")
async def list_documents(
    request: Request,
    collection: str,
    container: ContainerDep,
    _: QueryPrincipal,
    limit: Limit = 50,
    offset: Offset = 0,
    status_: Annotated[str | None, Query(alias="status", pattern="^(active|failed|processing)$")] = None,
) -> Envelope[list[SourceOut]]:
    records, total = await container.sources.list_page(collection, limit=limit, offset=offset, status=status_)
    return envelope(request, [source_out(r) for r in records], pagination=paginate(limit, offset, total))


@router.post(
    "/documents:delete",
    response_model=Envelope[JobOut],
    summary="Delete many sources",
    description="Deletes every chunk of each listed source and its registry entry.",
)
async def bulk_delete(
    request: Request,
    response: Response,
    collection: str,
    body: BulkDeleteRequest,
    container: ContainerDep,
    principal: IngestPrincipal,
) -> Envelope[JobOut]:
    job = await container.jobs.delete(collection, body.source_ids, mode=body.mode, requested_by=principal.name)
    return _job_response(request, response, container, job, body.mode)


@router.get(
    "/documents/{source_id}/chunks",
    response_model=Envelope[list[ChunkOut]],
    summary="List the chunks of a source",
)
async def list_chunks(
    request: Request,
    collection: str,
    source_id: SourceIdPath,
    container: ContainerDep,
    _: QueryPrincipal,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Envelope[list[ChunkOut]]:
    views, total = await container.sources.chunks(collection, source_id, limit=limit, offset=offset)
    return envelope(
        request, [chunk_out(v.chunk, v.chunk_hash) for v in views], pagination=paginate(limit, offset, total)
    )


@router.patch(
    "/documents/{source_id}/metadata",
    response_model=Envelope[SourceResultOut],
    summary="Update a source's metadata without re-embedding",
)
async def patch_document_metadata(
    request: Request,
    collection: str,
    source_id: SourceIdPath,
    body: MetadataPatch,
    container: ContainerDep,
    _: IngestPrincipal,
) -> Envelope[SourceResultOut]:
    result = await container.ingestion.patch_source_metadata(collection, source_id, body.metadata, merge=body.merge)
    return envelope(request, SourceResultOut(**result.as_dict()))


@router.get("/documents/{source_id}", response_model=Envelope[SourceOut], summary="Get a source")
async def get_document(
    request: Request, collection: str, source_id: SourceIdPath, container: ContainerDep, _: QueryPrincipal
) -> Envelope[SourceOut]:
    return envelope(request, source_out(await container.sources.get(collection, source_id)))


@router.delete(
    "/documents/{source_id}",
    response_model=Envelope[SourceResultOut],
    summary="Delete a source and all its chunks",
)
async def delete_document(
    request: Request, collection: str, source_id: SourceIdPath, container: ContainerDep, _: IngestPrincipal
) -> Envelope[SourceResultOut]:
    await container.collections.pipeline(collection)
    result = await container.ingestion.delete_source(collection, source_id)
    if result.action.value == "not_found":
        raise SourceNotFoundError(f"Source '{source_id}' not found in collection '{collection}'")
    return envelope(request, SourceResultOut(**result.as_dict()))


@router.get("/chunks/{chunk_id}", response_model=Envelope[ChunkOut], summary="Get a chunk")
async def get_chunk(
    request: Request,
    collection: str,
    chunk_id: Annotated[str, Path(max_length=64)],
    container: ContainerDep,
    _: QueryPrincipal,
) -> Envelope[ChunkOut]:
    return envelope(request, chunk_out(await container.sources.chunk(collection, chunk_id)))


@router.patch(
    "/chunks/{chunk_id}/metadata",
    response_model=Envelope[ChunkOut],
    summary="Attach metadata to a single chunk",
    description="Chunk-level metadata survives re-syncs while the chunk's content is unchanged.",
)
async def patch_chunk_metadata(
    request: Request,
    collection: str,
    chunk_id: Annotated[str, Path(max_length=64)],
    body: MetadataPatch,
    container: ContainerDep,
    _: IngestPrincipal,
) -> Envelope[ChunkOut]:
    await container.ingestion.patch_chunk_metadata(collection, chunk_id, body.metadata, merge=body.merge)
    return envelope(request, chunk_out(await container.sources.chunk(collection, chunk_id)))
