"""Collection management (admin scope for writes)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request, Response, status

from rag_service.api.deps import (
    AdminCollectionPrincipal,
    AdminPrincipal,
    AnyQueryPrincipal,
    ContainerDep,
    QueryPrincipal,
)
from rag_service.api.responses import envelope
from rag_service.api.schemas.common import Envelope
from rag_service.api.schemas.v1 import (
    CollectionCreate,
    CollectionOut,
    CollectionUpdate,
    JobOut,
    ReindexRequest,
)
from rag_service.api.v1.mappers import collection_out, job_out

router = APIRouter(prefix="/collections", tags=["Collections"])


@router.get("", response_model=Envelope[list[CollectionOut]], summary="List collections")
async def list_collections(
    request: Request, container: ContainerDep, principal: AnyQueryPrincipal
) -> Envelope[list[CollectionOut]]:
    infos = [i for i in await container.collections.list_all() if principal.can_access(i.record.name)]
    return envelope(request, [collection_out(i) for i in infos])


@router.post(
    "",
    response_model=Envelope[CollectionOut],
    status_code=status.HTTP_201_CREATED,
    summary="Create a collection",
    description="Creates the vector index and persists the collection with its pipeline overrides.",
)
async def create_collection(
    request: Request,
    response: Response,
    body: CollectionCreate,
    container: ContainerDep,
    principal: AdminPrincipal,
) -> Envelope[CollectionOut]:
    principal.require_collection(body.name)
    info = await container.collections.create(
        body.name, description=body.description, pipeline=body.pipeline, payload_indexes=body.payload_indexes
    )
    response.headers["Location"] = f"{container.settings.app.api_prefix}/collections/{body.name}"
    return envelope(request, collection_out(info))


@router.get("/{collection}", response_model=Envelope[CollectionOut], summary="Get a collection with statistics")
async def get_collection(
    request: Request, collection: str, container: ContainerDep, _: QueryPrincipal
) -> Envelope[CollectionOut]:
    return envelope(request, collection_out(await container.collections.get(collection)))


@router.patch(
    "/{collection}",
    response_model=Envelope[CollectionOut],
    summary="Update description or pipeline overrides",
)
async def update_collection(
    request: Request,
    collection: str,
    body: CollectionUpdate,
    container: ContainerDep,
    _: AdminCollectionPrincipal,
) -> Envelope[CollectionOut]:
    info = await container.collections.update(collection, description=body.description, pipeline=body.pipeline)
    return envelope(request, collection_out(info))


@router.delete("/{collection}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a collection and all its data")
async def delete_collection(collection: str, container: ContainerDep, _: AdminCollectionPrincipal) -> Response:
    await container.collections.delete(collection)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{collection}/reindex",
    response_model=Envelope[JobOut],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-index sources with the current configuration",
    description=(
        "Re-processes sources from stored content. Only chunks whose text changed are re-embedded unless "
        "`force` is set. `recreate=true` drops and rebuilds the vector index (required after changing the "
        "embedding model)."
    ),
)
async def reindex_collection(
    request: Request,
    response: Response,
    collection: str,
    container: ContainerDep,
    principal: AdminCollectionPrincipal,
    recreate: Annotated[bool, Query(description="Drop and rebuild the vector index first.")] = False,
    body: ReindexRequest | None = None,
) -> Envelope[JobOut]:
    body = body or ReindexRequest()
    job = await container.jobs.reindex(
        collection,
        source_ids=body.source_ids,
        force=body.force,
        recreate=recreate,
        requested_by=principal.name,
    )
    response.headers["Location"] = f"{container.settings.app.api_prefix}/jobs/{job.id}"
    return envelope(request, job_out(job, container.settings.app.api_prefix, include_results=False))
