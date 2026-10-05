"""Background job status."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, Request

from rag_service.api.deps import AnyQueryPrincipal, ContainerDep
from rag_service.api.responses import envelope, paginate
from rag_service.api.schemas.common import Envelope
from rag_service.api.schemas.v1 import JobOut
from rag_service.api.v1.mappers import job_out

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}", response_model=Envelope[JobOut], summary="Get job status and per-source results")
async def get_job(
    request: Request,
    job_id: Annotated[str, Path(max_length=64)],
    container: ContainerDep,
    principal: AnyQueryPrincipal,
) -> Envelope[JobOut]:
    job = await container.jobs.get(job_id)
    principal.require_collection(job.collection)
    return envelope(request, job_out(job, container.settings.app.api_prefix))


@router.get("", response_model=Envelope[list[JobOut]], summary="List jobs (newest first)")
async def list_jobs(
    request: Request,
    container: ContainerDep,
    principal: AnyQueryPrincipal,
    collection: Annotated[str | None, Query(max_length=64)] = None,
    status: Annotated[str | None, Query(pattern="^(queued|running|succeeded|partially_succeeded|failed)$")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Envelope[list[JobOut]]:
    if collection:
        principal.require_collection(collection)
    elif "*" not in principal.collections:
        principal.require_scope("admin")
    jobs, total = await container.jobs.list_page(collection=collection, status=status, limit=limit, offset=offset)
    prefix = container.settings.app.api_prefix
    return envelope(
        request,
        [job_out(j, prefix, include_results=False) for j in jobs],
        pagination=paginate(limit, offset, total),
    )
