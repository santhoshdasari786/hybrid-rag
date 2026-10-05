"""Kubernetes-style probes and the Prometheus scrape endpoint (unauthenticated, no envelope)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rag_service.api.deps import ContainerDep
from rag_service.api.metrics import JOBS_ACTIVE

router = APIRouter(tags=["Health"])


@router.get("/health/live", summary="Liveness probe")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe", responses={503: {"description": "Not ready"}})
async def ready(container: ContainerDep) -> JSONResponse:
    checks = await container.health()
    ok = container.ready and all(checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "not_ready", "started": container.ready, "checks": checks},
    )


def metrics_endpoint(request: Request) -> Response:
    JOBS_ACTIVE.set(request.app.state.container.runner.active)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
