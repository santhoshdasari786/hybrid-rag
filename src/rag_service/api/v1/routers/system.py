"""Service introspection (admin scope)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from rag_service.api.deps import AdminPrincipal, ContainerDep
from rag_service.api.responses import envelope
from rag_service.api.schemas.common import Envelope
from rag_service.core.registry import ALL_REGISTRIES

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/info", response_model=Envelope[dict[str, Any]], summary="Service, provider and component information")
async def info(request: Request, container: ContainerDep, _: AdminPrincipal) -> Envelope[dict[str, Any]]:
    s = container.settings
    caps = container.store.capabilities
    return envelope(
        request,
        {
            "name": s.app.name,
            "version": s.app.version,
            "environment": s.app.environment.value,
            "vector_store": {
                "provider": s.vector_store.provider,
                "capabilities": {
                    "dense": caps.dense,
                    "sparse": caps.sparse,
                    "native_hybrid": caps.native_hybrid,
                    "sparse_idf": caps.sparse_idf,
                },
            },
            "llm": {"provider": s.llm.provider, "model": container.factory.llm().model},
            "utility_llm": {"provider": s.effective_utility_llm().provider},
            "components": {name: registry.describe() for name, registry in ALL_REGISTRIES.items()},
            "jobs_active": container.runner.active,
        },
    )


@router.get("/config", response_model=Envelope[dict[str, Any]], summary="Effective configuration (secrets masked)")
async def config(request: Request, container: ContainerDep, _: AdminPrincipal) -> Envelope[dict[str, Any]]:
    return envelope(request, container.settings.redacted())
