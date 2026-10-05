"""FastAPI dependencies: container access, authentication, scopes and collection access."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Path, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from rag_service.api.security import ApiKeyAuthenticator, Principal, TokenBucketLimiter
from rag_service.container import Container

# Declared for the OpenAPI "Authorize" dialog; the effective header name comes from configuration.
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False, scheme_name="ApiKey")
bearer_scheme = HTTPBearer(auto_error=False, scheme_name="Bearer")

COLLECTION_PATH = Path(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$", description="Collection name", examples=["products"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


async def get_principal(
    request: Request,
    _api_key: Annotated[str | None, Security(api_key_scheme)] = None,
    bearer: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)] = None,
) -> Principal:
    authenticator: ApiKeyAuthenticator = request.app.state.authenticator
    header_name = request.app.state.settings.security.api_key_header
    raw = request.headers.get(header_name) or (bearer.credentials if bearer else None)
    principal = authenticator.authenticate(raw)
    limiter: TokenBucketLimiter | None = request.app.state.rate_limiter
    if limiter is not None:
        client = request.client.host if request.client else "unknown"
        limiter.check(principal.name if not principal.anonymous else f"ip:{client}")
    request.state.principal = principal.name
    return principal


def require(scope: str) -> Callable[..., Awaitable[Principal]]:
    async def dependency(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        principal.require_scope(scope)
        return principal

    dependency.__name__ = f"require_{scope}"
    return dependency


def require_collection(scope: str) -> Callable[..., Awaitable[Principal]]:
    """Scope check plus collection allow-list check for routes with a ``{collection}`` path param."""

    async def dependency(
        principal: Annotated[Principal, Depends(get_principal)],
        collection: Annotated[str, COLLECTION_PATH],
    ) -> Principal:
        principal.require_scope(scope)
        principal.require_collection(collection)
        return principal

    dependency.__name__ = f"require_{scope}_collection"
    return dependency


ContainerDep = Annotated[Container, Depends(get_container)]
QueryPrincipal = Annotated[Principal, Depends(require_collection("query"))]
IngestPrincipal = Annotated[Principal, Depends(require_collection("ingest"))]
AdminPrincipal = Annotated[Principal, Depends(require("admin"))]
AdminCollectionPrincipal = Annotated[Principal, Depends(require_collection("admin"))]
AnyQueryPrincipal = Annotated[Principal, Depends(require("query"))]
