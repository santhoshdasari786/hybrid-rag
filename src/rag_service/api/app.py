"""FastAPI application factory.

Run locally:   ``uv run rag-service``  (or ``uvicorn rag_service.api.app:app_factory --factory``)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rag_service.api import health, openai_compat
from rag_service.api.errors import install_exception_handlers
from rag_service.api.middleware import (
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from rag_service.api.schemas.common import PROBLEM_RESPONSES
from rag_service.api.security import ApiKeyAuthenticator, TokenBucketLimiter
from rag_service.api.v1.routers import collections, documents, jobs, search, system
from rag_service.container import Container
from rag_service.core.config import Settings
from rag_service.core.logging import configure_logging, get_logger

log = get_logger(__name__)

DESCRIPTION = """
Configurable **Advanced RAG** micro-service: on-demand ingestion with incremental re-sync,
pluggable chunking / embedding / vector stores, hybrid retrieval (dense + sparse + fusion),
re-ranking, and grounded chat with citations.

* Success responses use the envelope `{"data": ..., "meta": {"request_id", "timestamp", ...}}`.
* Errors use RFC 9457 `application/problem+json` with a stable `code`.
* Authenticate with `X-API-Key: <key>` or `Authorization: Bearer <key>` (when auth is enabled).
* The OpenAI-compatible endpoints live under `/v1` — use `rag:<collection>` as the model.
"""

TAGS = [
    {
        "name": "Collections",
        "description": "Create and configure collections (isolated indexes + pipelines).",
    },
    {"name": "Documents", "description": "On-demand ingestion, re-sync by source ID, metadata, deletion."},
    {"name": "Retrieval & Chat", "description": "Raw retrieval results, grounded chat, evaluation."},
    {"name": "Jobs", "description": "Asynchronous ingestion / reindex jobs."},
    {"name": "System", "description": "Introspection (admin)."},
    {"name": "OpenAI-compatible", "description": "Drop-in `/v1/chat/completions` backed by a collection."},
    {"name": "Health", "description": "Liveness / readiness probes."},
]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    configure_logging(settings.logging)
    container = Container.build(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await container.start()
        try:
            yield
        finally:
            await container.stop()

    docs = settings.app.docs_enabled
    app = FastAPI(
        title="Advanced RAG Service",
        version=settings.app.version,
        description=DESCRIPTION,
        lifespan=lifespan,
        root_path=settings.app.root_path,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
        openapi_tags=TAGS,
        responses=PROBLEM_RESPONSES,
    )
    app.state.settings = settings
    app.state.container = container
    app.state.authenticator = ApiKeyAuthenticator(settings.security)
    rl = settings.security.rate_limit
    app.state.rate_limiter = TokenBucketLimiter(rl.requests_per_minute, rl.burst) if rl.enabled else None

    # Middleware: the last one added is the outermost.
    app.add_middleware(GZipMiddleware, minimum_size=settings.server.gzip_min_bytes)
    if settings.security.trusted_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.security.trusted_hosts)
    if settings.security.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.security.cors_allow_origins,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "Location"],
        )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.security.max_request_body_mb * 1024 * 1024)
    if settings.security.security_headers:
        app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware, access_log=settings.logging.access_log)

    install_exception_handlers(app)

    app.include_router(health.router)
    if settings.observability.metrics_enabled:
        app.add_api_route(settings.observability.metrics_path, health.metrics_endpoint, include_in_schema=False)
    prefix = settings.app.api_prefix
    for router in (collections.router, documents.router, search.router, jobs.router, system.router):
        app.include_router(router, prefix=prefix)
    if settings.app.openai_compat_enabled:
        app.include_router(openai_compat.router)

    if settings.observability.tracing_enabled:
        _instrument(app, settings)
    return app


def _instrument(app: FastAPI, settings: Settings) -> None:
    try:
        from opentelemetry import trace  # noqa: PLC0415 - optional dependency
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # noqa: PLC0415
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
    except ImportError:
        log.warning("tracing.disabled", reason="install the 'otel' extra")
        return
    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.app.name, "service.version": settings.app.version})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.observability.otlp_endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health/live,health/ready,metrics")
    log.info("tracing.enabled", endpoint=settings.observability.otlp_endpoint)


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory`` / gunicorn workers."""
    return create_app()
