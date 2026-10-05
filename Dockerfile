# syntax=docker/dockerfile:1.7
# ----------------------------------------------------------------------------- builder
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Comma-separated optional extras baked into the image.
ARG EXTRAS="qdrant,fastembed,pgvector,anthropic,openai,gemini,pdf,migrations"
WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $(for e in ${EXTRAS//,/ }; do printf -- '--extra %s ' "$e"; done)
COPY src ./src
# --no-editable: install the package into the venv itself (the runtime stage has no /app/src).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable $(for e in ${EXTRAS//,/ }; do printf -- '--extra %s ' "$e"; done)

# ----------------------------------------------------------------------------- runtime
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="advanced-rag-service" \
      org.opencontainers.image.description="Configurable Advanced RAG micro-service" \
      org.opencontainers.image.source="https://example.invalid/advanced-rag-service"

RUN groupadd --system app && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app config ./config
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app alembic.ini ./alembic.ini
RUN mkdir -p /app/data/models && chown -R app:app /app/data

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_CONFIG_FILE=config/rag.yaml \
    LOG_FORMAT=json \
    FASTEMBED_CACHE=/app/data/models \
    PORT=8080

USER app
EXPOSE 8080
VOLUME ["/app/data"]

HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health/live', timeout=2).status == 200 else 1)"

CMD ["rag-service"]
