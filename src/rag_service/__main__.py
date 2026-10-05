"""``python -m rag_service`` / ``rag-service`` console entry point."""

from __future__ import annotations

import uvicorn

from rag_service.core.config import Settings


def main() -> None:
    settings = Settings.load()
    uvicorn.run(
        "rag_service.api.app:app_factory",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_graceful_shutdown=settings.server.graceful_shutdown_s,
        log_config=None,  # structlog owns logging
    )


if __name__ == "__main__":
    main()
