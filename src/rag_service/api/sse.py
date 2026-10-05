"""Server-Sent Events encoding for streaming chat responses."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse
from fastapi.sse import format_sse_event
from pydantic import BaseModel

from rag_service.core.errors import RAGError
from rag_service.core.logging import get_logger

log = get_logger(__name__)

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def _json(data: Any) -> str:
    if isinstance(data, BaseModel):
        return data.model_dump_json(exclude_none=True)
    return json.dumps(data, default=str, ensure_ascii=False)


def sse_event(event: str | None, data: Any) -> bytes:
    raw = data if isinstance(data, str) else _json(data)
    return bytes(format_sse_event(data_str=raw, event=event))


def sse_response(events: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(events, media_type="text/event-stream", headers=SSE_HEADERS)


def error_event(exc: Exception, request_id: str) -> bytes:
    """Errors after the stream has started can't change the HTTP status — send them as an event."""
    if isinstance(exc, RAGError):
        body = {
            "code": exc.code,
            "title": exc.title,
            "status": exc.status,
            "detail": exc.detail,
            "request_id": request_id,
        }
    else:
        log.exception("stream.failed")
        body = {
            "code": "INTERNAL_ERROR",
            "title": "Internal error",
            "status": 500,
            "detail": "Streaming failed",
            "request_id": request_id,
        }
    return sse_event("error", body)
