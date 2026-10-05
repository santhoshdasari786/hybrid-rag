"""Helpers that build the standard success envelope."""

from __future__ import annotations

import time
from typing import Any, TypeVar

from fastapi import Request

from rag_service.api.schemas.common import Envelope, Meta, Pagination
from rag_service.core.utils import utcnow

T = TypeVar("T")


def request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "") or "")


def meta(request: Request, *, pagination: Pagination | None = None) -> Meta:
    started = getattr(request.state, "started_at", None)
    duration = round((time.perf_counter() - started) * 1000, 3) if started else None
    return Meta(request_id=request_id(request), timestamp=utcnow(), duration_ms=duration, pagination=pagination)


def envelope(request: Request, data: T, *, pagination: Pagination | None = None) -> Envelope[T]:
    return Envelope[Any](data=data, meta=meta(request, pagination=pagination))


def paginate(limit: int, offset: int, total: int) -> Pagination:
    return Pagination(limit=limit, offset=offset, total=total, has_more=offset + limit < total)
