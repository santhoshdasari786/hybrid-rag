"""Exception → HTTP mapping.

* Native API routes answer with RFC 9457 ``application/problem+json``.
* OpenAI-compatible routes (``/v1/...``) answer with OpenAI's ``{"error": {...}}`` shape so
  existing SDKs surface the message correctly.
* Unexpected exceptions are logged with a stack trace but never leak internals to the client.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_service.api.responses import request_id
from rag_service.api.schemas.common import FieldError, Problem
from rag_service.core.errors import RAGError, RateLimitedError
from rag_service.core.logging import get_logger

log = get_logger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"
_HTTP_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHENTICATED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    406: "NOT_ACCEPTABLE",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    429: "RATE_LIMITED",
}


def _slug(code: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-")


def problem(
    request: Request,
    *,
    status: int,
    code: str,
    title: str,
    detail: str | None,
    errors: list[FieldError] | None = None,
    extra: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    if request.url.path.startswith("/v1/"):
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "message": detail or title,
                    "type": _slug(code).replace("-", "_"),
                    "code": code,
                    "param": None,
                }
            },
            headers=headers,
        )
    body = Problem(
        type=f"urn:rag-service:problem:{_slug(code)}",
        title=title,
        status=status,
        detail=detail,
        instance=request.url.path,
        code=code,
        request_id=request_id(request) or None,
        errors=errors,
        extra=extra or None,
    )
    return JSONResponse(
        status_code=status,
        content=body.model_dump(mode="json", exclude_none=True),
        media_type=PROBLEM_MEDIA_TYPE,
        headers=headers,
    )


async def _rag_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RAGError)
    level = log.error if exc.status >= 500 else log.info
    level("request.error", code=exc.code, status=exc.status, detail=exc.detail)
    headers = {"Retry-After": str(exc.extra.get("retry_after", 60))} if isinstance(exc, RateLimitedError) else None
    extra = {k: v for k, v in exc.extra.items() if k != "retry_after"}
    return problem(
        request,
        status=exc.status,
        code=exc.code,
        title=exc.title,
        detail=exc.detail,
        extra=extra,
        headers=headers,
    )


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        FieldError(
            loc=[str(x) if not isinstance(x, int) else x for x in e.get("loc", ())],
            msg=str(e.get("msg", "")),
            type=str(e.get("type", "")),
        )
        for e in exc.errors()
    ]
    return problem(
        request,
        status=422,
        code="VALIDATION_FAILED",
        title="Validation failed",
        detail=f"{len(errors)} validation error(s) in the request",
        errors=errors,
    )


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _HTTP_CODES.get(exc.status_code, "HTTP_ERROR")
    return problem(
        request,
        status=exc.status_code,
        code=code,
        title=code.replace("_", " ").capitalize(),
        detail=str(exc.detail) if exc.detail else None,
        headers=getattr(exc, "headers", None),
    )


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("request.unhandled_exception", path=request.url.path)
    return problem(
        request,
        status=500,
        code="INTERNAL_ERROR",
        title="Internal error",
        detail="An unexpected error occurred. Quote the request_id when contacting support.",
    )


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RAGError, _rag_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled)
