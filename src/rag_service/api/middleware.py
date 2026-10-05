"""Pure-ASGI middleware (streaming-safe — no response buffering).

* :class:`RequestContextMiddleware` — request ID propagation (``X-Request-ID``), structured access
  log, Prometheus metrics, ``Server-Timing`` style duration header.
* :class:`SecurityHeadersMiddleware` — conservative security headers.
* :class:`BodySizeLimitMiddleware` — rejects oversized bodies early (Content-Length) and while
  streaming (chunked uploads).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_service.api.metrics import HTTP_IN_FLIGHT, HTTP_LATENCY, HTTP_REQUESTS
from rag_service.core.logging import bind_context, clear_context, get_logger

log = get_logger("rag_service.access")
_VALID_ID = re.compile(r"^[A-Za-z0-9._\-]{8,128}$")
_QUIET_PATHS = {"/health/live", "/health/ready", "/metrics"}


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return str(value.decode("latin-1"))
    return None


def _route_template(scope: Scope) -> str:
    """Full route template for metric labels, e.g. ``/api/v1/collections/{collection}/retrieve``.

    Rebuilt from the concrete path and the matched path parameters so it is correct regardless of
    how routers are nested/prefixed (bounded label cardinality, no IDs in labels).
    """
    if scope.get("route") is None:
        return "unmatched"
    params = {str(v): k for k, v in (scope.get("path_params") or {}).items()}
    return "/".join(f"{{{params[seg]}}}" if seg in params else seg for seg in str(scope.get("path", "")).split("/"))


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, access_log: bool = True) -> None:
        self.app = app
        self.access_log = access_log

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = _header(scope, b"x-request-id")
        request_id = incoming if incoming and _VALID_ID.match(incoming) else uuid.uuid4().hex
        started = time.perf_counter()
        state: dict[str, Any] = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["started_at"] = started
        clear_context()
        bind_context(request_id=request_id)
        status_holder = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        HTTP_IN_FLIGHT.inc()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            HTTP_IN_FLIGHT.dec()
            duration = time.perf_counter() - started
            template = _route_template(scope)
            method = scope.get("method", "")
            status = status_holder["status"]
            HTTP_REQUESTS.labels(method, template, str(status)).inc()
            HTTP_LATENCY.labels(method, template).observe(duration)
            if self.access_log and scope.get("path") not in _QUIET_PATHS:
                client = scope.get("client") or ("", 0)
                log.info(
                    "http.access",
                    method=method,
                    path=scope.get("path"),
                    route=template,
                    status=status,
                    duration_ms=round(duration * 1000, 2),
                    client_ip=client[0],
                    principal=state.get("principal"),
                )
            clear_context()


class SecurityHeadersMiddleware:
    HEADERS = [
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"no-referrer"),
        (b"permissions-policy", b"geolocation=(), microphone=(), camera=()"),
        (b"cross-origin-opener-policy", b"same-origin"),
    ]

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_api = str(scope.get("path", "")).startswith(("/api/", "/v1/"))

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(self.HEADERS)
                if is_api:
                    headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, scope: Scope, send: Send) -> None:
        body = json.dumps(
            {
                "type": "urn:rag-service:problem:payload-too-large",
                "title": "Payload too large",
                "status": 413,
                "detail": f"Request body exceeds {self.max_bytes} bytes",
                "instance": scope.get("path"),
                "code": "PAYLOAD_TOO_LARGE",
                "request_id": scope.get("state", {}).get("request_id"),
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/problem+json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = _header(scope, b"content-length")
        if length and length.isdigit() and int(length) > self.max_bytes:
            await self._reject(scope, send)
            return
        received = 0
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def receive_wrapper() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError
            return message

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except _BodyTooLargeError:
            if not response_started:
                await self._reject(scope, send)


class _BodyTooLargeError(Exception):
    pass
