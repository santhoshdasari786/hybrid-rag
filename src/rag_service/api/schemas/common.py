"""Standard response envelope and RFC 9457 problem details shared by every endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Pagination(BaseModel):
    limit: int
    offset: int
    total: int
    has_more: bool


class Meta(BaseModel):
    request_id: str = Field(..., examples=["3f2c9a4e1b7d4c0e"])
    timestamp: datetime
    duration_ms: float | None = None
    api_version: str = "v1"
    pagination: Pagination | None = None


class Envelope(BaseModel, Generic[T]):
    """Every successful response: ``{"data": ..., "meta": {...}}``."""

    data: T
    meta: Meta


class FieldError(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class Problem(BaseModel):
    """RFC 9457 problem details (``application/problem+json``)."""

    type: str = Field(..., examples=["urn:rag-service:problem:collection-not-found"])
    title: str = Field(..., examples=["Collection not found"])
    status: int = Field(..., examples=[404])
    detail: str | None = Field(None, examples=["Collection 'products' does not exist"])
    instance: str | None = Field(None, examples=["/api/v1/collections/products/retrieve"])
    code: str = Field(..., examples=["COLLECTION_NOT_FOUND"])
    request_id: str | None = None
    errors: list[FieldError] | None = None
    extra: dict[str, Any] | None = None


PROBLEM_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {"model": Problem, "content": {"application/problem+json": {}}, "description": desc}
    for status, desc in {
        401: "Missing or invalid API key",
        403: "API key lacks the required scope or collection access",
        404: "Resource not found",
        409: "Conflict",
        413: "Payload too large",
        422: "Validation failed",
        429: "Rate limited",
        500: "Internal error",
        502: "Upstream provider error",
        503: "Upstream provider unavailable",
    }.items()
}
