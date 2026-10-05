"""Exception hierarchy.

Every error raised by the domain, application or adapter layers derives from :class:`RAGError`.
The API layer converts them into RFC 9457 ``application/problem+json`` responses using the
``status``, ``code`` and ``title`` class attributes, so layers below the API never deal with HTTP.
"""

from __future__ import annotations

from typing import Any, ClassVar


class RAGError(Exception):
    """Base class for all expected (handled) service errors."""

    status: ClassVar[int] = 500
    code: ClassVar[str] = "INTERNAL_ERROR"
    title: ClassVar[str] = "Internal error"

    def __init__(self, detail: str, *, extra: dict[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extra = extra or {}


# --------------------------------------------------------------------------- 4xx


class ValidationFailedError(RAGError):
    status = 422
    code = "VALIDATION_FAILED"
    title = "Validation failed"


class InvalidFilterError(ValidationFailedError):
    code = "INVALID_FILTER"
    title = "Invalid metadata filter"


class UnsupportedMediaTypeError(RAGError):
    status = 415
    code = "UNSUPPORTED_MEDIA_TYPE"
    title = "Unsupported media type"


class PayloadTooLargeError(RAGError):
    status = 413
    code = "PAYLOAD_TOO_LARGE"
    title = "Payload too large"


class NotFoundError(RAGError):
    status = 404
    code = "NOT_FOUND"
    title = "Resource not found"


class CollectionNotFoundError(NotFoundError):
    code = "COLLECTION_NOT_FOUND"
    title = "Collection not found"


class SourceNotFoundError(NotFoundError):
    code = "SOURCE_NOT_FOUND"
    title = "Source not found"


class ChunkNotFoundError(NotFoundError):
    code = "CHUNK_NOT_FOUND"
    title = "Chunk not found"


class JobNotFoundError(NotFoundError):
    code = "JOB_NOT_FOUND"
    title = "Job not found"


class ConflictError(RAGError):
    status = 409
    code = "CONFLICT"
    title = "Conflict"


class CollectionExistsError(ConflictError):
    code = "COLLECTION_EXISTS"
    title = "Collection already exists"


class ConcurrentModificationError(ConflictError):
    code = "CONCURRENT_MODIFICATION"
    title = "Concurrent modification"


class EmbeddingConfigChangedError(ConflictError):
    code = "EMBEDDING_CONFIG_CHANGED"
    title = "Embedding configuration changed"


class AuthenticationError(RAGError):
    status = 401
    code = "UNAUTHENTICATED"
    title = "Authentication required"


class PermissionDeniedError(RAGError):
    status = 403
    code = "FORBIDDEN"
    title = "Permission denied"


class RateLimitedError(RAGError):
    status = 429
    code = "RATE_LIMITED"
    title = "Too many requests"


# --------------------------------------------------------------------------- 5xx


class ConfigurationError(RAGError):
    status = 500
    code = "CONFIGURATION_ERROR"
    title = "Configuration error"


class CapabilityNotSupportedError(RAGError):
    status = 501
    code = "NOT_SUPPORTED"
    title = "Capability not supported"


class ProviderError(RAGError):
    """An upstream dependency (vector DB, embedding model, LLM) failed."""

    status = 502
    code = "PROVIDER_ERROR"
    title = "Upstream provider error"


class ProviderUnavailableError(ProviderError):
    status = 503
    code = "PROVIDER_UNAVAILABLE"
    title = "Upstream provider unavailable"


class DependencyMissingError(ConfigurationError):
    code = "DEPENDENCY_MISSING"
    title = "Optional dependency not installed"

    @classmethod
    def for_extra(cls, package: str, extra: str) -> DependencyMissingError:
        return cls(f"'{package}' is not installed. Install the extra: pip install 'advanced-rag-service[{extra}]'")
