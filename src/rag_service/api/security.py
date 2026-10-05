"""Authentication (API keys), authorisation (scopes + collection allow-lists) and rate limiting.

* Keys are accepted in the configured header (default ``X-API-Key``) or as ``Authorization: Bearer``
  (what OpenAI SDKs send). Only SHA-256 digests are kept in memory; lookup is by digest, so the
  comparison never touches the raw secret.
* Scopes: ``query`` (retrieve, chat, read), ``ingest`` (write documents), ``admin`` (collections,
  configuration, jobs of others). ``admin`` implies the others.
* ``collections`` restricts a key to named collections (``*`` = all) — simple tenant isolation.
* Rate limiting is a per-principal token bucket (in-process; use a gateway for global limits).

To integrate OIDC/JWT, replace :class:`ApiKeyAuthenticator` with an implementation that maps a
verified token to a :class:`Principal`.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from rag_service.core.config import SecurityConfig
from rag_service.core.errors import AuthenticationError, PermissionDeniedError, RateLimitedError

SCOPES = ("query", "ingest", "admin")


@dataclass(frozen=True, slots=True)
class Principal:
    name: str
    scopes: frozenset[str]
    collections: frozenset[str] = field(default_factory=lambda: frozenset({"*"}))
    anonymous: bool = False

    def has_scope(self, scope: str) -> bool:
        return "admin" in self.scopes or scope in self.scopes

    def require_scope(self, scope: str) -> None:
        if not self.has_scope(scope):
            raise PermissionDeniedError(f"API key '{self.name}' lacks the '{scope}' scope")

    def can_access(self, collection: str) -> bool:
        return "*" in self.collections or collection in self.collections

    def require_collection(self, collection: str) -> None:
        if not self.can_access(collection):
            raise PermissionDeniedError(f"API key '{self.name}' may not access collection '{collection}'")


ANONYMOUS = Principal(name="anonymous", scopes=frozenset(SCOPES), anonymous=True)


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ApiKeyAuthenticator:
    def __init__(self, config: SecurityConfig) -> None:
        self.enabled = config.auth_enabled
        self._by_digest: dict[str, Principal] = {}
        for key in config.api_keys:
            digest = key.key_sha256.lower() if key.key_sha256 else sha256_hex(key.key.get_secret_value())  # type: ignore[union-attr]
            self._by_digest[digest] = Principal(
                name=key.name, scopes=frozenset(key.scopes), collections=frozenset(key.collections)
            )

    def authenticate(self, raw_key: str | None) -> Principal:
        if not self.enabled:
            return ANONYMOUS
        if not raw_key:
            raise AuthenticationError("Provide an API key in the X-API-Key header or as a Bearer token")
        principal = self._by_digest.get(sha256_hex(raw_key.strip()))
        if principal is None:
            raise AuthenticationError("Invalid API key")
        return principal


class TokenBucketLimiter:
    def __init__(self, requests_per_minute: int, burst: int) -> None:
        self.rate = requests_per_minute / 60.0
        self.capacity = float(burst)
        self._buckets: dict[str, tuple[float, float]] = {}

    def check(self, key: str) -> None:
        now = time.monotonic()
        tokens, last = self._buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if tokens < 1.0:
            retry_after = max(1, int((1.0 - tokens) / self.rate) + 1)
            self._buckets[key] = (tokens, now)
            raise RateLimitedError("Rate limit exceeded", extra={"retry_after": retry_after})
        self._buckets[key] = (tokens - 1.0, now)
        if len(self._buckets) > 50_000:  # bound memory under key churn
            self._buckets.clear()
