"""Small, dependency-free helpers shared by all layers."""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, TypeVar

T = TypeVar("T")

# Stable namespace so chunk IDs are identical across deployments and restarts.
CHUNK_NAMESPACE = uuid.UUID("7b1e4f2a-9c3d-4e5f-8a6b-1c2d3e4f5a6b")


def utcnow() -> datetime:
    return datetime.now(UTC)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``. Lists and scalars are replaced."""
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = value
    return result


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def hash_json(data: Any) -> str:
    return sha256_text(canonical_json(data))


def stable_hash32(token: str) -> int:
    """Process-independent 32-bit hash (Python's ``hash`` is salted per process)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")


def chunk_id(collection: str, source_id: str, chunk_hash: str, occurrence: int) -> str:
    """Deterministic chunk ID: identical content in the same source always maps to the same point."""
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{collection}/{source_id}/{chunk_hash}/{occurrence}"))


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def normalize_text(text: str, *, unicode_nfkc: bool = True, collapse_whitespace: bool = True) -> str:
    """Normalize extracted text without destroying paragraph structure."""
    if unicode_nfkc:
        text = unicodedata.normalize("NFKC", text)
    # Drop control characters except newline / tab.
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    if collapse_whitespace:
        lines = [" ".join(line.split()) for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        collapsed: list[str] = []
        blank = 0
        for line in lines:
            if line:
                blank = 0
                collapsed.append(line)
            else:
                blank += 1
                if blank == 1:
                    collapsed.append("")
        text = "\n".join(collapsed)
    return text.strip()


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Stopwatch:
    """Collects named stage timings in milliseconds."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(self.timings.get(name, 0.0) + (time.perf_counter() - start) * 1000, 3)


def get_path(data: Mapping[str, Any], dotted: str) -> Any:
    """Read ``a.b.c`` from nested mappings; returns ``None`` when absent."""
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current
