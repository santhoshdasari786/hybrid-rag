"""Domain entities and value objects (framework-free dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from rag_service.core.config import DistanceMetric


class PayloadKeys:
    """Field names stored on every vector point. User metadata lives under ``metadata``."""

    TEXT = "text"
    SOURCE_ID = "source_id"
    CHUNK_INDEX = "chunk_index"
    CHUNK_HASH = "chunk_hash"
    TITLE = "title"
    MIME_TYPE = "mime_type"
    METADATA = "metadata"

    SYSTEM = frozenset({SOURCE_ID, CHUNK_INDEX, CHUNK_HASH, TITLE, MIME_TYPE})


class SourceStatus(StrEnum):
    ACTIVE = "active"
    PROCESSING = "processing"
    FAILED = "failed"


class IngestAction(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    METADATA_UPDATED = "metadata_updated"
    DELETED = "deleted"
    NOT_FOUND = "not_found"
    FAILED = "failed"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.SUCCEEDED, JobStatus.PARTIALLY_SUCCEEDED, JobStatus.FAILED}


class JobType(StrEnum):
    INGEST = "ingest"
    DELETE = "delete"
    REINDEX = "reindex"


# ============================================================================ vectors


@dataclass(slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.indices, self.values, strict=True))


# ============================================================================ ingestion


@dataclass(slots=True)
class Section:
    """A structural unit of a parsed document (page, markdown section, CSV row…)."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    sections: list[Section]
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text)


@dataclass(slots=True)
class ChunkDraft:
    """Chunker output before identity / enrichment is assigned."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    source_id: str
    index: int
    text: str
    embed_text: str
    hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    """System-generated per-chunk metadata (page, section, offsets) plus client chunk metadata."""


@dataclass(slots=True)
class VectorRecord:
    id: str
    payload: dict[str, Any]
    dense: list[float] | None = None
    sparse: SparseVector | None = None


@dataclass(slots=True)
class PayloadPatch:
    """Partial payload update for one point (keys are replaced, others kept)."""

    id: str
    payload: dict[str, Any]


@dataclass(slots=True)
class CollectionSpec:
    """Physical index definition derived from a collection's embedding configuration."""

    name: str
    dense_dim: int | None
    distance: DistanceMetric | None
    sparse: bool
    sparse_idf: bool = False
    payload_indexes: list[tuple[str, str]] = field(default_factory=list)
    tenant_fields: list[str] = field(default_factory=list)


# ============================================================================ retrieval


@dataclass(slots=True)
class ScoredPoint:
    id: str
    score: float
    payload: dict[str, Any]
    dense: list[float] | None = None


@dataclass(slots=True)
class RankedList:
    """One retriever's ranked output, the unit consumed by fusion."""

    name: str
    weight: float
    items: list[ScoredPoint]


@dataclass(slots=True)
class FusedItem:
    id: str
    score: float
    contributions: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per-list rank/score, e.g. ``{"dense": {"rank": 1, "score": 0.82}}``."""


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    source_id: str
    text: str
    score: float
    chunk_index: int = 0
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    context: str | None = None
    dense: list[float] | None = field(default=None, repr=False)

    @classmethod
    def from_point(cls, point: ScoredPoint, score: float | None = None) -> RetrievedChunk:
        p = point.payload
        return cls(
            chunk_id=point.id,
            source_id=str(p.get(PayloadKeys.SOURCE_ID, "")),
            text=str(p.get(PayloadKeys.TEXT, "")),
            score=point.score if score is None else score,
            chunk_index=int(p.get(PayloadKeys.CHUNK_INDEX, 0) or 0),
            title=p.get(PayloadKeys.TITLE),
            metadata=dict(p.get(PayloadKeys.METADATA) or {}),
            dense=point.dense,
        )


@dataclass(slots=True)
class QueryVariant:
    text: str
    kind: str = "original"  # original | multi_query | hyde | rewrite
    embed_dense: bool = True
    embed_sparse: bool = True


# ============================================================================ registry records


@dataclass(slots=True)
class CollectionRecord:
    name: str
    description: str = ""
    pipeline_overrides: dict[str, Any] = field(default_factory=dict)
    embedding_fingerprint: str = ""
    managed: bool = False
    payload_indexes: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class ChunkRef:
    chunk_id: str
    chunk_hash: str
    chunk_index: int
    chunk_metadata: dict[str, Any] = field(default_factory=dict)
    metadata_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceRecord:
    collection: str
    source_id: str
    content_hash: str
    metadata_hash: str
    pipeline_fingerprint: str
    version: int = 1
    status: SourceStatus = SourceStatus.ACTIVE
    title: str | None = None
    mime_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    content: str | None = None
    chunking_override: dict[str, Any] | None = None
    prechunked: bool = False
    error: str | None = None
    last_job_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class JobRecord:
    id: str
    collection: str
    type: JobType
    status: JobStatus = JobStatus.QUEUED
    total: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    requested_by: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


# ============================================================================ generation


@dataclass(slots=True)
class ChatMessage:
    role: str  # system | user | assistant
    content: str


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class LLMRequest:
    messages: list[ChatMessage]
    system: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    cache_system: bool = False
    """Hint that the system prompt repeats across calls (enables provider prompt caching)."""


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    finish_reason: str = "stop"
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(slots=True)
class LLMStreamEvent:
    """Either a text ``delta`` or the terminal event (``done=True``) carrying usage."""

    delta: str = ""
    done: bool = False
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    model: str | None = None
