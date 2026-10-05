"""Ports — the interfaces the application layer depends on (Dependency Inversion).

Adapters in ``rag_service.adapters`` implement these Protocols. Interfaces are deliberately narrow
(Interface Segregation): the ingestion service only sees :class:`VectorWriter`, the retrieval
service only :class:`VectorReader`, the collection service only :class:`CollectionAdmin`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from rag_service.core.config import DistanceMetric, FusionConfig
from rag_service.domain.filters import FilterSpec
from rag_service.domain.models import (
    ChunkDraft,
    ChunkRef,
    CollectionRecord,
    CollectionSpec,
    FusedItem,
    JobRecord,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    ParsedDocument,
    PayloadPatch,
    QueryVariant,
    RankedList,
    RetrievedChunk,
    ScoredPoint,
    SourceRecord,
    SparseVector,
    VectorRecord,
)

# =============================================================================== ingestion


@runtime_checkable
class DocumentParser(Protocol):
    """Turns raw bytes of one media type into structured text sections."""

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument: ...


@runtime_checkable
class LengthFunction(Protocol):
    """Measures text length in the configured unit (chars, words, tokens)."""

    name: str

    def __call__(self, text: str) -> int: ...


@runtime_checkable
class Chunker(Protocol):
    name: str

    async def split(self, document: ParsedDocument) -> list[ChunkDraft]: ...


# =============================================================================== embedding


@runtime_checkable
class DenseEmbedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def distance(self) -> DistanceMetric: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class SparseEmbedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def requires_idf(self) -> bool:
        """True when document weights are term frequencies and IDF must be applied at query time."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    async def embed_query(self, text: str) -> SparseVector: ...


# =============================================================================== vector store


@dataclass(frozen=True, slots=True)
class StoreCapabilities:
    dense: bool = True
    sparse: bool = True
    native_hybrid: bool = False
    sparse_idf: bool = False


@runtime_checkable
class CollectionAdmin(Protocol):
    capabilities: StoreCapabilities

    async def ensure_collection(self, spec: CollectionSpec) -> None: ...

    async def drop_collection(self, name: str) -> None: ...

    async def collection_exists(self, name: str) -> bool: ...

    async def count(self, name: str, filter_: FilterSpec | None = None) -> int: ...


@runtime_checkable
class VectorWriter(Protocol):
    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None: ...

    async def update_payloads(self, collection: str, patches: Sequence[PayloadPatch]) -> None: ...

    async def delete_ids(self, collection: str, ids: Sequence[str]) -> None: ...

    async def delete_by_filter(self, collection: str, filter_: FilterSpec) -> None: ...


@runtime_checkable
class VectorReader(Protocol):
    capabilities: StoreCapabilities

    async def search_dense(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int,
        filter_: FilterSpec | None = None,
        score_threshold: float | None = None,
        with_vectors: bool = False,
    ) -> list[ScoredPoint]: ...

    async def search_sparse(
        self,
        collection: str,
        vector: SparseVector,
        *,
        limit: int,
        filter_: FilterSpec | None = None,
    ) -> list[ScoredPoint]: ...

    async def search_hybrid(
        self,
        collection: str,
        dense: Sequence[float],
        sparse: SparseVector,
        *,
        limit: int,
        candidate_k: int,
        fusion: FusionConfig,
        filter_: FilterSpec | None = None,
    ) -> list[ScoredPoint]:
        """Store-native fusion. Raise ``CapabilityNotSupportedError`` when unsupported."""
        ...

    async def fetch(self, collection: str, ids: Sequence[str], *, with_vectors: bool = False) -> list[ScoredPoint]: ...

    async def scroll(
        self,
        collection: str,
        *,
        filter_: FilterSpec | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[ScoredPoint], str | None]: ...


@runtime_checkable
class VectorStore(CollectionAdmin, VectorWriter, VectorReader, Protocol):
    """Full store = admin + writer + reader. Services should depend on the narrow ports."""

    name: str

    async def health(self) -> bool: ...

    async def close(self) -> None: ...


# =============================================================================== retrieval


@runtime_checkable
class FusionStrategy(Protocol):
    name: str

    def fuse(self, lists: Sequence[RankedList], *, config: FusionConfig, limit: int) -> list[FusedItem]: ...


@runtime_checkable
class Reranker(Protocol):
    model_id: str

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return chunks sorted by relevance with ``scores['rerank']`` populated."""
        ...


@runtime_checkable
class QueryTransformer(Protocol):
    name: str

    async def transform(self, query: str) -> list[QueryVariant]: ...


# =============================================================================== generation


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]: ...

    async def close(self) -> None: ...


# =============================================================================== persistence


@runtime_checkable
class CollectionRepository(Protocol):
    async def get(self, name: str) -> CollectionRecord | None: ...

    async def list_all(self) -> list[CollectionRecord]: ...

    async def upsert(self, record: CollectionRecord) -> CollectionRecord: ...

    async def delete(self, name: str) -> bool: ...


@runtime_checkable
class SourceRepository(Protocol):
    async def get(self, collection: str, source_id: str, *, with_content: bool = False) -> SourceRecord | None: ...

    async def list_page(
        self, collection: str, *, limit: int, offset: int, status: str | None = None
    ) -> tuple[list[SourceRecord], int]: ...

    async def iter_ids(self, collection: str) -> list[str]: ...

    async def chunks(self, collection: str, source_id: str) -> list[ChunkRef]: ...

    async def save(
        self,
        record: SourceRecord,
        chunks: Sequence[ChunkRef] | None,
        *,
        expected_version: int | None,
        update_content: bool = True,
    ) -> SourceRecord:
        """Insert/update a source (and replace its chunk refs when given) atomically.

        ``expected_version`` enables optimistic concurrency: ``None`` means "must not exist yet".
        Raises ``ConcurrentModificationError`` on mismatch. ``update_content=False`` keeps the stored
        content column untouched (metadata-only updates).
        """
        ...

    async def find_chunk(self, collection: str, chunk_id: str) -> tuple[str, ChunkRef] | None:
        """Return ``(source_id, chunk_ref)`` for a chunk of the source's current version."""
        ...

    async def mark(
        self,
        collection: str,
        source_id: str,
        *,
        status: str,
        error: str | None = None,
        job_id: str | None = None,
    ) -> None: ...

    async def set_chunk_overrides(
        self, collection: str, chunk_id: str, overrides: dict[str, Any]
    ) -> ChunkRef | None: ...

    async def delete(self, collection: str, source_id: str) -> bool: ...

    async def delete_collection(self, collection: str) -> int: ...

    async def stats(self, collection: str) -> dict[str, int]: ...


@runtime_checkable
class JobRepository(Protocol):
    async def create(self, job: JobRecord) -> JobRecord: ...

    async def get(self, job_id: str) -> JobRecord | None: ...

    async def update(self, job: JobRecord) -> None: ...

    async def list_page(
        self, *, collection: str | None, status: str | None, limit: int, offset: int
    ) -> tuple[list[JobRecord], int]: ...

    async def fail_interrupted(self) -> int:
        """Mark jobs left RUNNING/QUEUED by a previous process as failed. Returns count."""
        ...


@runtime_checkable
class JobRunner(Protocol):
    async def submit(self, job_id: str, work: Callable[[], Awaitable[None]]) -> None: ...

    async def shutdown(self, timeout_s: float) -> None: ...

    @property
    def active(self) -> int: ...
