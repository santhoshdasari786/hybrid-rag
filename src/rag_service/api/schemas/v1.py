"""Request / response models for the ``/api/v1`` routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from rag_service.api.schemas.common import ApiModel
from rag_service.core.config import PayloadIndexConfig, PipelineConfig, RetrievalMode
from rag_service.domain.filters import FilterSpec

SOURCE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_\-.:@=+~]{0,511}$"
"""Source IDs are URL-path safe (no '/'); use ':' or '.' for hierarchy, e.g. ``kb:fans:fv-20vq3``."""

FILTER_DESCRIPTION = (
    'Metadata filter. Structured: `{"must": [{"field": "brand", "op": "eq", "value": "Delta"}]}` '
    "(ops: eq, ne, in, nin, gt, gte, lt, lte, exists; groups: must / should / must_not, nestable). "
    'Shorthand: `{"brand": "Delta", "tags": ["quiet"], "price": {"lte": 150}}`.'
)


def _check_metadata_keys(value: dict[str, Any]) -> dict[str, Any]:
    for key in value:
        if not key or "." in key or key.startswith("$"):
            raise ValueError(f"invalid metadata key '{key}': keys must be non-empty and contain no '.' or leading '$'")
    return value


# ============================================================================ collections


class CollectionCreate(ApiModel):
    name: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$", examples=["products"])
    description: str = Field("", max_length=2_000)
    pipeline: dict[str, Any] = Field(
        default_factory=dict,
        description="Partial pipeline configuration merged over the service defaults.",
        examples=[
            {
                "chunking": {"strategy": "markdown", "chunk_size": 400, "chunk_overlap": 50},
                "retrieval": {"mode": "hybrid", "top_k": 8, "rerank": {"enabled": True}},
            }
        ],
    )
    payload_indexes: list[PayloadIndexConfig] = Field(
        default_factory=list, examples=[[{"field": "brand", "type": "keyword"}]]
    )


class CollectionUpdate(ApiModel):
    description: str | None = Field(None, max_length=2_000)
    pipeline: dict[str, Any] | None = Field(None, description="Replaces the collection's pipeline overrides.")


class CollectionOut(ApiModel):
    name: str
    description: str
    managed: bool = Field(..., description="Declared in the configuration file (read-only through the API).")
    stale: bool = Field(..., description="Embedding configuration changed since indexing; reindex with recreate=true.")
    embedding_fingerprint: str
    pipeline: PipelineConfig = Field(..., description="Effective pipeline configuration.")
    overrides: dict[str, Any]
    payload_indexes: list[dict[str, Any]]
    stats: dict[str, int] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ReindexRequest(ApiModel):
    source_ids: list[str] | None = Field(None, max_length=100_000, description="Limit to these sources (default: all).")
    force: bool = Field(False, description="Re-embed even when content and configuration are unchanged.")


# ============================================================================ documents


class ChunkIn(ApiModel):
    text: str = Field(..., min_length=1, max_length=200_000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _keys = field_validator("metadata")(_check_metadata_keys)


class DocumentBody(ApiModel):
    title: str | None = Field(None, max_length=1_000)
    content: str | None = Field(
        None, description="Raw text (text/plain, text/markdown, text/html, text/csv, application/json)."
    )
    content_base64: str | None = Field(None, description="Base64-encoded file bytes (e.g. application/pdf).")
    chunks: list[ChunkIn] | None = Field(
        None,
        max_length=20_000,
        description="Pre-chunked content; bypasses the chunker. Each chunk may carry its own metadata.",
    )
    mime_type: str = Field("text/plain", max_length=128, examples=["text/markdown"])
    filename: str | None = Field(None, max_length=512)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata stored on every chunk and usable in filters.",
        examples=[{"brand": "Panasonic", "category": "bathroom-fans", "price": 129.99}],
    )
    chunking: dict[str, Any] | None = Field(
        None,
        description="Per-document chunking override.",
        examples=[{"strategy": "sentence", "chunk_size": 256}],
    )

    _keys = field_validator("metadata")(_check_metadata_keys)

    @model_validator(mode="after")
    def _exactly_one_body(self) -> DocumentBody:
        provided = [x for x in (self.content, self.content_base64, self.chunks) if x is not None]
        if len(provided) != 1:
            raise ValueError("provide exactly one of `content`, `content_base64` or `chunks`")
        return self


class DocumentIn(DocumentBody):
    source_id: str | None = Field(
        None,
        pattern=SOURCE_ID_PATTERN,
        description="Stable processing ID. Re-sending it updates the source in place; omit to have one generated.",
        examples=["kb:fans:panasonic-fv-20vq3"],
    )


class IngestOptions(ApiModel):
    mode: Literal["sync", "async"] | None = Field(
        None, description="Default: service configuration (ingestion.default_mode)."
    )
    force: bool = Field(False, description="Re-embed every chunk even if unchanged.")
    callback_url: str | None = Field(
        None, max_length=2_048, description="Notified when the job finishes (host allow-listed)."
    )


class IngestRequest(ApiModel):
    documents: list[DocumentIn] = Field(..., min_length=1)
    options: IngestOptions = Field(default_factory=IngestOptions)


class UpsertDocumentRequest(DocumentBody):
    options: IngestOptions = Field(default_factory=IngestOptions)


class BulkDeleteRequest(ApiModel):
    source_ids: list[str] = Field(..., min_length=1, max_length=10_000)
    mode: Literal["sync", "async"] = "sync"


class MetadataPatch(ApiModel):
    metadata: dict[str, Any]
    merge: bool = Field(True, description="Deep-merge into existing metadata (false = replace).")

    _keys = field_validator("metadata")(_check_metadata_keys)


class SourceResultOut(ApiModel):
    source_id: str
    action: str = Field(..., examples=["created", "updated", "unchanged", "metadata_updated", "deleted", "failed"])
    version: int | None = None
    chunks_total: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    chunks_unchanged: int = 0
    chunks_updated: int = 0
    error: str | None = None
    error_code: str | None = None
    duration_ms: float = 0.0


class JobOut(ApiModel):
    id: str
    collection: str
    type: str
    status: str
    total: int
    processed: int
    succeeded: int
    failed: int
    error: str | None = None
    requested_by: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    results: list[SourceResultOut] | None = None
    links: dict[str, str] = Field(default_factory=dict)


class SourceOut(ApiModel):
    source_id: str
    title: str | None
    mime_type: str
    status: str
    version: int
    chunk_count: int
    metadata: dict[str, Any]
    content_hash: str
    error: str | None = None
    last_job_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChunkOut(ApiModel):
    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_hash: str | None = None


# ============================================================================ retrieval


class RetrievalOverrides(ApiModel):
    """Per-request overrides of the collection's retrieval configuration (all optional)."""

    mode: RetrievalMode | None = None
    top_k: int | None = Field(None, ge=1, le=100)
    candidate_k: int | None = Field(None, ge=1, le=1_000)
    min_score: float | None = None
    dense_score_threshold: float | None = None
    dedupe: bool | None = None
    group_by: str | None = Field(
        None, description='Collapse results sharing this field (`source_id`, `family`…); "" turns grouping off.'
    )
    group_size: int | None = Field(None, ge=1, le=20)
    fusion: dict[str, Any] | None = Field(
        None, examples=[{"method": "rrf", "k": 60, "weights": {"dense": 1.0, "sparse": 1.5}}]
    )
    query_transform: dict[str, Any] | None = Field(None, examples=[{"strategy": "multi_query", "num_queries": 3}])
    rerank: dict[str, Any] | None = Field(None, examples=[{"enabled": True, "candidates": 30}])
    mmr: dict[str, Any] | None = Field(None, examples=[{"enabled": True, "lambda_mult": 0.6}])
    context_expansion: dict[str, Any] | None = Field(None, examples=[{"enabled": True, "window": 1}])

    def as_overrides(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class RetrieveRequest(ApiModel):
    query: str = Field(..., min_length=1, max_length=8_000, examples=["revent 80 cfm"])
    top_k: int | None = Field(None, ge=1, le=100, description="Shortcut for retrieval.top_k.")
    mode: RetrievalMode | None = Field(None, description="Shortcut for retrieval.mode.")
    filter: FilterSpec | None = Field(None, description=FILTER_DESCRIPTION)
    retrieval: RetrievalOverrides | None = None
    include_debug: bool = Field(False, description="Return per-retriever ranked lists, fused list and stage timings.")

    def overrides(self) -> dict[str, Any]:
        data = self.retrieval.as_overrides() if self.retrieval else {}
        if self.top_k is not None:
            data["top_k"] = self.top_k
        if self.mode is not None:
            data["mode"] = self.mode.value
        return data


class RetrievedChunkOut(ApiModel):
    rank: int
    chunk_id: str
    source_id: str
    chunk_index: int
    score: float
    scores: dict[str, float] = Field(
        default_factory=dict, description="Score breakdown: fused, dense/sparse rank & score, rerank, mmr."
    )
    text: str
    context: str | None = Field(None, description="Text with neighbouring chunks (when context expansion is enabled).")
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryVariantOut(ApiModel):
    text: str
    kind: str


class RetrieveResponse(ApiModel):
    collection: str
    query: str
    mode: str
    results: list[RetrievedChunkOut]
    variants: list[QueryVariantOut]
    timings_ms: dict[str, float]
    config: dict[str, Any] | None = Field(None, description="Effective retrieval configuration (debug only).")
    debug: dict[str, Any] | None = None


# ============================================================================ chat


class ChatMessageIn(ApiModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., max_length=200_000)


class GenerationOverrides(ApiModel):
    system_prompt: str | None = Field(None, max_length=20_000)
    max_tokens: int | None = Field(None, ge=16, le=128_000)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_context_tokens: int | None = Field(None, ge=256, le=900_000)
    condense_history: bool | None = None
    history_turns: int | None = Field(None, ge=0, le=50)

    def as_overrides(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class ChatRequest(ApiModel):
    messages: list[ChatMessageIn] = Field(..., min_length=1, max_length=200)
    stream: bool = Field(False, description="Stream Server-Sent Events: retrieval → delta* → done.")
    conversation_id: str | None = Field(None, max_length=128, description="Echoed back for client-side correlation.")
    filter: FilterSpec | None = Field(None, description=FILTER_DESCRIPTION)
    retrieval: RetrievalOverrides | None = None
    generation: GenerationOverrides | None = None
    include_sources: bool = Field(
        True, description="Return every source placed in the prompt, not only the cited ones."
    )


class CitationOut(ApiModel):
    index: int = Field(..., description="The [n] marker used in the answer.")
    chunk_id: str
    source_id: str
    title: str | None = None
    snippet: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssistantMessage(ApiModel):
    role: Literal["assistant"] = "assistant"
    content: str


class UsageOut(ApiModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatRetrievalOut(ApiModel):
    standalone_query: str
    mode: str
    results: int
    timings_ms: dict[str, float]


class ChatResponse(ApiModel):
    id: str
    conversation_id: str | None = None
    collection: str
    model: str
    message: AssistantMessage
    citations: list[CitationOut]
    sources: list[CitationOut] | None = None
    finish_reason: str = Field(..., examples=["stop", "length", "refusal", "no_context"])
    usage: UsageOut
    retrieval: ChatRetrievalOut


# ============================================================================ evaluation


class EvalCaseIn(ApiModel):
    query: str = Field(..., min_length=1, max_length=8_000)
    relevant: list[str] | dict[str, float] = Field(
        ...,
        description="Relevant source IDs (or chunk IDs for level=chunk); a mapping gives graded relevance for nDCG.",
    )
    filter: FilterSpec | None = None


class EvaluateRequest(ApiModel):
    cases: list[EvalCaseIn] = Field(..., min_length=1, max_length=1_000)
    k: int = Field(5, ge=1, le=50)
    level: Literal["source", "chunk"] = "source"
    variants: dict[str, dict[str, Any]] | None = Field(
        None,
        description="Named retrieval overrides to compare. Default: dense, sparse, hybrid.",
        examples=[{"hybrid": {"mode": "hybrid"}, "hybrid+rerank": {"mode": "hybrid", "rerank": {"enabled": True}}}],
    )
    include_cases: bool = False


class EvalCaseOut(ApiModel):
    query: str
    retrieved: list[str]
    first_relevant_rank: int | None
    recall: float
    precision: float
    ndcg: float


class VariantReportOut(ApiModel):
    name: str
    overrides: dict[str, Any]
    metrics: dict[str, float]
    cases: list[EvalCaseOut] = Field(default_factory=list)


class EvaluateResponse(ApiModel):
    collection: str
    k: int
    level: str
    variants: list[VariantReportOut]
    best: str | None = Field(
        None, description="Variant with the highest nDCG (MRR breaks ties); null when the top variants tie."
    )
