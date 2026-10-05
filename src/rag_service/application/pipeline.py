"""Builds pipeline components from configuration (Factory + component cache).

Heavy components — embedding models, re-rankers, LLM clients — are cached by their configuration
so collections that share a model share one loaded instance. Cheap components (chunkers, query
transforms) are built per use because their configuration can be overridden per request.

Fingerprints decide when work must be redone:

* ``embedding_fingerprint`` — changes when vectors become incompatible (model, dimension,
  distance, prefixes). A collection whose stored fingerprint differs must be rebuilt.
* ``ingest_fingerprint``    — embedding fingerprint + chunking + enrichment. When it changes, sources
  are re-chunked (and only chunks whose text changed are re-embedded).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from rag_service.adapters.chunking.length import length_function_for
from rag_service.adapters.embedding.cache import CachedDenseEmbedder, CachedSparseEmbedder
from rag_service.adapters.enrichment import ContextualHeaderEnricher, LLMContextEnricher
from rag_service.core.config import (
    ChunkingConfig,
    DenseEmbeddingConfig,
    LLMConfig,
    PipelineConfig,
    QueryTransformConfig,
    RerankConfig,
    Settings,
    SparseEmbeddingConfig,
)
from rag_service.core.errors import ConfigurationError
from rag_service.core.logging import get_logger
from rag_service.core.registry import (
    CHUNKERS,
    DENSE_EMBEDDERS,
    FUSIONS,
    LLMS,
    QUERY_TRANSFORMS,
    RERANKERS,
    SPARSE_EMBEDDERS,
)
from rag_service.core.utils import hash_json
from rag_service.domain.models import CollectionSpec
from rag_service.domain.ports import (
    Chunker,
    DenseEmbedder,
    FusionStrategy,
    LLMClient,
    QueryTransformer,
    Reranker,
    SparseEmbedder,
    StoreCapabilities,
)

log = get_logger(__name__)


@dataclass(slots=True)
class CollectionPipeline:
    """Everything needed to ingest into / query one collection."""

    name: str
    config: PipelineConfig
    dense: DenseEmbedder | None
    sparse: SparseEmbedder | None
    embedding_fingerprint: str


# Options that change *how* a model runs, not *what* vectors it produces. They are excluded from the
# fingerprint so moving a model cache, adding threads or rotating an API key never marks data stale.
OPERATIONAL_OPTIONS = frozenset(
    {
        "cache_dir",
        "threads",
        "providers",
        "cuda",
        "device_ids",
        "lazy_load",
        "parallel",
        "api_key",
        "base_url",
        "timeout_s",
        "max_retries",
        "headers",
    }
)


def _vector_affecting(section: Any) -> dict[str, Any] | None:
    if section is None:
        return None
    data: dict[str, Any] = section.model_dump(mode="json", exclude={"batch_size"})
    data["options"] = {k: v for k, v in (data.get("options") or {}).items() if k not in OPERATIONAL_OPTIONS}
    return data


def embedding_fingerprint(config: PipelineConfig) -> str:
    emb = config.embedding
    return hash_json({"dense": _vector_affecting(emb.dense), "sparse": _vector_affecting(emb.sparse)})[:32]


def ingest_fingerprint(pipeline: CollectionPipeline, chunking: ChunkingConfig | None, *, prechunked: bool) -> str:
    return hash_json(
        {
            "embedding": pipeline.embedding_fingerprint,
            "chunking": "prechunked" if prechunked else (chunking or pipeline.config.chunking).model_dump(mode="json"),
            "enrichment": pipeline.config.enrichment.model_dump(mode="json"),
        }
    )[:32]


def _key(config: Any) -> str:
    return hash_json(config.model_dump(mode="json"))


class PipelineFactory:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._dense: dict[str, DenseEmbedder] = {}
        self._skipped_features: set[str] = set()
        self._sparse: dict[str, SparseEmbedder] = {}
        self._rerankers: dict[str, Reranker] = {}
        self._llms: dict[str, LLMClient] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ embedders

    async def dense_embedder(self, config: DenseEmbeddingConfig, cache_size: int) -> DenseEmbedder:
        key = _key(config)
        if key not in self._dense:
            async with self._lock:
                if key not in self._dense:
                    inner: DenseEmbedder = DENSE_EMBEDDERS.create(config.provider, config=config)
                    embedder: DenseEmbedder = CachedDenseEmbedder(inner, cache_size) if cache_size else inner
                    warm = getattr(embedder, "warmup", None)
                    if warm is not None and not embedder.dimension:
                        await warm()  # some providers (OpenAI-compatible) discover their dimension on first call
                    if not embedder.dimension:
                        raise ConfigurationError(f"Could not determine the dimension of dense model '{config.model}'")
                    self._dense[key] = embedder
        return self._dense[key]

    async def sparse_embedder(self, config: SparseEmbeddingConfig, cache_size: int) -> SparseEmbedder:
        key = _key(config)
        if key not in self._sparse:
            async with self._lock:
                if key not in self._sparse:
                    inner: SparseEmbedder = SPARSE_EMBEDDERS.create(config.provider, config=config)
                    self._sparse[key] = CachedSparseEmbedder(inner, cache_size) if cache_size else inner
        return self._sparse[key]

    async def build(self, name: str, config: PipelineConfig) -> CollectionPipeline:
        emb = config.embedding
        dense = await self.dense_embedder(emb.dense, emb.query_cache_size) if emb.dense else None
        sparse = await self.sparse_embedder(emb.sparse, emb.query_cache_size) if emb.sparse else None
        # Validate the remaining strategy names eagerly so bad config fails at creation time.
        for registry, value in (
            (CHUNKERS, config.chunking.strategy),
            (FUSIONS, config.retrieval.fusion.method),
            (QUERY_TRANSFORMS, config.retrieval.query_transform.strategy),
        ):
            if value not in registry:
                raise ConfigurationError(f"Unknown {registry.kind} '{value}'. Available: {', '.join(registry.names())}")
        if config.retrieval.rerank.enabled and config.retrieval.rerank.provider not in RERANKERS:
            raise ConfigurationError(f"Unknown reranker '{config.retrieval.rerank.provider}'")
        return CollectionPipeline(
            name=name,
            config=config,
            dense=dense,
            sparse=sparse,
            embedding_fingerprint=embedding_fingerprint(config),
        )

    async def warmup(self) -> None:
        for embedder in [*self._dense.values(), *self._sparse.values(), *self._rerankers.values()]:
            warm = getattr(embedder, "warmup", None)
            if warm is not None:
                await warm()

    def collection_spec(
        self,
        pipeline: CollectionPipeline,
        capabilities: StoreCapabilities,
        payload_indexes: list[dict[str, Any]],
    ) -> CollectionSpec:
        emb = pipeline.config.embedding
        if pipeline.sparse is not None and pipeline.sparse.requires_idf and not capabilities.sparse_idf:
            log.warning(
                "pipeline.sparse_idf_unsupported",
                collection=pipeline.name,
                model=pipeline.sparse.model_id,
                hint="This store cannot apply IDF; prefer a learned sparse model such as SPLADE.",
            )
        return CollectionSpec(
            name=pipeline.name,
            dense_dim=pipeline.dense.dimension if pipeline.dense else None,
            distance=emb.dense.distance if emb.dense else None,
            sparse=pipeline.sparse is not None,
            sparse_idf=bool(pipeline.sparse and pipeline.sparse.requires_idf),
            payload_indexes=[(p["field"], p.get("type", "keyword")) for p in payload_indexes],
            tenant_fields=[p["field"] for p in payload_indexes if p.get("tenant")],
        )

    # ------------------------------------------------------------------ per-use components

    def chunker(self, config: ChunkingConfig, dense: DenseEmbedder | None) -> Chunker:
        chunker: Chunker = CHUNKERS.create(
            config.strategy, config=config, length_fn=length_function_for(config), dense_embedder=dense
        )
        return chunker

    def header_enricher(self, config: PipelineConfig) -> ContextualHeaderEnricher:
        return ContextualHeaderEnricher(config.enrichment.contextual_header)

    def llm_context_enricher(self, config: PipelineConfig) -> LLMContextEnricher | None:
        if not config.enrichment.llm_context.enabled:
            return None
        llm = self.utility_llm()
        if not self._can_generate(llm, "enrichment.llm_context"):
            return None
        return LLMContextEnricher(config.enrichment.llm_context, llm)

    def fusion(self, method: str) -> FusionStrategy:
        fusion: FusionStrategy = FUSIONS.create(method)
        return fusion

    def reranker(self, config: RerankConfig) -> Reranker:
        key = hash_json(
            {
                "provider": config.provider,
                "model": config.model,
                "options": config.options,
                "batch": config.batch_size,
            }
        )
        if key not in self._rerankers:
            self._rerankers[key] = RERANKERS.create(config.provider, config=config)
        return self._rerankers[key]

    def query_transformer(self, config: QueryTransformConfig) -> QueryTransformer:
        llm = None if config.strategy == "none" else self.utility_llm()
        if llm is not None and not self._can_generate(llm, f"query_transform.{config.strategy}"):
            config, llm = config.model_copy(update={"strategy": "none"}), None
        transformer: QueryTransformer = QUERY_TRANSFORMS.create(config.strategy, config=config, llm=llm)
        return transformer

    def _can_generate(self, llm: LLMClient, feature: str) -> bool:
        """LLM-driven features are skipped (with one warning each) when the LLM cannot write text, e.g. the
        offline ``extractive`` provider — its output would be quoted context, not a query or a summary."""
        if self.is_generative(llm):
            return True
        if feature not in self._skipped_features:
            self._skipped_features.add(feature)
            log.warning("pipeline.feature_skipped", feature=feature, llm=llm.provider, hint="needs a generative LLM")
        return False

    # ------------------------------------------------------------------ LLMs

    def _llm(self, config: LLMConfig) -> LLMClient:
        key = _key(config)
        if key not in self._llms:
            self._llms[key] = LLMS.create(config.provider, config=config)
        return self._llms[key]

    def llm(self) -> LLMClient:
        return self._llm(self.settings.llm)

    def utility_llm(self) -> LLMClient:
        return self._llm(self.settings.effective_utility_llm())

    def is_generative(self, llm: LLMClient) -> bool:
        return llm.provider != "extractive"

    async def close(self) -> None:
        for component in [
            *self._llms.values(),
            *self._rerankers.values(),
            *self._dense.values(),
            *self._sparse.values(),
        ]:
            target = getattr(component, "inner", component)
            close = getattr(target, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception as exc:  # shutdown must continue
                    log.warning("component.close_failed", component=type(target).__name__, error=str(exc))
