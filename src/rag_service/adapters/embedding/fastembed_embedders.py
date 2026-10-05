"""FastEmbed (ONNX Runtime) embedders — dense (BGE, E5, GTE, Nomic…) and sparse (SPLADE, BM25, BM42).

Models are loaded lazily on first use (or eagerly via ``warmup``) and inference runs in a worker
thread so the event loop is never blocked. Install with the ``fastembed`` extra.

Useful options (``embedding.dense.options`` / ``embedding.sparse.options``):
``cache_dir``, ``threads``, ``providers`` (e.g. ``["CUDAExecutionProvider"]``), ``parallel``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Any

from rag_service.core.config import DenseEmbeddingConfig, SparseEmbeddingConfig
from rag_service.core.errors import ConfigurationError, DependencyMissingError, ProviderError
from rag_service.core.logging import get_logger
from rag_service.core.registry import DENSE_EMBEDDERS, SPARSE_EMBEDDERS
from rag_service.domain.models import SparseVector

log = get_logger(__name__)

_MODEL_OPTIONS = ("cache_dir", "threads", "providers", "cuda", "device_ids", "lazy_load")


def _require_fastembed() -> Any:
    try:
        import fastembed  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise DependencyMissingError.for_extra("fastembed", "fastembed") from exc
    return fastembed


class _LazyModel:
    """Thread-safe lazy loader shared by dense and sparse wrappers."""

    def __init__(self, factory: Any, model_name: str, options: dict[str, Any]) -> None:
        self._factory = factory
        self._model_name = model_name
        self._kwargs = {k: v for k, v in options.items() if k in _MODEL_OPTIONS}
        self._model: Any = None
        self._lock = threading.Lock()

    def get(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    log.info("embedding.model.loading", model=self._model_name)
                    try:
                        self._model = self._factory(model_name=self._model_name, **self._kwargs)
                    except Exception as exc:
                        raise ProviderError(f"Failed to load embedding model '{self._model_name}': {exc}") from exc
                    log.info("embedding.model.loaded", model=self._model_name)
        return self._model


class FastEmbedDenseEmbedder:
    def __init__(self, config: DenseEmbeddingConfig) -> None:
        fastembed = _require_fastembed()
        self.model_id = config.model
        self.distance = config.distance
        self.batch_size = config.batch_size
        self.parallel = config.options.get("parallel")
        self.query_prefix = config.query_prefix
        self.document_prefix = config.document_prefix
        self.dimension = config.dimension or self._lookup_dimension(fastembed.TextEmbedding, config.model)
        self._model = _LazyModel(fastembed.TextEmbedding, config.model, config.options)

    @staticmethod
    def _lookup_dimension(model_cls: Any, model: str) -> int:
        for info in model_cls.list_supported_models():
            if str(info.get("model", "")).lower() == model.lower():
                return int(info["dim"])
        raise ConfigurationError(
            f"FastEmbed model '{model}' is not in the supported list; set embedding.dense.dimension explicitly."
        )

    async def warmup(self) -> None:
        await asyncio.to_thread(self._model.get)

    def _embed_sync(self, texts: list[str], *, query: bool) -> list[list[float]]:
        model = self._model.get()
        if query:
            vectors = model.query_embed(texts)
        else:
            vectors = model.passage_embed(texts, batch_size=self.batch_size, parallel=self.parallel)
        return [v.tolist() for v in vectors]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [self.document_prefix + t for t in texts]
        return await asyncio.to_thread(self._embed_sync, prepared, query=False)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(self._embed_sync, [self.query_prefix + text], query=True)
        return vectors[0]


class FastEmbedSparseEmbedder:
    def __init__(self, config: SparseEmbeddingConfig) -> None:
        fastembed = _require_fastembed()
        self.model_id = config.model
        self.batch_size = config.batch_size
        self.parallel = config.options.get("parallel")
        info = next(
            (
                m
                for m in fastembed.SparseTextEmbedding.list_supported_models()
                if str(m.get("model", "")).lower() == config.model.lower()
            ),
            None,
        )
        if info is None:
            raise ConfigurationError(f"FastEmbed sparse model '{config.model}' is not supported.")
        self.requires_idf = bool(info.get("requires_idf"))
        self._model = _LazyModel(fastembed.SparseTextEmbedding, config.model, config.options)

    async def warmup(self) -> None:
        await asyncio.to_thread(self._model.get)

    def _embed_sync(self, texts: list[str], *, query: bool) -> list[SparseVector]:
        model = self._model.get()
        if query:
            embeddings = model.query_embed(texts)
        else:
            embeddings = model.passage_embed(texts, batch_size=self.batch_size, parallel=self.parallel)
        return [SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in embeddings]

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return await asyncio.to_thread(self._embed_sync, list(texts), query=False)

    async def embed_query(self, text: str) -> SparseVector:
        vectors = await asyncio.to_thread(self._embed_sync, [text], query=True)
        return vectors[0]


@DENSE_EMBEDDERS.register("fastembed", description="FastEmbed ONNX dense models (BGE, E5, GTE, Nomic, …)")
def _dense(*, config: DenseEmbeddingConfig) -> FastEmbedDenseEmbedder:
    return FastEmbedDenseEmbedder(config)


@SPARSE_EMBEDDERS.register("fastembed", description="FastEmbed sparse models (SPLADE++, BM25, BM42, miniCOIL)")
def _sparse(*, config: SparseEmbeddingConfig) -> FastEmbedSparseEmbedder:
    return FastEmbedSparseEmbedder(config)
