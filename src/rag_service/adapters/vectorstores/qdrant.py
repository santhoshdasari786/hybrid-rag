"""Qdrant adapter (extra ``qdrant``).

* One Qdrant collection per logical collection, named ``<collection_prefix><name>``.
* Named vectors ``dense`` (HNSW) and ``sparse`` (inverted index, optional IDF modifier).
* Payload indexes on ``source_id`` / ``chunk_index`` are always created (fast deletes and
  neighbour expansion); extra indexes come from ``collections.<name>.payload_indexes``.
* ``search_hybrid`` uses Qdrant's Query API (``prefetch`` + weighted RRF or DBSF).
* Works against a server (``url``), embedded in-memory (``location: ':memory:'``) or embedded on
  disk (``path``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Any, TypeVar

from rag_service.core.config import DistanceMetric, FusionConfig, QdrantConfig, VectorStoreConfig
from rag_service.core.errors import (
    CapabilityNotSupportedError,
    CollectionNotFoundError,
    DependencyMissingError,
    EmbeddingConfigChangedError,
    ProviderError,
    ProviderUnavailableError,
)
from rag_service.core.logging import get_logger
from rag_service.core.registry import VECTOR_STORES
from rag_service.core.utils import batched
from rag_service.domain.filters import RANGE_OPS, Condition, FilterOp, FilterSpec
from rag_service.domain.models import (
    CollectionSpec,
    PayloadKeys,
    PayloadPatch,
    ScoredPoint,
    SparseVector,
    VectorRecord,
)
from rag_service.domain.ports import StoreCapabilities

log = get_logger(__name__)
T = TypeVar("T")

DENSE = "dense"
SPARSE = "sparse"
_UPSERT_BATCH = 256
_PATCH_BATCH = 512


def _models() -> Any:
    try:
        from qdrant_client import models  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise DependencyMissingError.for_extra("qdrant-client", "qdrant") from exc
    return models


class QdrantFilterTranslator:
    """FilterSpec → ``qdrant_client.models.Filter`` (semantics mirror ``domain.filters.matches``)."""

    def __init__(self) -> None:
        self.m = _models()

    def translate(self, spec: FilterSpec | None) -> Any:
        if spec is None or spec.is_empty():
            return None
        return self._group(spec)

    def _group(self, spec: FilterSpec) -> Any:
        m = self.m
        must: list[Any] = []
        must_not: list[Any] = []
        for item in spec.must:
            positive, negative = self._item(item)
            must.extend(positive)
            if negative:
                must.append(m.Filter(must_not=negative))
        should = []
        for item in spec.should:
            positive, negative = self._item(item)
            parts = list(positive) + ([m.Filter(must_not=negative)] if negative else [])
            should.append(parts[0] if len(parts) == 1 else m.Filter(must=parts))
        for item in spec.must_not:
            positive, negative = self._item(item)
            must_not.extend(positive)
            if negative:
                must_not.append(m.Filter(must_not=negative))
        return m.Filter(must=must or None, should=should or None, must_not=must_not or None)

    def _item(self, item: Condition | FilterSpec) -> tuple[list[Any], list[Any]]:
        """Returns (conditions that must hold, conditions that must NOT hold)."""
        m = self.m
        if isinstance(item, FilterSpec):
            return [self._group(item)], []
        key, op, value = item.path, item.op, item.value
        if op == FilterOp.EXISTS:
            empty = m.IsEmptyCondition(is_empty=m.PayloadField(key=key))
            return ([], [empty]) if value else ([empty], [])
        if op in (FilterOp.IN, FilterOp.NIN):
            cond = m.FieldCondition(key=key, match=m.MatchAny(any=list(value)))
            return ([cond], []) if op == FilterOp.IN else ([], [cond])
        if op in RANGE_OPS:
            bound = {op.value: value}
            rng = m.DatetimeRange(**bound) if isinstance(value, str) else m.Range(**bound)
            return [m.FieldCondition(key=key, range=rng)], []
        # EQ / NE — MatchValue only supports str/int/bool; floats use a closed range.
        if isinstance(value, float):
            cond = m.FieldCondition(key=key, range=m.Range(gte=value, lte=value))
        else:
            cond = m.FieldCondition(key=key, match=m.MatchValue(value=value))
        return ([cond], []) if op == FilterOp.EQ else ([], [cond])


class QdrantVectorStore:
    name = "qdrant"
    capabilities = StoreCapabilities(dense=True, sparse=True, native_hybrid=True, sparse_idf=True)

    def __init__(self, config: QdrantConfig) -> None:
        self.m = _models()
        from qdrant_client import AsyncQdrantClient  # noqa: PLC0415

        self.config = config
        if config.location or config.path:
            self._client = AsyncQdrantClient(location=config.location, path=config.path)
            self._embedded = True
        else:
            self._client = AsyncQdrantClient(
                url=config.url,
                api_key=config.api_key.get_secret_value() if config.api_key else None,
                prefer_grpc=config.prefer_grpc,
                timeout=config.timeout_s,
            )
            self._embedded = False
        self._filters = QdrantFilterTranslator()
        self._search_params = self.m.SearchParams(hnsw_ef=config.hnsw.ef_search) if config.hnsw.ef_search else None

    def physical(self, name: str) -> str:
        return f"{self.config.collection_prefix}{name}"

    async def _call(self, collection: str | None, awaitable: Awaitable[T]) -> T:
        try:
            return await awaitable
        except (CollectionNotFoundError, ProviderError):
            raise
        except Exception as exc:
            text = str(exc)
            if collection and ("Not found" in text or "doesn't exist" in text or "not found" in text):
                raise CollectionNotFoundError(f"Vector collection '{collection}' does not exist") from exc
            if isinstance(exc, ConnectionError | TimeoutError) or "Connection" in type(exc).__name__:
                raise ProviderUnavailableError(f"Qdrant unavailable: {exc}") from exc
            raise ProviderError(f"Qdrant error: {exc}") from exc

    # ------------------------------------------------------------------ admin

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        m = self.m
        name = self.physical(spec.name)
        if await self._call(None, self._client.collection_exists(name)):
            await self._verify_compatible(spec, name)
            return
        vectors: dict[str, Any] = {}
        if spec.dense_dim:
            distance = {
                DistanceMetric.COSINE: m.Distance.COSINE,
                DistanceMetric.DOT: m.Distance.DOT,
                DistanceMetric.EUCLID: m.Distance.EUCLID,
            }[spec.distance or DistanceMetric.COSINE]
            vectors[DENSE] = m.VectorParams(size=spec.dense_dim, distance=distance, on_disk=self.config.on_disk_vectors)
        sparse = (
            {
                SPARSE: m.SparseVectorParams(
                    index=m.SparseIndexParams(on_disk=False),
                    modifier=m.Modifier.IDF if spec.sparse_idf else None,
                )
            }
            if spec.sparse
            else None
        )
        quantization = None
        if self.config.quantization == "scalar":
            quantization = m.ScalarQuantization(
                scalar=m.ScalarQuantizationConfig(type=m.ScalarType.INT8, always_ram=True)
            )
        elif self.config.quantization == "binary":
            quantization = m.BinaryQuantization(binary=m.BinaryQuantizationConfig(always_ram=True))
        await self._call(
            None,
            self._client.create_collection(
                name,
                vectors_config=vectors or {},
                sparse_vectors_config=sparse,
                hnsw_config=m.HnswConfigDiff(m=self.config.hnsw.m, ef_construct=self.config.hnsw.ef_construct),
                on_disk_payload=self.config.on_disk_payload,
                quantization_config=quantization,
                shard_number=self.config.shard_number,
                replication_factor=self.config.replication_factor,
            ),
        )
        log.info("qdrant.collection.created", collection=name, dense_dim=spec.dense_dim, sparse=spec.sparse)
        if not self._embedded:  # payload indexes are a no-op in embedded mode
            indexes = [
                (PayloadKeys.SOURCE_ID, "keyword"),
                (PayloadKeys.CHUNK_INDEX, "integer"),
                *spec.payload_indexes,
            ]
            for field, kind in indexes:
                path = field if field in PayloadKeys.SYSTEM or field.startswith("metadata.") else f"metadata.{field}"
                schema: Any = m.PayloadSchemaType(kind)
                if field in spec.tenant_fields:
                    # Co-locates each tenant's points on disk and lets Qdrant optimise per-tenant search.
                    schema = m.KeywordIndexParams(type=m.KeywordIndexType.KEYWORD, is_tenant=True)
                await self._call(name, self._client.create_payload_index(name, path, field_schema=schema))

    async def _verify_compatible(self, spec: CollectionSpec, name: str) -> None:
        info = await self._call(spec.name, self._client.get_collection(name))
        vectors = info.config.params.vectors
        dense_params = vectors.get(DENSE) if isinstance(vectors, dict) else None
        existing_dim = dense_params.size if dense_params is not None else None
        if spec.dense_dim and existing_dim and existing_dim != spec.dense_dim:
            raise EmbeddingConfigChangedError(
                f"Collection '{spec.name}' was created with dense dimension {existing_dim}, configuration now "
                f"produces {spec.dense_dim}. Rebuild it with POST /collections/{spec.name}/reindex?recreate=true."
            )

    async def drop_collection(self, name: str) -> None:
        physical = self.physical(name)
        if await self._call(None, self._client.collection_exists(physical)):
            await self._call(name, self._client.delete_collection(physical))

    async def collection_exists(self, name: str) -> bool:
        return bool(await self._call(None, self._client.collection_exists(self.physical(name))))

    async def count(self, name: str, filter_: FilterSpec | None = None) -> int:
        result = await self._call(
            name,
            self._client.count(self.physical(name), count_filter=self._filters.translate(filter_), exact=True),
        )
        return int(result.count)

    # ------------------------------------------------------------------ writes

    def _sparse(self, vector: SparseVector) -> Any:
        return self.m.SparseVector(indices=list(vector.indices), values=list(vector.values))

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        m = self.m
        for batch in batched(records, _UPSERT_BATCH):
            points = []
            for r in batch:
                vector: dict[str, Any] = {}
                if r.dense is not None:
                    vector[DENSE] = r.dense
                if r.sparse is not None and len(r.sparse):
                    vector[SPARSE] = self._sparse(r.sparse)
                points.append(m.PointStruct(id=r.id, vector=vector, payload=r.payload))
            await self._call(collection, self._client.upsert(self.physical(collection), points=points, wait=True))

    async def update_payloads(self, collection: str, patches: Sequence[PayloadPatch]) -> None:
        m = self.m
        for batch in batched(patches, _PATCH_BATCH):
            ops = [m.SetPayloadOperation(set_payload=m.SetPayload(payload=p.payload, points=[p.id])) for p in batch]
            await self._call(
                collection,
                self._client.batch_update_points(self.physical(collection), update_operations=ops, wait=True),
            )

    async def delete_ids(self, collection: str, ids: Sequence[str]) -> None:
        for batch in batched(list(ids), 1_000):
            await self._call(
                collection,
                self._client.delete(
                    self.physical(collection),
                    points_selector=self.m.PointIdsList(points=list(batch)),
                    wait=True,
                ),
            )

    async def delete_by_filter(self, collection: str, filter_: FilterSpec) -> None:
        await self._call(
            collection,
            self._client.delete(
                self.physical(collection),
                points_selector=self.m.FilterSelector(filter=self._filters.translate(filter_)),
                wait=True,
            ),
        )

    # ------------------------------------------------------------------ reads

    def _to_point(self, p: Any, with_vectors: bool = False) -> ScoredPoint:
        dense = None
        if with_vectors and isinstance(p.vector, dict):
            raw = p.vector.get(DENSE)
            dense = list(raw) if raw is not None else None
        return ScoredPoint(
            id=str(p.id),
            score=float(getattr(p, "score", 0.0) or 0.0),
            payload=dict(p.payload or {}),
            dense=dense,
        )

    async def search_dense(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int,
        filter_: FilterSpec | None = None,
        score_threshold: float | None = None,
        with_vectors: bool = False,
    ) -> list[ScoredPoint]:
        response = await self._call(
            collection,
            self._client.query_points(
                self.physical(collection),
                query=list(vector),
                using=DENSE,
                limit=limit,
                query_filter=self._filters.translate(filter_),
                score_threshold=score_threshold,
                search_params=self._search_params,
                with_payload=True,
                with_vectors=[DENSE] if with_vectors else False,
            ),
        )
        return [self._to_point(p, with_vectors) for p in response.points]

    async def search_sparse(
        self,
        collection: str,
        vector: SparseVector,
        *,
        limit: int,
        filter_: FilterSpec | None = None,
    ) -> list[ScoredPoint]:
        if not len(vector):
            return []
        response = await self._call(
            collection,
            self._client.query_points(
                self.physical(collection),
                query=self._sparse(vector),
                using=SPARSE,
                limit=limit,
                query_filter=self._filters.translate(filter_),
                with_payload=True,
            ),
        )
        return [self._to_point(p) for p in response.points]

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
        m = self.m
        qfilter = self._filters.translate(filter_)
        prefetch = [
            m.Prefetch(query=list(dense), using=DENSE, limit=candidate_k, filter=qfilter, params=self._search_params),
            m.Prefetch(query=self._sparse(sparse), using=SPARSE, limit=candidate_k, filter=qfilter),
        ]
        method = fusion.method.lower()
        if method == "rrf":
            weights = [fusion.weights.get("dense", 1.0), fusion.weights.get("sparse", 1.0)]
            query: Any = m.RrfQuery(rrf=m.Rrf(k=fusion.k, weights=weights))
        elif method == "dbsf":
            query = m.FusionQuery(fusion=m.Fusion.DBSF)
        else:
            raise CapabilityNotSupportedError(f"Qdrant has no native '{fusion.method}' fusion; use execution=client")
        response = await self._call(
            collection,
            self._client.query_points(
                self.physical(collection), prefetch=prefetch, query=query, limit=limit, with_payload=True
            ),
        )
        return [self._to_point(p) for p in response.points]

    async def fetch(self, collection: str, ids: Sequence[str], *, with_vectors: bool = False) -> list[ScoredPoint]:
        if not ids:
            return []
        records = await self._call(
            collection,
            self._client.retrieve(
                self.physical(collection),
                ids=list(ids),
                with_payload=True,
                with_vectors=[DENSE] if with_vectors else False,
            ),
        )
        return [self._to_point(r, with_vectors) for r in records]

    async def scroll(
        self,
        collection: str,
        *,
        filter_: FilterSpec | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[ScoredPoint], str | None]:
        records, next_offset = await self._call(
            collection,
            self._client.scroll(
                self.physical(collection),
                scroll_filter=self._filters.translate(filter_),
                limit=limit,
                offset=offset,
                with_payload=True,
            ),
        )
        return [self._to_point(r) for r in records], (str(next_offset) if next_offset is not None else None)

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()


@VECTOR_STORES.register("qdrant", description="Qdrant (HNSW dense + sparse inverted index, native RRF/DBSF)")
def _qdrant(*, config: VectorStoreConfig) -> QdrantVectorStore:
    return QdrantVectorStore(config.qdrant)
