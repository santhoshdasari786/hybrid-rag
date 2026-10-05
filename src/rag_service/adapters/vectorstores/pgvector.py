"""PostgreSQL + pgvector adapter (extra ``pgvector``). Requires pgvector >= 0.7 (``sparsevec``).

Schema: one table per logical collection, ``<db_schema>.<table_prefix><name>``::

    id uuid PK · source_id text · chunk_index int · payload jsonb · dense vector(D) · sparse sparsevec(S)

* Dense search uses an HNSW index with the operator class matching the configured distance.
  On pgvector >= 0.8 iterative index scans are enabled so filtered queries still return ``limit`` rows.
* Sparse search ranks by inner product (``<#>``). pgvector has no IDF modifier, so pair it with
  learned sparse models (SPLADE) rather than raw term-frequency vectors.
* Metadata filters compile to JSONPath predicates (``payload @? '...'``) that use the GIN index.
  Values are JSON-encoded and the whole path is a bind parameter, so filters cannot inject SQL.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from rag_service.core.config import DistanceMetric, FusionConfig, PgVectorConfig, VectorStoreConfig
from rag_service.core.errors import (
    CapabilityNotSupportedError,
    CollectionNotFoundError,
    DependencyMissingError,
    EmbeddingConfigChangedError,
    ProviderError,
)
from rag_service.core.logging import get_logger
from rag_service.core.registry import VECTOR_STORES
from rag_service.core.utils import batched
from rag_service.domain.filters import Condition, FilterOp, FilterSpec
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

_OPS = {
    DistanceMetric.COSINE: ("<=>", "vector_cosine_ops"),
    DistanceMetric.DOT: ("<#>", "vector_ip_ops"),
    DistanceMetric.EUCLID: ("<->", "vector_l2_ops"),
}
_JSONPATH_CMP = {FilterOp.GT: ">", FilterOp.GTE: ">=", FilterOp.LT: "<", FilterOp.LTE: "<="}


def dense_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.8g}" for v in vector) + "]"


def sparse_literal(vector: SparseVector, dimension: int) -> str:
    # sparsevec text format is 1-based: {index:value,...}/dimension
    return (
        "{"
        + ",".join(f"{i + 1}:{float(v):.8g}" for i, v in zip(vector.indices, vector.values, strict=True))
        + f"}}/{dimension}"
    )


def _jsonpath(path: str) -> str:
    return "$" + "".join(f".{json.dumps(part)}" for part in path.split("."))


class PgFilterCompiler:
    """FilterSpec → (SQL fragment, params). Semantics mirror ``domain.filters.matches``."""

    def compile(self, spec: FilterSpec | None) -> tuple[str, list[Any]]:
        if spec is None or spec.is_empty():
            return "TRUE", []
        return self._group(spec)

    def _group(self, spec: FilterSpec) -> tuple[str, list[Any]]:
        parts: list[str] = []
        params: list[Any] = []
        for item in spec.must:
            sql, p = self._item(item)
            parts.append(sql)
            params.extend(p)
        if spec.should:
            ors = []
            for item in spec.should:
                sql, p = self._item(item)
                ors.append(sql)
                params.extend(p)
            parts.append("(" + " OR ".join(ors) + ")")
        for item in spec.must_not:
            sql, p = self._item(item)
            parts.append(f"NOT ({sql})")
            params.extend(p)
        return "(" + " AND ".join(parts or ["TRUE"]) + ")", params

    def _item(self, item: Condition | FilterSpec) -> tuple[str, list[Any]]:
        if isinstance(item, FilterSpec):
            return self._group(item)
        path, op, value = item.path, item.op, item.value
        # Column fast-paths for system fields.
        if path == PayloadKeys.SOURCE_ID and op in (FilterOp.EQ, FilterOp.IN):
            values = value if op == FilterOp.IN else [value]
            return "source_id = ANY(%s)", [[str(v) for v in values]]
        jp = _jsonpath(path)
        if op == FilterOp.EXISTS:
            sql = "payload @? %s::jsonpath"
            return (sql if value else f"NOT ({sql})"), [f"{jp} ? (@ != null)"]
        if op in (FilterOp.EQ, FilterOp.NE):
            sql = "payload @? %s::jsonpath"
            return (sql if op == FilterOp.EQ else f"NOT ({sql})"), [f"{jp} ? (@ == {json.dumps(value)})"]
        if op in (FilterOp.IN, FilterOp.NIN):
            predicate = " || ".join(f"@ == {json.dumps(v)}" for v in value)
            sql = "payload @? %s::jsonpath"
            return (sql if op == FilterOp.IN else f"NOT ({sql})"), [f"{jp} ? ({predicate})"]
        return "payload @? %s::jsonpath", [f"{jp} ? (@ {_JSONPATH_CMP[op]} {json.dumps(value)})"]


class PgVectorStore:
    name = "pgvector"
    capabilities = StoreCapabilities(dense=True, sparse=True, native_hybrid=False, sparse_idf=False)

    def __init__(self, config: PgVectorConfig) -> None:
        try:
            from psycopg import sql  # noqa: PLC0415 - optional dependency
            from psycopg_pool import AsyncConnectionPool  # noqa: PLC0415
        except ImportError as exc:
            raise DependencyMissingError.for_extra("psycopg[pool]", "pgvector") from exc
        self.sql = sql
        self.config = config
        self._pool = AsyncConnectionPool(
            config.dsn.get_secret_value(),
            min_size=config.pool_min_size,
            max_size=config.pool_max_size,
            open=False,
        )
        self._opened = False
        self._iterative_scan = False
        self._filters = PgFilterCompiler()
        self._specs: dict[str, CollectionSpec] = {}

    async def _open(self) -> None:
        if not self._opened:
            await self._pool.open(wait=True, timeout=30)
            self._opened = True
            async with self._pool.connection() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute(
                    self.sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(self.sql.Identifier(self.config.db_schema))
                )
                row = await (
                    await conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                ).fetchone()
                version = tuple(int(x) for x in str(row[0]).split(".")[:2]) if row else (0, 0)
                self._iterative_scan = version >= (0, 8)
                log.info("pgvector.ready", version=row[0] if row else None)

    def _table(self, name: str) -> Any:
        return self.sql.Identifier(self.config.db_schema, f"{self.config.table_prefix}{name}")

    async def _exec(
        self,
        collection: str | None,
        query: Any,
        params: Sequence[Any] | None = None,
        *,
        fetch: bool = False,
        setup: Sequence[str] = (),
    ) -> list[Any]:
        await self._open()
        try:
            async with self._pool.connection() as conn, conn.transaction():
                for stmt in setup:
                    await conn.execute(stmt)
                cur = await conn.execute(query, params)
                return list(await cur.fetchall()) if fetch else []
        except Exception as exc:
            name = type(exc).__name__
            if collection and name == "UndefinedTable":
                raise CollectionNotFoundError(f"Vector collection '{collection}' does not exist") from exc
            raise ProviderError(f"pgvector error: {exc}") from exc

    async def _spec(self, name: str) -> CollectionSpec:
        spec = self._specs.get(name)
        if spec is None:
            rows = await self._exec(
                name,
                "SELECT format_type(a.atttypid, a.atttypmod), a.attname FROM pg_attribute a "
                "WHERE a.attrelid = %s::regclass AND a.attname IN ('dense', 'sparse') AND NOT a.attisdropped",
                [f'"{self.config.db_schema}"."{self.config.table_prefix}{name}"'],
                fetch=True,
            )
            if not rows:
                raise CollectionNotFoundError(f"Vector collection '{name}' does not exist")
            types = {r[1]: r[0] for r in rows}
            dim = int(types["dense"].split("(")[1].rstrip(")")) if "dense" in types else None
            spec = CollectionSpec(name=name, dense_dim=dim, distance=DistanceMetric.COSINE, sparse="sparse" in types)
            self._specs[name] = spec
        return spec

    # ------------------------------------------------------------------ admin

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        sql = self.sql
        table = self._table(spec.name)
        if await self.collection_exists(spec.name):
            existing = await self._spec(spec.name)
            if spec.dense_dim and existing.dense_dim and existing.dense_dim != spec.dense_dim:
                raise EmbeddingConfigChangedError(
                    f"Collection '{spec.name}' has dense dimension {existing.dense_dim}; configuration produces "
                    f"{spec.dense_dim}. Rebuild it with POST /collections/{spec.name}/reindex?recreate=true."
                )
            self._specs[spec.name] = spec
            return
        columns = [
            sql.SQL("id uuid PRIMARY KEY"),
            sql.SQL("source_id text NOT NULL"),
            sql.SQL("chunk_index integer NOT NULL DEFAULT 0"),
            sql.SQL("payload jsonb NOT NULL DEFAULT '{}'::jsonb"),
        ]
        if spec.dense_dim:
            columns.append(sql.SQL("dense vector({})").format(sql.Literal(spec.dense_dim)))
        if spec.sparse:
            columns.append(sql.SQL("sparse sparsevec({})").format(sql.Literal(self.config.sparse_dimension)))
        await self._exec(None, sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(table, sql.SQL(", ").join(columns)))
        prefix = f"{self.config.table_prefix}{spec.name}"
        await self._exec(
            None,
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (source_id, chunk_index)").format(
                sql.Identifier(f"{prefix}_src_idx"), table
            ),
        )
        await self._exec(
            None,
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING gin (payload jsonb_path_ops)").format(
                sql.Identifier(f"{prefix}_payload_idx"), table
            ),
        )
        if spec.dense_dim:
            _, opclass = _OPS[spec.distance or DistanceMetric.COSINE]
            await self._exec(
                None,
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (dense {}) WITH (m = {}, ef_construction = {})"
                ).format(
                    sql.Identifier(f"{prefix}_dense_hnsw"),
                    table,
                    sql.SQL(opclass),
                    sql.Literal(self.config.hnsw_m),
                    sql.Literal(self.config.hnsw_ef_construction),
                ),
            )
        self._specs[spec.name] = spec
        log.info("pgvector.collection.created", collection=spec.name, dense_dim=spec.dense_dim, sparse=spec.sparse)

    async def drop_collection(self, name: str) -> None:
        await self._exec(None, self.sql.SQL("DROP TABLE IF EXISTS {}").format(self._table(name)))
        self._specs.pop(name, None)

    async def collection_exists(self, name: str) -> bool:
        rows = await self._exec(
            None,
            "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
            [self.config.db_schema, f"{self.config.table_prefix}{name}"],
            fetch=True,
        )
        return bool(rows)

    async def count(self, name: str, filter_: FilterSpec | None = None) -> int:
        where, params = self._filters.compile(filter_)
        rows = await self._exec(
            name,
            self.sql.SQL("SELECT count(*) FROM {} WHERE " + where).format(self._table(name)),
            params,
            fetch=True,
        )
        return int(rows[0][0])

    # ------------------------------------------------------------------ writes

    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        sql = self.sql
        await self._open()
        query = sql.SQL(
            "INSERT INTO {} (id, source_id, chunk_index, payload, dense, sparse) "
            "VALUES (%s, %s, %s, %s::jsonb, %s::vector, %s::sparsevec) "
            "ON CONFLICT (id) DO UPDATE SET source_id = EXCLUDED.source_id, chunk_index = EXCLUDED.chunk_index, "
            "payload = EXCLUDED.payload, dense = EXCLUDED.dense, sparse = EXCLUDED.sparse"
        ).format(self._table(collection))
        spec = await self._spec(collection)
        for batch in batched(records, 256):
            rows = [
                (
                    r.id,
                    str(r.payload.get(PayloadKeys.SOURCE_ID, "")),
                    int(r.payload.get(PayloadKeys.CHUNK_INDEX, 0) or 0),
                    json.dumps(r.payload, default=str),
                    dense_literal(r.dense) if r.dense is not None and spec.dense_dim else None,
                    sparse_literal(r.sparse, self.config.sparse_dimension)
                    if r.sparse is not None and spec.sparse
                    else None,
                )
                for r in batch
            ]
            try:
                async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                    await cur.executemany(query, rows)
            except Exception as exc:
                raise ProviderError(f"pgvector upsert failed: {exc}") from exc

    async def update_payloads(self, collection: str, patches: Sequence[PayloadPatch]) -> None:
        await self._open()
        query = self.sql.SQL(
            "UPDATE {} SET payload = payload || %s::jsonb, "
            "chunk_index = COALESCE((%s::jsonb ->> 'chunk_index')::int, chunk_index) WHERE id = %s"
        ).format(self._table(collection))
        for batch in batched(patches, 512):
            rows = [(json.dumps(p.payload, default=str), json.dumps(p.payload, default=str), p.id) for p in batch]
            try:
                async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                    await cur.executemany(query, rows)
            except Exception as exc:
                raise ProviderError(f"pgvector payload update failed: {exc}") from exc

    async def delete_ids(self, collection: str, ids: Sequence[str]) -> None:
        for batch in batched(list(ids), 1_000):
            await self._exec(
                collection,
                self.sql.SQL("DELETE FROM {} WHERE id = ANY(%s::uuid[])").format(self._table(collection)),
                [list(batch)],
            )

    async def delete_by_filter(self, collection: str, filter_: FilterSpec) -> None:
        where, params = self._filters.compile(filter_)
        await self._exec(
            collection, self.sql.SQL("DELETE FROM {} WHERE " + where).format(self._table(collection)), params
        )

    # ------------------------------------------------------------------ reads

    @staticmethod
    def _parse_dense(raw: Any) -> list[float] | None:
        if raw is None:
            return None
        return [float(x) for x in str(raw).strip("[]").split(",") if x]

    def _rows_to_points(self, rows: list[Any], score_fn: Any = None, with_vectors: bool = False) -> list[ScoredPoint]:
        points = []
        for row in rows:
            payload = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            score = score_fn(float(row[2])) if score_fn is not None else 0.0
            dense = self._parse_dense(row[3]) if with_vectors and len(row) > 3 else None
            points.append(ScoredPoint(id=str(row[0]), score=score, payload=payload, dense=dense))
        return points

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
        spec = await self._spec(collection)
        metric = spec.distance or DistanceMetric.COSINE
        op, _ = _OPS[metric]
        where, params = self._filters.compile(filter_)
        vector_col = ", dense::text" if with_vectors else ""
        query = self.sql.SQL(
            f"SELECT id, payload, dense {op} %s::vector AS d{vector_col} FROM {{}} "
            f"WHERE dense IS NOT NULL AND {where} ORDER BY d LIMIT %s"
        ).format(self._table(collection))
        setup = [f"SET LOCAL hnsw.ef_search = {int(self.config.ef_search)}"]
        if self._iterative_scan and filter_ is not None:
            setup.append("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
        rows = await self._exec(collection, query, [dense_literal(vector), *params, limit], fetch=True, setup=setup)
        to_score = {
            DistanceMetric.COSINE: lambda d: 1.0 - d,
            DistanceMetric.DOT: lambda d: -d,
            DistanceMetric.EUCLID: lambda d: -d,
        }[metric]
        points = self._rows_to_points(rows, to_score, with_vectors)
        if score_threshold is not None:
            points = [p for p in points if p.score >= score_threshold]
        return points

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
        where, params = self._filters.compile(filter_)
        query = self.sql.SQL(
            f"SELECT id, payload, sparse <#> %s::sparsevec AS d FROM {{}} "
            f"WHERE sparse IS NOT NULL AND {where} ORDER BY d LIMIT %s"
        ).format(self._table(collection))
        rows = await self._exec(
            collection,
            query,
            [sparse_literal(vector, self.config.sparse_dimension), *params, limit],
            fetch=True,
        )
        return [p for p in self._rows_to_points(rows, lambda d: -d) if p.score > 0]

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
        raise CapabilityNotSupportedError("pgvector has no native hybrid fusion; use fusion.execution=client")

    async def fetch(self, collection: str, ids: Sequence[str], *, with_vectors: bool = False) -> list[ScoredPoint]:
        if not ids:
            return []
        vector_col = ", dense::text" if with_vectors else ""
        query = self.sql.SQL(f"SELECT id, payload, 0.0{vector_col} FROM {{}} WHERE id = ANY(%s::uuid[])").format(
            self._table(collection)
        )
        rows = await self._exec(collection, query, [list(ids)], fetch=True)
        return self._rows_to_points(rows, None, with_vectors)

    async def scroll(
        self,
        collection: str,
        *,
        filter_: FilterSpec | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[ScoredPoint], str | None]:
        where, params = self._filters.compile(filter_)
        if offset:
            where += " AND id >= %s::uuid"
            params.append(offset)
        query = self.sql.SQL(f"SELECT id, payload, 0.0 FROM {{}} WHERE {where} ORDER BY id LIMIT %s").format(
            self._table(collection)
        )
        rows = await self._exec(collection, query, [*params, limit + 1], fetch=True)
        points = self._rows_to_points(rows)
        next_offset = points[limit].id if len(points) > limit else None
        return points[:limit], next_offset

    async def health(self) -> bool:
        try:
            await self._exec(None, "SELECT 1", fetch=True)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()


@VECTOR_STORES.register(
    "pgvector", aliases=("postgres",), description="PostgreSQL + pgvector (HNSW, sparsevec, JSONPath filters)"
)
def _pgvector(*, config: VectorStoreConfig) -> PgVectorStore:
    return PgVectorStore(config.pgvector)
