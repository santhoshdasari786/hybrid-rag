"""SQL source registry (SQLAlchemy 2 async). SQLite for development, PostgreSQL for production.

The registry is the service's system of record for *what was ingested*:

* ``rag_collections`` — logical collections, their pipeline overrides and embedding fingerprint.
* ``rag_sources``     — one row per (collection, source_id): content/metadata hashes, pipeline
  fingerprint, version (optimistic concurrency), status and — optionally — the normalized content
  so the collection can be re-indexed after a configuration change.
* ``rag_chunks``      — the chunk IDs/hashes of each source's current version. Diffing against them
  is what lets an update re-embed only changed chunks and delete stale ones.
* ``rag_jobs``        — asynchronous ingestion/deletion/reindex jobs and their per-source results.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    delete,
    event,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from rag_service.core.config import RegistryConfig
from rag_service.core.errors import ConcurrentModificationError
from rag_service.core.utils import utcnow
from rag_service.domain.models import (
    ChunkRef,
    CollectionRecord,
    JobRecord,
    JobStatus,
    JobType,
    SourceRecord,
    SourceStatus,
)

JsonType = JSON().with_variant(JSONB(), "postgresql")
PkType = BigInteger().with_variant(Integer(), "sqlite")


class Base(DeclarativeBase):
    pass


class CollectionRow(Base):
    __tablename__ = "rag_collections"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str] = mapped_column(Text, default="")
    pipeline_overrides: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    embedding_fingerprint: Mapped[str] = mapped_column(String(64), default="")
    payload_indexes: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, default=list)
    managed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class SourceRow(Base):
    __tablename__ = "rag_sources"
    __table_args__ = (
        UniqueConstraint("collection", "source_id", name="uq_rag_sources_collection_source"),
        Index("ix_rag_sources_collection_status", "collection", "status"),
    )

    id: Mapped[int] = mapped_column(PkType, primary_key=True, autoincrement=True)
    collection: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    mime_type: Mapped[str] = mapped_column(String(128), default="text/plain")
    content_hash: Mapped[str] = mapped_column(String(64))
    metadata_hash: Mapped[str] = mapped_column(String(64))
    pipeline_fingerprint: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default=SourceStatus.ACTIVE.value)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JsonType, default=dict)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunking_override: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    prechunked: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChunkRow(Base):
    __tablename__ = "rag_chunks"
    __table_args__ = (Index("ix_rag_chunks_source", "source_pk"),)

    chunk_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_pk: Mapped[int] = mapped_column(PkType, ForeignKey("rag_sources.id", ondelete="CASCADE"), nullable=False)
    collection: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String(64))
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    metadata_overrides: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)


class JobRow(Base):
    __tablename__ = "rag_jobs"
    __table_args__ = (Index("ix_rag_jobs_collection_created", "collection", "created_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    collection: Mapped[str] = mapped_column(String(64))
    type: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(24), index=True)
    total: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    succeeded: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    results: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    options: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    requested_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def _aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


# ============================================================================== database


class Database:
    """Engine + session management.

    SQLite (development) allows a single writer and, in ``:memory:`` mode, shares one connection,
    so sessions are serialised with an asyncio lock and WAL + busy timeout are enabled. PostgreSQL
    (production) uses a normal connection pool with full concurrency.
    """

    def __init__(self, config: RegistryConfig) -> None:
        self.config = config
        url = config.url
        self.is_sqlite = url.startswith("sqlite")
        kwargs: dict[str, Any] = {"echo": config.echo}
        if self.is_sqlite:
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
                kwargs["connect_args"] = {"check_same_thread": False}
            else:
                db_path = url.split("///", 1)[-1]
                if db_path:
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                kwargs["connect_args"] = {"timeout": 30}
        else:
            kwargs["pool_size"] = config.pool_size
            kwargs["pool_pre_ping"] = True
        self.engine: AsyncEngine = create_async_engine(url, **kwargs)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        self._lock = asyncio.Lock() if self.is_sqlite else None
        if self.is_sqlite:
            event.listen(self.engine.sync_engine, "connect", _sqlite_pragmas)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._lock is None:
            async with self.sessions() as session:
                yield session
            return
        async with self._lock, self.sessions() as session:
            yield session

    async def create_schema(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def health(self) -> bool:
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def dispose(self) -> None:
        await self.engine.dispose()


def _sqlite_pragmas(dbapi_connection: Any, _: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# ============================================================================== repositories


def _collection(row: CollectionRow) -> CollectionRecord:
    return CollectionRecord(
        name=row.name,
        description=row.description or "",
        pipeline_overrides=dict(row.pipeline_overrides or {}),
        embedding_fingerprint=row.embedding_fingerprint or "",
        managed=bool(row.managed),
        payload_indexes=list(row.payload_indexes or []),
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
    )


class SqlCollectionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def get(self, name: str) -> CollectionRecord | None:
        async with self.db.session() as session:
            row = await session.get(CollectionRow, name)
            return _collection(row) if row else None

    async def list_all(self) -> list[CollectionRecord]:
        async with self.db.session() as session:
            rows = (await session.execute(select(CollectionRow).order_by(CollectionRow.name))).scalars().all()
            return [_collection(r) for r in rows]

    async def upsert(self, record: CollectionRecord) -> CollectionRecord:
        async with self.db.session() as session, session.begin():
            row = await session.get(CollectionRow, record.name)
            if row is None:
                row = CollectionRow(name=record.name, created_at=utcnow())
                session.add(row)
            row.description = record.description
            row.pipeline_overrides = record.pipeline_overrides
            row.embedding_fingerprint = record.embedding_fingerprint
            row.payload_indexes = record.payload_indexes
            row.managed = record.managed
            row.updated_at = utcnow()
        return _collection(row)

    async def delete(self, name: str) -> bool:
        async with self.db.session() as session, session.begin():
            result = await session.execute(delete(CollectionRow).where(CollectionRow.name == name))
            return bool(result.rowcount)  # type: ignore[attr-defined]


def _source(row: SourceRow, with_content: bool) -> SourceRecord:
    return SourceRecord(
        collection=row.collection,
        source_id=row.source_id,
        content_hash=row.content_hash,
        metadata_hash=row.metadata_hash,
        pipeline_fingerprint=row.pipeline_fingerprint,
        version=row.version,
        status=SourceStatus(row.status),
        title=row.title,
        mime_type=row.mime_type,
        metadata=dict(row.meta or {}),
        chunk_count=row.chunk_count,
        content=row.content if with_content else None,
        chunking_override=row.chunking_override,
        prechunked=bool(row.prechunked),
        error=row.error,
        last_job_id=row.last_job_id,
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
    )


class SqlSourceRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def _row(self, session: AsyncSession, collection: str, source_id: str) -> SourceRow | None:
        stmt = select(SourceRow).where(SourceRow.collection == collection, SourceRow.source_id == source_id)
        return (await session.execute(stmt)).scalar_one_or_none()

    async def get(self, collection: str, source_id: str, *, with_content: bool = False) -> SourceRecord | None:
        async with self.db.session() as session:
            row = await self._row(session, collection, source_id)
            return _source(row, with_content) if row else None

    async def list_page(
        self, collection: str, *, limit: int, offset: int, status: str | None = None
    ) -> tuple[list[SourceRecord], int]:
        async with self.db.session() as session:
            base = select(SourceRow).where(SourceRow.collection == collection)
            if status:
                base = base.where(SourceRow.status == status)
            total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
            rows = (
                (await session.execute(base.order_by(SourceRow.source_id).limit(limit).offset(offset))).scalars().all()
            )
            return [_source(r, False) for r in rows], int(total)

    async def iter_ids(self, collection: str) -> list[str]:
        async with self.db.session() as session:
            stmt = select(SourceRow.source_id).where(SourceRow.collection == collection).order_by(SourceRow.source_id)
            return list((await session.execute(stmt)).scalars().all())

    async def chunks(self, collection: str, source_id: str) -> list[ChunkRef]:
        async with self.db.session() as session:
            stmt = (
                select(ChunkRow)
                .join(SourceRow, SourceRow.id == ChunkRow.source_pk)
                .where(SourceRow.collection == collection, SourceRow.source_id == source_id)
                .order_by(ChunkRow.chunk_index)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [
                ChunkRef(
                    chunk_id=r.chunk_id,
                    chunk_hash=r.chunk_hash,
                    chunk_index=r.chunk_index,
                    chunk_metadata=dict(r.chunk_metadata or {}),
                    metadata_overrides=dict(r.metadata_overrides or {}),
                )
                for r in rows
            ]

    async def save(
        self,
        record: SourceRecord,
        chunks: Sequence[ChunkRef] | None,
        *,
        expected_version: int | None,
        update_content: bool = True,
    ) -> SourceRecord:
        now = utcnow()
        values: dict[str, Any] = {
            "title": record.title,
            "mime_type": record.mime_type,
            "content_hash": record.content_hash,
            "metadata_hash": record.metadata_hash,
            "pipeline_fingerprint": record.pipeline_fingerprint,
            "version": record.version,
            "status": record.status.value,
            "chunk_count": record.chunk_count,
            "meta": record.metadata,
            "content": record.content,
            "chunking_override": record.chunking_override,
            "prechunked": record.prechunked,
            "error": record.error,
            "last_job_id": record.last_job_id,
            "updated_at": now,
        }
        if not update_content and expected_version is not None:
            values.pop("content")
        try:
            async with self.db.session() as session, session.begin():
                if expected_version is None:
                    row = SourceRow(collection=record.collection, source_id=record.source_id, created_at=now, **values)
                    session.add(row)
                    await session.flush()
                    pk = row.id
                else:
                    stmt = (
                        update(SourceRow)
                        .where(
                            SourceRow.collection == record.collection,
                            SourceRow.source_id == record.source_id,
                            SourceRow.version == expected_version,
                        )
                        .values(**values)
                        .returning(SourceRow.id)
                    )
                    pk_row = (await session.execute(stmt)).first()
                    if pk_row is None:
                        raise ConcurrentModificationError(
                            f"Source '{record.source_id}' was modified concurrently "
                            f"(expected version {expected_version})"
                        )
                    pk = pk_row[0]
                if chunks is not None:
                    await session.execute(delete(ChunkRow).where(ChunkRow.source_pk == pk))
                    session.add_all(
                        ChunkRow(
                            chunk_id=c.chunk_id,
                            source_pk=pk,
                            collection=record.collection,
                            chunk_hash=c.chunk_hash,
                            chunk_index=c.chunk_index,
                            chunk_metadata=c.chunk_metadata,
                            metadata_overrides=c.metadata_overrides,
                        )
                        for c in chunks
                    )
        except IntegrityError as exc:
            raise ConcurrentModificationError(f"Source '{record.source_id}' was created concurrently") from exc
        record.updated_at = now
        return record

    async def find_chunk(self, collection: str, chunk_id: str) -> tuple[str, ChunkRef] | None:
        async with self.db.session() as session:
            stmt = (
                select(SourceRow.source_id, ChunkRow)
                .join(SourceRow, SourceRow.id == ChunkRow.source_pk)
                .where(ChunkRow.chunk_id == chunk_id, ChunkRow.collection == collection)
            )
            row = (await session.execute(stmt)).first()
            if row is None:
                return None
            source_id, chunk = row
            return source_id, ChunkRef(
                chunk_id=chunk.chunk_id,
                chunk_hash=chunk.chunk_hash,
                chunk_index=chunk.chunk_index,
                chunk_metadata=dict(chunk.chunk_metadata or {}),
                metadata_overrides=dict(chunk.metadata_overrides or {}),
            )

    async def mark(
        self,
        collection: str,
        source_id: str,
        *,
        status: str,
        error: str | None = None,
        job_id: str | None = None,
    ) -> None:
        async with self.db.session() as session, session.begin():
            values: dict[str, Any] = {"status": status, "error": error, "updated_at": utcnow()}
            if job_id:
                values["last_job_id"] = job_id
            await session.execute(
                update(SourceRow)
                .where(SourceRow.collection == collection, SourceRow.source_id == source_id)
                .values(**values)
            )

    async def set_chunk_overrides(self, collection: str, chunk_id: str, overrides: dict[str, Any]) -> ChunkRef | None:
        async with self.db.session() as session, session.begin():
            row = await session.get(ChunkRow, chunk_id)
            if row is None or row.collection != collection:
                return None
            row.metadata_overrides = overrides
            return ChunkRef(
                chunk_id=row.chunk_id,
                chunk_hash=row.chunk_hash,
                chunk_index=row.chunk_index,
                chunk_metadata=dict(row.chunk_metadata or {}),
                metadata_overrides=dict(overrides),
            )

    async def delete(self, collection: str, source_id: str) -> bool:
        async with self.db.session() as session, session.begin():
            row = await self._row(session, collection, source_id)
            if row is None:
                return False
            await session.execute(delete(ChunkRow).where(ChunkRow.source_pk == row.id))
            await session.delete(row)
            return True

    async def delete_collection(self, collection: str) -> int:
        async with self.db.session() as session, session.begin():
            await session.execute(delete(ChunkRow).where(ChunkRow.collection == collection))
            result = await session.execute(delete(SourceRow).where(SourceRow.collection == collection))
            return int(result.rowcount or 0)  # type: ignore[attr-defined]

    async def stats(self, collection: str) -> dict[str, int]:
        async with self.db.session() as session:
            stmt = (
                select(SourceRow.status, func.count(), func.coalesce(func.sum(SourceRow.chunk_count), 0))
                .where(SourceRow.collection == collection)
                .group_by(SourceRow.status)
            )
            result = {"sources": 0, "chunks": 0, "failed_sources": 0}
            for status, count, chunks in (await session.execute(stmt)).all():
                result["sources"] += int(count)
                if status == SourceStatus.FAILED.value:
                    result["failed_sources"] += int(count)
                else:
                    result["chunks"] += int(chunks)
            return result


def _job(row: JobRow) -> JobRecord:
    return JobRecord(
        id=row.id,
        collection=row.collection,
        type=JobType(row.type),
        status=JobStatus(row.status),
        total=row.total,
        processed=row.processed,
        succeeded=row.succeeded,
        failed=row.failed,
        results=list(row.results or []),
        error=row.error,
        options=dict(row.options or {}),
        requested_by=row.requested_by,
        created_at=_aware(row.created_at),
        started_at=_aware(row.started_at),
        finished_at=_aware(row.finished_at),
    )


class SqlJobRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def create(self, job: JobRecord) -> JobRecord:
        job.created_at = job.created_at or utcnow()
        async with self.db.session() as session, session.begin():
            session.add(
                JobRow(
                    id=job.id,
                    collection=job.collection,
                    type=job.type.value,
                    status=job.status.value,
                    total=job.total,
                    processed=job.processed,
                    succeeded=job.succeeded,
                    failed=job.failed,
                    results=job.results,
                    error=job.error,
                    options=job.options,
                    requested_by=job.requested_by,
                    created_at=job.created_at,
                )
            )
        return job

    async def get(self, job_id: str) -> JobRecord | None:
        async with self.db.session() as session:
            row = await session.get(JobRow, job_id)
            return _job(row) if row else None

    async def update(self, job: JobRecord) -> None:
        async with self.db.session() as session, session.begin():
            await session.execute(
                update(JobRow)
                .where(JobRow.id == job.id)
                .values(
                    status=job.status.value,
                    total=job.total,
                    processed=job.processed,
                    succeeded=job.succeeded,
                    failed=job.failed,
                    results=job.results,
                    error=job.error,
                    started_at=job.started_at,
                    finished_at=job.finished_at,
                )
            )

    async def list_page(
        self, *, collection: str | None, status: str | None, limit: int, offset: int
    ) -> tuple[list[JobRecord], int]:
        async with self.db.session() as session:
            base = select(JobRow)
            if collection:
                base = base.where(JobRow.collection == collection)
            if status:
                base = base.where(JobRow.status == status)
            total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
            rows = (
                (await session.execute(base.order_by(JobRow.created_at.desc()).limit(limit).offset(offset)))
                .scalars()
                .all()
            )
            return [_job(r) for r in rows], int(total)

    async def fail_interrupted(self) -> int:
        async with self.db.session() as session, session.begin():
            result = await session.execute(
                update(JobRow)
                .where(JobRow.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]))
                .values(
                    status=JobStatus.FAILED.value,
                    error="Interrupted by service restart",
                    finished_at=utcnow(),
                )
            )
            return int(result.rowcount or 0)  # type: ignore[attr-defined]
