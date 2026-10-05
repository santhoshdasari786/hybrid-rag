"""Ingestion and source synchronisation.

The client-supplied ``source_id`` is the stable *processing ID* of a document. Re-sending the same
``source_id`` synchronises the index with the new content:

====================================  ======================================================
situation                             effect
====================================  ======================================================
new source_id                         parse → chunk → enrich → embed → upsert  (``created``)
same content, same metadata/config    nothing is touched                     (``unchanged``)
same content, new metadata/title      payloads patched, **no re-embedding**  (``metadata_updated``)
content or chunking/embedding changed re-chunk; embed only chunks whose hash is new; patch
                                      moved chunks; delete chunks that disappeared (``updated``)
delete                                every point of the source + registry rows (``deleted``)
====================================  ======================================================

Chunk IDs are ``uuid5(collection/source_id/chunk_hash/occurrence)``, so re-processing is idempotent
and an edit to one paragraph re-embeds one or two chunks, not the whole document.

Write ordering is *upsert new → save registry → delete stale*. A crash in between can leave a
harmless duplicate that the next sync removes, never a hole in the index.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from rag_service.adapters.jobs.runner import KeyedLocks
from rag_service.application.collections import CollectionService
from rag_service.application.pipeline import CollectionPipeline, PipelineFactory, ingest_fingerprint
from rag_service.core.config import ChunkingConfig, Settings
from rag_service.core.errors import (
    ChunkNotFoundError,
    PayloadTooLargeError,
    RAGError,
    SourceNotFoundError,
    UnsupportedMediaTypeError,
    ValidationFailedError,
)
from rag_service.core.logging import get_logger
from rag_service.core.registry import PARSERS
from rag_service.core.utils import (
    batched,
    canonical_json,
    chunk_id,
    deep_merge,
    hash_json,
    normalize_text,
    sha256_text,
)
from rag_service.domain.filters import FilterSpec
from rag_service.domain.models import (
    Chunk,
    ChunkDraft,
    ChunkRef,
    IngestAction,
    ParsedDocument,
    PayloadKeys,
    PayloadPatch,
    Section,
    SourceRecord,
    SourceStatus,
    SparseVector,
    VectorRecord,
)
from rag_service.domain.ports import SourceRepository, VectorWriter

log = get_logger(__name__)


@dataclass(slots=True)
class ChunkInput:
    """Client-provided chunk (bring-your-own-chunking) with its own metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentInput:
    source_id: str
    mime_type: str = "text/plain"
    content: str | None = None
    content_base64: str | None = None
    chunks: list[ChunkInput] | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunking: dict[str, Any] | None = None
    filename: str | None = None
    stored_sections: list[Section] | None = None
    """Internal: already-normalized sections re-hydrated from the registry (reindex)."""


@dataclass(slots=True)
class SourceResult:
    source_id: str
    action: IngestAction
    version: int | None = None
    chunks_total: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    chunks_unchanged: int = 0
    chunks_updated: int = 0
    error: str | None = None
    error_code: str | None = None
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "action": self.action.value,
            "version": self.version,
            "chunks_total": self.chunks_total,
            "chunks_added": self.chunks_added,
            "chunks_removed": self.chunks_removed,
            "chunks_unchanged": self.chunks_unchanged,
            "chunks_updated": self.chunks_updated,
            "error": self.error,
            "error_code": self.error_code,
            "duration_ms": self.duration_ms,
        }


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        factory: PipelineFactory,
        collections: CollectionService,
        sources: SourceRepository,
        writer: VectorWriter,
    ) -> None:
        self.settings = settings
        self.config = settings.ingestion
        self.factory = factory
        self.collections = collections
        self.sources = sources
        self.writer = writer
        self._locks = KeyedLocks()
        self._embed_slots = asyncio.Semaphore(self.config.embed_concurrency)

    # ================================================================== batch entry points

    async def ingest(
        self,
        collection: str,
        documents: Sequence[DocumentInput],
        *,
        force: bool = False,
        job_id: str | None = None,
        on_result: Any = None,
    ) -> list[SourceResult]:
        pipeline = await self.collections.pipeline(collection)
        semaphore = asyncio.Semaphore(self.config.worker_concurrency)

        async def one(doc: DocumentInput) -> SourceResult:
            async with semaphore:
                result = await self.sync_source(pipeline, doc, force=force, job_id=job_id)
            if on_result is not None:
                await on_result(result)
            return result

        return list(await asyncio.gather(*(one(d) for d in documents)))

    async def source_ids(self, collection: str) -> list[str]:
        return await self.sources.iter_ids(collection)

    async def delete_many(
        self, collection: str, source_ids: Sequence[str], *, on_result: Any = None
    ) -> list[SourceResult]:
        await self.collections.pipeline(collection)  # validates existence / staleness
        results = []
        for source_id in source_ids:
            result = await self.delete_source(collection, source_id)
            if on_result is not None:
                await on_result(result)
            results.append(result)
        return results

    # ================================================================== single source sync

    async def sync_source(
        self,
        pipeline: CollectionPipeline,
        doc: DocumentInput,
        *,
        force: bool = False,
        job_id: str | None = None,
    ) -> SourceResult:
        started = time.perf_counter()
        async with self._locks.hold(f"{pipeline.name}/{doc.source_id}"):
            existing = await self.sources.get(pipeline.name, doc.source_id)
            try:
                result = await self._sync(pipeline, doc, existing, force=force, job_id=job_id)
            except RAGError as exc:
                await self._record_failure(pipeline.name, doc, existing, exc.detail, job_id)
                result = SourceResult(doc.source_id, IngestAction.FAILED, error=exc.detail, error_code=exc.code)
            except Exception as exc:
                log.exception("ingest.source_failed", collection=pipeline.name, source_id=doc.source_id)
                await self._record_failure(pipeline.name, doc, existing, f"Unexpected error: {exc}", job_id)
                result = SourceResult(
                    doc.source_id,
                    IngestAction.FAILED,
                    error="Unexpected error while ingesting",
                    error_code="INTERNAL_ERROR",
                )
        result.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        log.info(
            "ingest.source",
            collection=pipeline.name,
            source_id=doc.source_id,
            action=result.action.value,
            added=result.chunks_added,
            removed=result.chunks_removed,
            unchanged=result.chunks_unchanged,
            duration_ms=result.duration_ms,
        )
        return result

    async def _sync(
        self,
        pipeline: CollectionPipeline,
        doc: DocumentInput,
        existing: SourceRecord | None,
        *,
        force: bool,
        job_id: str | None,
    ) -> SourceResult:
        prechunked = doc.chunks is not None
        chunking = self._chunking_config(pipeline, doc.chunking)
        parsed = self._parse(doc)
        title = doc.title or parsed.title
        content_hash = self._content_hash(parsed, doc)
        metadata_hash = hash_json({"metadata": doc.metadata, "title": title, "mime_type": doc.mime_type})
        fingerprint = ingest_fingerprint(pipeline, chunking, prechunked=prechunked)

        unchanged_content = (
            existing is not None
            and existing.status == SourceStatus.ACTIVE
            and not force
            and existing.content_hash == content_hash
            and existing.pipeline_fingerprint == fingerprint
        )
        if unchanged_content and existing is not None:
            if existing.metadata_hash == metadata_hash:
                return SourceResult(
                    doc.source_id,
                    IngestAction.UNCHANGED,
                    version=existing.version,
                    chunks_total=existing.chunk_count,
                    chunks_unchanged=existing.chunk_count,
                )
            return await self._update_metadata_only(
                pipeline.name, existing, doc.metadata, title, doc.mime_type, metadata_hash, job_id
            )

        # ---- chunk + identity
        drafts = (
            self._prechunked_drafts(doc)
            if prechunked
            else await self.factory.chunker(chunking, pipeline.dense).split(parsed)
        )
        if not drafts:
            raise ValidationFailedError("The document contains no extractable text")
        if len(drafts) > self.config.max_chunks_per_document:
            raise ValidationFailedError(
                f"The document produced {len(drafts)} chunks (limit {self.config.max_chunks_per_document}); "
                "increase chunk_size or split the document"
            )
        chunks = self._identify(pipeline, doc.source_id, drafts, title, fingerprint)

        # ---- diff
        old_refs = {r.chunk_id: r for r in await self.sources.chunks(pipeline.name, doc.source_id)} if existing else {}
        if not old_refs:
            # Safety net: points may exist without registry rows (failed first attempt, registry reset).
            await self.writer.delete_by_filter(pipeline.name, FilterSpec.for_source(doc.source_id))
        new_ids = {c.id for c in chunks}
        if force:
            # Forced: re-embed everything (e.g. the index was recreated); upserts overwrite in place.
            to_add, to_keep = list(chunks), []
        else:
            to_add = [c for c in chunks if c.id not in old_refs]
            to_keep = [c for c in chunks if c.id in old_refs]
        to_remove = [cid for cid in old_refs if cid not in new_ids]

        # ---- enrich + embed only what is new
        enricher = self.factory.llm_context_enricher(pipeline.config)
        if enricher is not None and to_add:
            await enricher.enrich(parsed.full_text, to_add)
        base_payload = {PayloadKeys.TITLE: title, PayloadKeys.MIME_TYPE: doc.mime_type}
        records = await self._embed(pipeline, to_add, doc, base_payload)
        if records:
            await self.writer.upsert(pipeline.name, records)

        # ---- patch kept chunks whose position or metadata changed
        metadata_changed = existing is None or existing.metadata_hash != metadata_hash
        patches = []
        for c in to_keep:
            ref = old_refs[c.id]
            if metadata_changed or ref.chunk_index != c.index or ref.chunk_metadata != _json_safe(c.metadata):
                patches.append(
                    PayloadPatch(
                        id=c.id,
                        payload={
                            **base_payload,
                            PayloadKeys.CHUNK_INDEX: c.index,
                            PayloadKeys.METADATA: {**doc.metadata, **c.metadata, **ref.metadata_overrides},
                        },
                    )
                )
        if patches:
            await self.writer.update_payloads(pipeline.name, patches)

        # ---- registry, then delete stale
        refs = [
            ChunkRef(
                chunk_id=c.id,
                chunk_hash=c.hash,
                chunk_index=c.index,
                chunk_metadata=_json_safe(c.metadata),
                metadata_overrides=old_refs[c.id].metadata_overrides if c.id in old_refs else {},
            )
            for c in chunks
        ]
        record = SourceRecord(
            collection=pipeline.name,
            source_id=doc.source_id,
            content_hash=content_hash,
            metadata_hash=metadata_hash,
            pipeline_fingerprint=fingerprint,
            version=(existing.version + 1) if existing else 1,
            status=SourceStatus.ACTIVE,
            title=title,
            mime_type=doc.mime_type,
            metadata=doc.metadata,
            chunk_count=len(chunks),
            content=self._stored_content(parsed, doc) if self.settings.registry.store_content else None,
            chunking_override=doc.chunking,
            prechunked=prechunked,
            last_job_id=job_id,
        )
        await self.sources.save(record, refs, expected_version=existing.version if existing else None)
        if to_remove:
            await self.writer.delete_ids(pipeline.name, to_remove)

        return SourceResult(
            doc.source_id,
            IngestAction.UPDATED if existing and existing.chunk_count else IngestAction.CREATED,
            version=record.version,
            chunks_total=len(chunks),
            chunks_added=len(to_add),
            chunks_removed=len(to_remove),
            chunks_unchanged=len(to_keep) - len(patches),
            chunks_updated=len(patches),
        )

    # ================================================================== metadata-only paths

    async def _update_metadata_only(
        self,
        collection: str,
        existing: SourceRecord,
        metadata: dict[str, Any],
        title: str | None,
        mime_type: str,
        metadata_hash: str,
        job_id: str | None,
    ) -> SourceResult:
        refs = await self.sources.chunks(collection, existing.source_id)
        patches = [
            PayloadPatch(
                id=ref.chunk_id,
                payload={
                    PayloadKeys.TITLE: title,
                    PayloadKeys.MIME_TYPE: mime_type,
                    PayloadKeys.METADATA: {**metadata, **ref.chunk_metadata, **ref.metadata_overrides},
                },
            )
            for ref in refs
        ]
        if patches:
            await self.writer.update_payloads(collection, patches)
        record = replace(
            existing,
            metadata=metadata,
            metadata_hash=metadata_hash,
            title=title,
            mime_type=mime_type,
            version=existing.version + 1,
            last_job_id=job_id,
            error=None,
        )
        await self.sources.save(record, None, expected_version=existing.version, update_content=False)
        return SourceResult(
            existing.source_id,
            IngestAction.METADATA_UPDATED,
            version=record.version,
            chunks_total=len(refs),
            chunks_updated=len(patches),
        )

    async def patch_source_metadata(
        self, collection: str, source_id: str, metadata: dict[str, Any], *, merge: bool = True
    ) -> SourceResult:
        """Update document-level metadata on every chunk of a source without re-embedding."""
        await self.collections.pipeline(collection)
        async with self._locks.hold(f"{collection}/{source_id}"):
            existing = await self.sources.get(collection, source_id)
            if existing is None or existing.status != SourceStatus.ACTIVE:
                raise SourceNotFoundError(f"Source '{source_id}' not found in collection '{collection}'")
            new_meta = deep_merge(existing.metadata, metadata) if merge else metadata
            metadata_hash = hash_json({"metadata": new_meta, "title": existing.title, "mime_type": existing.mime_type})
            return await self._update_metadata_only(
                collection, existing, new_meta, existing.title, existing.mime_type, metadata_hash, None
            )

    async def patch_chunk_metadata(
        self, collection: str, chunk_id_: str, metadata: dict[str, Any], *, merge: bool = True
    ) -> dict[str, Any]:
        """Attach metadata to a single chunk. Survives re-syncs as long as the chunk's content is unchanged."""
        await self.collections.pipeline(collection)
        found = await self.sources.find_chunk(collection, chunk_id_)
        if found is None:
            raise ChunkNotFoundError(f"Chunk '{chunk_id_}' not found in collection '{collection}'")
        source_id = found[0]
        async with self._locks.hold(f"{collection}/{source_id}"):
            found = await self.sources.find_chunk(collection, chunk_id_)  # re-read under the lock
            if found is None:
                raise ChunkNotFoundError(f"Chunk '{chunk_id_}' not found in collection '{collection}'")
            _, ref = found
            overrides = deep_merge(ref.metadata_overrides, metadata) if merge else metadata
            source = await self.sources.get(collection, source_id)
            doc_meta = source.metadata if source else {}
            await self.sources.set_chunk_overrides(collection, chunk_id_, overrides)
            merged = {**doc_meta, **ref.chunk_metadata, **overrides}
            await self.writer.update_payloads(
                collection, [PayloadPatch(id=chunk_id_, payload={PayloadKeys.METADATA: merged})]
            )
            return merged

    # ================================================================== delete / reindex

    async def delete_source(self, collection: str, source_id: str) -> SourceResult:
        started = time.perf_counter()
        async with self._locks.hold(f"{collection}/{source_id}"):
            existing = await self.sources.get(collection, source_id)
            await self.writer.delete_by_filter(collection, FilterSpec.for_source(source_id))
            deleted = await self.sources.delete(collection, source_id)
        action = IngestAction.DELETED if deleted else IngestAction.NOT_FOUND
        log.info("ingest.source_deleted", collection=collection, source_id=source_id, found=deleted)
        return SourceResult(
            source_id,
            action,
            version=existing.version if existing else None,
            chunks_removed=existing.chunk_count if existing else 0,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def reindex(
        self,
        collection: str,
        *,
        source_ids: Sequence[str] | None = None,
        force: bool = False,
        on_result: Any = None,
        job_id: str | None = None,
    ) -> list[SourceResult]:
        """Re-process sources from their stored content with the collection's current configuration."""
        pipeline = await self.collections.pipeline(collection)
        ids = list(source_ids) if source_ids else await self.sources.iter_ids(collection)
        results: list[SourceResult] = []
        semaphore = asyncio.Semaphore(self.config.worker_concurrency)

        async def one(source_id: str) -> None:
            async with semaphore:
                record = await self.sources.get(collection, source_id, with_content=True)
                if record is None:
                    result = SourceResult(source_id, IngestAction.NOT_FOUND)
                elif record.content is None:
                    result = SourceResult(
                        source_id,
                        IngestAction.FAILED,
                        error="No stored content (registry.store_content was disabled); re-send the document",
                        error_code="CONTENT_NOT_STORED",
                    )
                else:
                    result = await self.sync_source(
                        pipeline, self._document_from_record(record), force=force, job_id=job_id
                    )
                results.append(result)
                if on_result is not None:
                    await on_result(result)

        await asyncio.gather(*(one(s) for s in ids))
        return results

    # ================================================================== helpers

    def _chunking_config(self, pipeline: CollectionPipeline, override: dict[str, Any] | None) -> ChunkingConfig:
        if not override:
            return pipeline.config.chunking
        try:
            return ChunkingConfig.model_validate(deep_merge(pipeline.config.chunking.model_dump(), override))
        except Exception as exc:
            raise ValidationFailedError(f"Invalid chunking override: {exc}") from exc

    def _parse(self, doc: DocumentInput) -> ParsedDocument:
        if doc.stored_sections is not None:
            return ParsedDocument(sections=list(doc.stored_sections), title=doc.title)
        if doc.chunks is not None:
            text = "\n\n".join(c.text for c in doc.chunks)
            return ParsedDocument(sections=[Section(text=self._normalize(text))], title=None)
        if doc.mime_type not in self.config.allowed_mime_types:
            raise UnsupportedMediaTypeError(
                f"MIME type '{doc.mime_type}' is not allowed. Allowed: {', '.join(self.config.allowed_mime_types)}"
            )
        if doc.mime_type not in PARSERS:
            raise UnsupportedMediaTypeError(f"No parser registered for '{doc.mime_type}'")
        if doc.content_base64 is not None:
            try:
                data = base64.b64decode(doc.content_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValidationFailedError(f"content_base64 is not valid base64: {exc}") from exc
        else:
            data = (doc.content or "").encode("utf-8")
        if len(data) > self.config.max_document_bytes:
            raise PayloadTooLargeError(f"Document exceeds {self.config.max_document_bytes} bytes")
        parsed: ParsedDocument = PARSERS.create(doc.mime_type).parse(data, filename=doc.filename)
        parsed.sections = [Section(text=self._normalize(s.text), metadata=s.metadata) for s in parsed.sections]
        parsed.sections = [s for s in parsed.sections if s.text]
        return parsed

    def _normalize(self, text: str) -> str:
        return normalize_text(
            text,
            unicode_nfkc=self.config.normalize_unicode,
            collapse_whitespace=self.config.collapse_whitespace,
        )

    def _prechunked_drafts(self, doc: DocumentInput) -> list[ChunkDraft]:
        return [
            ChunkDraft(text=self._normalize(c.text), metadata=dict(c.metadata))
            for c in doc.chunks or []
            if c.text and c.text.strip()
        ]

    def _content_hash(self, parsed: ParsedDocument, doc: DocumentInput) -> str:
        if doc.chunks is not None:
            return sha256_text(canonical_json([[c.text, c.metadata] for c in doc.chunks]))
        return sha256_text(canonical_json([[s.text, s.metadata] for s in parsed.sections]))

    def _stored_content(self, parsed: ParsedDocument, doc: DocumentInput) -> str:
        if doc.chunks is not None:
            return canonical_json({"chunks": [{"text": c.text, "metadata": c.metadata} for c in doc.chunks]})
        return canonical_json(
            {
                "sections": [{"text": s.text, "metadata": s.metadata} for s in parsed.sections],
                "title": parsed.title,
            }
        )

    def _document_from_record(self, record: SourceRecord) -> DocumentInput:
        stored = json.loads(record.content or "{}")
        if record.prechunked:
            chunks = [ChunkInput(text=c["text"], metadata=c.get("metadata", {})) for c in stored.get("chunks", [])]
            return DocumentInput(
                source_id=record.source_id,
                mime_type=record.mime_type,
                chunks=chunks,
                title=record.title,
                metadata=record.metadata,
            )
        sections = [Section(text=sec["text"], metadata=sec.get("metadata", {})) for sec in stored.get("sections", [])]
        return DocumentInput(
            source_id=record.source_id,
            mime_type=record.mime_type,
            title=record.title,
            metadata=record.metadata,
            chunking=record.chunking_override,
            stored_sections=sections,
        )

    def _identify(
        self,
        pipeline: CollectionPipeline,
        source_id: str,
        drafts: list[ChunkDraft],
        title: str | None,
        fingerprint: str,
    ) -> list[Chunk]:
        header = self.factory.header_enricher(pipeline.config)
        seen: Counter[str] = Counter()
        chunks = []
        for index, draft in enumerate(drafts):
            embed_text = header.embed_text(
                draft.text, title=title, section=draft.metadata.get("section"), source_id=source_id
            )
            digest = sha256_text(f"{fingerprint}\x00{embed_text}\x00{draft.text}")
            occurrence = seen[digest]
            seen[digest] += 1
            chunks.append(
                Chunk(
                    id=chunk_id(pipeline.name, source_id, digest, occurrence),
                    source_id=source_id,
                    index=index,
                    text=draft.text,
                    embed_text=embed_text,
                    hash=digest,
                    metadata=dict(draft.metadata),
                )
            )
        return chunks

    async def _embed(
        self,
        pipeline: CollectionPipeline,
        chunks: list[Chunk],
        doc: DocumentInput,
        base_payload: dict[str, Any],
    ) -> list[VectorRecord]:
        records: list[VectorRecord] = []
        dense_cfg = pipeline.config.embedding.dense
        batch_size = dense_cfg.batch_size if dense_cfg else 64
        for batch in batched(chunks, batch_size):
            texts = [c.embed_text for c in batch]
            async with self._embed_slots:
                dense, sparse = await asyncio.gather(_dense_vectors(pipeline, texts), _sparse_vectors(pipeline, texts))
            for i, chunk in enumerate(batch):
                payload = {
                    **base_payload,
                    PayloadKeys.TEXT: chunk.text,
                    PayloadKeys.SOURCE_ID: chunk.source_id,
                    PayloadKeys.CHUNK_INDEX: chunk.index,
                    PayloadKeys.CHUNK_HASH: chunk.hash,
                    PayloadKeys.METADATA: {**doc.metadata, **chunk.metadata},
                }
                records.append(VectorRecord(id=chunk.id, payload=payload, dense=dense[i], sparse=sparse[i]))
        return records

    async def _record_failure(
        self,
        collection: str,
        doc: DocumentInput,
        existing: SourceRecord | None,
        error: str,
        job_id: str | None,
    ) -> None:
        try:
            if existing is not None:
                await self.sources.mark(
                    collection, doc.source_id, status=SourceStatus.FAILED.value, error=error, job_id=job_id
                )
            else:
                await self.sources.save(
                    SourceRecord(
                        collection=collection,
                        source_id=doc.source_id,
                        content_hash="",
                        metadata_hash="",
                        pipeline_fingerprint="",
                        version=0,
                        status=SourceStatus.FAILED,
                        title=doc.title,
                        mime_type=doc.mime_type,
                        metadata=doc.metadata,
                        error=error,
                        last_job_id=job_id,
                    ),
                    [],
                    expected_version=None,
                )
        except Exception as exc:  # never mask the original failure
            log.warning("ingest.failure_not_recorded", source_id=doc.source_id, error=str(exc))


async def _dense_vectors(pipeline: CollectionPipeline, texts: list[str]) -> list[list[float] | None]:
    if pipeline.dense is None:
        return [None] * len(texts)
    return list(await pipeline.dense.embed_documents(texts))


async def _sparse_vectors(pipeline: CollectionPipeline, texts: list[str]) -> list[SparseVector | None]:
    if pipeline.sparse is None:
        return [None] * len(texts)
    return list(await pipeline.sparse.embed_documents(texts))


def _json_safe(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(canonical_json(data))
    return result
