"""Job orchestration for ingestion, deletion and re-indexing.

Every write request produces a durable :class:`JobRecord`, whether it runs synchronously (small
batches, result in the HTTP response) or asynchronously (``202 Accepted`` + ``GET /jobs/{id}``).
Progress and per-source results are persisted as sources complete, and an optional callback URL
(host allow-listed) is notified when the job finishes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from urllib.parse import urlparse

import httpx

from rag_service.application.collections import CollectionService
from rag_service.application.ingestion import DocumentInput, IngestionService, SourceResult
from rag_service.core.config import Settings
from rag_service.core.errors import JobNotFoundError, ValidationFailedError
from rag_service.core.logging import get_logger
from rag_service.core.utils import new_id, utcnow
from rag_service.domain.models import IngestAction, JobRecord, JobStatus, JobType
from rag_service.domain.ports import JobRepository, JobRunner

log = get_logger(__name__)


class JobService:
    def __init__(
        self,
        settings: Settings,
        jobs: JobRepository,
        runner: JobRunner,
        ingestion: IngestionService,
        collections: CollectionService,
    ) -> None:
        self.settings = settings
        self.jobs = jobs
        self.runner = runner
        self.ingestion = ingestion
        self.collections = collections
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ public commands

    async def ingest(
        self,
        collection: str,
        documents: Sequence[DocumentInput],
        *,
        mode: str,
        force: bool,
        requested_by: str | None,
        callback_url: str | None = None,
    ) -> JobRecord:
        await self.collections.pipeline(collection)  # fail fast on unknown / stale collections

        async def work(on_result: Callable[[SourceResult], Awaitable[None]], job_id: str) -> list[SourceResult]:
            return await self.ingestion.ingest(collection, documents, force=force, job_id=job_id, on_result=on_result)

        return await self._run(
            collection,
            JobType.INGEST,
            len(documents),
            work,
            mode=mode,
            requested_by=requested_by,
            options={"force": force, "mode": mode, "callback_url": callback_url},
        )

    async def delete(
        self, collection: str, source_ids: Sequence[str], *, mode: str, requested_by: str | None
    ) -> JobRecord:
        async def work(on_result: Callable[[SourceResult], Awaitable[None]], job_id: str) -> list[SourceResult]:
            return await self.ingestion.delete_many(collection, source_ids, on_result=on_result)

        return await self._run(
            collection,
            JobType.DELETE,
            len(source_ids),
            work,
            mode=mode,
            requested_by=requested_by,
            options={"mode": mode},
        )

    async def reindex(
        self,
        collection: str,
        *,
        source_ids: Sequence[str] | None,
        force: bool,
        recreate: bool,
        requested_by: str | None,
    ) -> JobRecord:
        if recreate:
            await self.collections.recreate_index(collection)
            force = True  # the index is empty: every source must be embedded again
        ids = list(source_ids) if source_ids else await self.ingestion.source_ids(collection)

        async def work(on_result: Callable[[SourceResult], Awaitable[None]], job_id: str) -> list[SourceResult]:
            return await self.ingestion.reindex(
                collection, source_ids=ids, force=force, on_result=on_result, job_id=job_id
            )

        return await self._run(
            collection,
            JobType.REINDEX,
            len(ids),
            work,
            mode="async",
            requested_by=requested_by,
            options={"force": force, "recreate": recreate},
        )

    async def get(self, job_id: str) -> JobRecord:
        job = await self.jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Job '{job_id}' not found")
        return job

    async def list_page(
        self, *, collection: str | None, status: str | None, limit: int, offset: int
    ) -> tuple[list[JobRecord], int]:
        return await self.jobs.list_page(collection=collection, status=status, limit=limit, offset=offset)

    # ------------------------------------------------------------------ execution

    async def _run(
        self,
        collection: str,
        type_: JobType,
        total: int,
        work: Callable[..., Awaitable[list[SourceResult]]],
        *,
        mode: str,
        requested_by: str | None,
        options: dict[str, Any],
    ) -> JobRecord:
        callback_url = options.get("callback_url")
        if callback_url:
            self._validate_callback(str(callback_url))
        job = await self.jobs.create(
            JobRecord(
                id=new_id("job"),
                collection=collection,
                type=type_,
                total=total,
                requested_by=requested_by,
                options=options,
            )
        )
        if mode == "sync":
            await self._execute(job, work)
            return job
        await self.runner.submit(job.id, lambda: self._execute(job, work))
        return job

    async def _execute(self, job: JobRecord, work: Callable[..., Awaitable[list[SourceResult]]]) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = utcnow()
        await self.jobs.update(job)
        lock = asyncio.Lock()

        async def on_result(result: SourceResult) -> None:
            async with lock:
                job.processed += 1
                if result.action == IngestAction.FAILED:
                    job.failed += 1
                else:
                    job.succeeded += 1
                job.results.append(result.as_dict())
                # Persist progress periodically (every 10 sources) to bound write amplification.
                if job.processed % 10 == 0:
                    await self.jobs.update(job)

        try:
            await work(on_result, job.id)
            if job.failed and job.succeeded:
                job.status = JobStatus.PARTIALLY_SUCCEEDED
            elif job.failed:
                job.status = JobStatus.FAILED
                job.error = "All sources failed; see results for details"
            else:
                job.status = JobStatus.SUCCEEDED
        except Exception as exc:
            log.exception("job.failed", job_id=job.id)
            job.status = JobStatus.FAILED
            job.error = getattr(exc, "detail", None) or str(exc)
        job.finished_at = utcnow()
        await self.jobs.update(job)
        log.info("job.finished", job_id=job.id, status=job.status.value, succeeded=job.succeeded, failed=job.failed)
        if job.options.get("callback_url"):
            await self._notify(job)

    # ------------------------------------------------------------------ callbacks

    def _validate_callback(self, url: str) -> None:
        cfg = self.settings.ingestion.callbacks
        if not cfg.enabled:
            raise ValidationFailedError("callback_url is not enabled on this service (ingestion.callbacks.enabled)")
        parsed = urlparse(url)
        if parsed.scheme not in ("https", "http") or not parsed.hostname:
            raise ValidationFailedError("callback_url must be an absolute http(s) URL")
        if parsed.hostname not in cfg.allowed_hosts:
            raise ValidationFailedError(
                f"callback host '{parsed.hostname}' is not in ingestion.callbacks.allowed_hosts"
            )

    async def _notify(self, job: JobRecord) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.settings.ingestion.callbacks.timeout_s, follow_redirects=False)
        body = {
            "job_id": job.id,
            "collection": job.collection,
            "type": job.type.value,
            "status": job.status.value,
            "succeeded": job.succeeded,
            "failed": job.failed,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        }
        try:
            await self._http.post(str(job.options["callback_url"]), json=body)
        except httpx.HTTPError as exc:
            log.warning("job.callback_failed", job_id=job.id, error=str(exc))

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
