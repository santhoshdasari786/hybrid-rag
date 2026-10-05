"""In-process background job runner and per-key async locks.

The runner bounds concurrency with a semaphore, tracks tasks so shutdown can drain them, and never
lets a task exception escape unobserved. For multi-replica deployments that need work stealing or
retries across restarts, implement the ``JobRunner`` port on top of a queue (Redis/Arq, Celery,
SQS) — the services do not change.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from rag_service.core.logging import get_logger

log = get_logger(__name__)


class InProcessJobRunner:
    def __init__(self, concurrency: int) -> None:
        self._semaphore = asyncio.Semaphore(concurrency)
        self._tasks: set[asyncio.Task[None]] = set()
        self._closing = False

    @property
    def active(self) -> int:
        return len(self._tasks)

    async def submit(self, job_id: str, work: Callable[[], Awaitable[None]]) -> None:
        if self._closing:
            raise RuntimeError("job runner is shutting down")

        async def run() -> None:
            async with self._semaphore:
                try:
                    await work()
                except asyncio.CancelledError:
                    log.warning("job.cancelled", job_id=job_id)
                    raise
                except Exception:
                    log.exception("job.crashed", job_id=job_id)

        # Copy the request context so the job's log lines keep the request_id.
        task = asyncio.create_task(run(), name=f"job:{job_id}", context=contextvars.copy_context())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def shutdown(self, timeout_s: float) -> None:
        self._closing = True
        if not self._tasks:
            return
        log.info("job_runner.draining", active=len(self._tasks), timeout_s=timeout_s)
        _, pending = await asyncio.wait(set(self._tasks), timeout=timeout_s)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            log.warning("job_runner.cancelled_pending", count=len(pending))


class KeyedLocks:
    """One asyncio.Lock per key, garbage-collected when unused (serialises work per source)."""

    def __init__(self) -> None:
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        lock, refs = self._locks.get(key, (asyncio.Lock(), 0))
        self._locks[key] = (lock, refs + 1)
        try:
            async with lock:
                yield
        finally:
            lock, refs = self._locks[key]
            if refs <= 1:
                del self._locks[key]
            else:
                self._locks[key] = (lock, refs - 1)
