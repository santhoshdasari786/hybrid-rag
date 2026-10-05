"""Composition root — the only place that knows which adapter implements which port.

``Container.start()`` performs the startup sequence (plugins → registry schema → recover
interrupted jobs → provision collections → warm up models) and ``Container.stop()`` tears
everything down in reverse order, draining in-flight jobs first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_service.adapters.jobs.runner import InProcessJobRunner
from rag_service.adapters.persistence.sql import (
    Database,
    SqlCollectionRepository,
    SqlJobRepository,
    SqlSourceRepository,
)
from rag_service.application.chat import ChatService
from rag_service.application.collections import CollectionService
from rag_service.application.evaluation import EvaluationService
from rag_service.application.ingestion import IngestionService
from rag_service.application.jobs import JobService
from rag_service.application.pipeline import PipelineFactory
from rag_service.application.retrieval import RetrievalService
from rag_service.application.sources import SourceService
from rag_service.core.config import Settings
from rag_service.core.logging import get_logger
from rag_service.core.registry import VECTOR_STORES, load_plugins
from rag_service.domain.ports import VectorStore

log = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    db: Database
    store: VectorStore
    factory: PipelineFactory
    runner: InProcessJobRunner
    collections: CollectionService
    ingestion: IngestionService
    retrieval: RetrievalService
    chat: ChatService
    jobs: JobService
    sources: SourceService
    evaluation: EvaluationService
    ready: bool = field(default=False)

    @classmethod
    def build(cls, settings: Settings) -> Container:
        load_plugins(settings.plugins)
        db = Database(settings.registry)
        collection_repo = SqlCollectionRepository(db)
        source_repo = SqlSourceRepository(db)
        job_repo = SqlJobRepository(db)
        store: VectorStore = VECTOR_STORES.create(settings.vector_store.provider, config=settings.vector_store)
        factory = PipelineFactory(settings)
        runner = InProcessJobRunner(settings.ingestion.worker_concurrency)

        collections = CollectionService(settings, factory, collection_repo, source_repo, store)
        ingestion = IngestionService(settings, factory, collections, source_repo, store)
        retrieval = RetrievalService(factory, collections, store)
        return cls(
            settings=settings,
            db=db,
            store=store,
            factory=factory,
            runner=runner,
            collections=collections,
            ingestion=ingestion,
            retrieval=retrieval,
            chat=ChatService(factory, retrieval),
            jobs=JobService(settings, job_repo, runner, ingestion, collections),
            sources=SourceService(collections, source_repo, store),
            evaluation=EvaluationService(retrieval),
        )

    async def start(self) -> None:
        if self.settings.registry.auto_create_schema:
            await self.db.create_schema()
        interrupted = await self.jobs.jobs.fail_interrupted()
        if interrupted:
            log.warning("jobs.recovered_interrupted", count=interrupted)
        await self.collections.bootstrap()
        # Build LLM clients now: a missing SDK, CLI or model name fails startup instead of the first chat.
        self.factory.llm()
        self.factory.utility_llm()
        if self.settings.app.warmup_models:
            await self.factory.warmup()
        self.ready = True
        log.info(
            "service.started",
            environment=self.settings.app.environment.value,
            vector_store=self.settings.vector_store.provider,
            llm=self.settings.llm.provider,
        )

    async def stop(self) -> None:
        self.ready = False
        await self.runner.shutdown(self.settings.server.graceful_shutdown_s)
        await self.jobs.close()
        await self.factory.close()
        await self.store.close()
        await self.db.dispose()
        log.info("service.stopped")

    async def health(self) -> dict[str, bool]:
        return {"registry": await self.db.health(), "vector_store": await self.store.health()}
