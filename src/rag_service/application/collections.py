"""Collection lifecycle: bootstrap from configuration, create/update/delete via API, staleness checks.

A *collection* is the unit of isolation: its own vector index, pipeline configuration and sources.
Collections declared in the configuration file are **managed** (configuration is the source of
truth); collections created through the API are **dynamic** and persisted in the registry.

A collection is *stale* when the embedding fingerprint of its configuration differs from the one
its vectors were built with. Stale collections refuse ingestion and retrieval until rebuilt with
``POST /collections/{name}/reindex?recreate=true``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_service.application.pipeline import CollectionPipeline, PipelineFactory, embedding_fingerprint
from rag_service.core.config import (
    PayloadIndexConfig,
    PipelineConfig,
    Settings,
    merge_pipeline,
    validate_collection_name,
)
from rag_service.core.errors import (
    CollectionExistsError,
    CollectionNotFoundError,
    ConflictError,
    EmbeddingConfigChangedError,
    PermissionDeniedError,
    ValidationFailedError,
)
from rag_service.core.logging import get_logger
from rag_service.domain.models import CollectionRecord
from rag_service.domain.ports import CollectionAdmin, CollectionRepository, SourceRepository

log = get_logger(__name__)


@dataclass(slots=True)
class CollectionInfo:
    record: CollectionRecord
    config: PipelineConfig
    stale: bool
    stats: dict[str, int] = field(default_factory=dict)


class CollectionService:
    def __init__(
        self,
        settings: Settings,
        factory: PipelineFactory,
        repository: CollectionRepository,
        sources: SourceRepository,
        admin: CollectionAdmin,
    ) -> None:
        self.settings = settings
        self.factory = factory
        self.repository = repository
        self.sources = sources
        self.admin = admin
        self._pipelines: dict[str, CollectionPipeline] = {}

    # ------------------------------------------------------------------ helpers

    def effective_config(self, record: CollectionRecord) -> PipelineConfig:
        return merge_pipeline(self.settings.defaults, record.pipeline_overrides)

    def is_stale(self, record: CollectionRecord) -> bool:
        return bool(record.embedding_fingerprint) and record.embedding_fingerprint != embedding_fingerprint(
            self.effective_config(record)
        )

    async def _record(self, name: str) -> CollectionRecord:
        record = await self.repository.get(name)
        if record is None:
            raise CollectionNotFoundError(f"Collection '{name}' does not exist")
        return record

    async def _provision(self, record: CollectionRecord) -> CollectionPipeline:
        pipeline = await self.factory.build(record.name, self.effective_config(record))
        spec = self.factory.collection_spec(pipeline, self.admin.capabilities, record.payload_indexes)
        await self.admin.ensure_collection(spec)
        self._pipelines[record.name] = pipeline
        return pipeline

    # ------------------------------------------------------------------ bootstrap

    async def bootstrap(self) -> None:
        """Sync configuration-declared collections into the registry and provision indexes."""
        for name, declared in self.settings.collections.items():
            existing = await self.repository.get(name)
            config = merge_pipeline(self.settings.defaults, declared.pipeline)
            fingerprint = embedding_fingerprint(config)
            record = CollectionRecord(
                name=name,
                description=declared.description,
                pipeline_overrides=declared.pipeline,
                embedding_fingerprint=existing.embedding_fingerprint if existing else fingerprint,
                managed=True,
                payload_indexes=[p.model_dump() for p in declared.payload_indexes],
            )
            if existing and existing.embedding_fingerprint and existing.embedding_fingerprint != fingerprint:
                stats = await self.sources.stats(name)
                if stats["sources"] == 0:
                    log.info("collection.rebuild_empty", collection=name)
                    await self.admin.drop_collection(name)
                    record.embedding_fingerprint = fingerprint
                else:
                    log.error(
                        "collection.stale",
                        collection=name,
                        hint=f"Embedding config changed; run POST /api/v1/collections/{name}/reindex?recreate=true",
                    )
            await self.repository.upsert(record)
        for record in await self.repository.list_all():
            if record.managed and record.name not in self.settings.collections:
                # Removed from the config file: keep the data, hand ownership to the API.
                record.managed = False
                await self.repository.upsert(record)
            if self.is_stale(record):
                continue
            try:
                await self._provision(record)
            except EmbeddingConfigChangedError as exc:
                log.error("collection.incompatible", collection=record.name, error=exc.detail)
        log.info("collections.ready", count=len(self._pipelines))

    # ------------------------------------------------------------------ queries

    async def pipeline(self, name: str) -> CollectionPipeline:
        cached = self._pipelines.get(name)
        if cached is not None:
            return cached
        record = await self._record(name)
        if self.is_stale(record):
            raise EmbeddingConfigChangedError(
                f"Collection '{name}' was indexed with a different embedding configuration. "
                f"Rebuild it with POST /collections/{name}/reindex?recreate=true."
            )
        return await self._provision(record)

    async def get(self, name: str, *, with_stats: bool = True) -> CollectionInfo:
        record = await self._record(name)
        stats: dict[str, int] = {}
        if with_stats:
            stats = await self.sources.stats(name)
            if await self.admin.collection_exists(name):
                stats["points"] = await self.admin.count(name)
        return CollectionInfo(
            record=record, config=self.effective_config(record), stale=self.is_stale(record), stats=stats
        )

    async def list_all(self) -> list[CollectionInfo]:
        return [
            CollectionInfo(record=r, config=self.effective_config(r), stale=self.is_stale(r))
            for r in await self.repository.list_all()
        ]

    # ------------------------------------------------------------------ commands

    async def create(
        self,
        name: str,
        *,
        description: str = "",
        pipeline: dict[str, Any] | None = None,
        payload_indexes: list[PayloadIndexConfig] | None = None,
    ) -> CollectionInfo:
        if not self.settings.allow_dynamic_collections:
            raise PermissionDeniedError("Dynamic collection creation is disabled (allow_dynamic_collections=false)")
        try:
            validate_collection_name(name)
        except ValueError as exc:
            raise ValidationFailedError(str(exc)) from exc
        if await self.repository.get(name):
            raise CollectionExistsError(f"Collection '{name}' already exists")
        overrides = pipeline or {}
        config = merge_pipeline(self.settings.defaults, overrides)
        record = CollectionRecord(
            name=name,
            description=description,
            pipeline_overrides=overrides,
            embedding_fingerprint=embedding_fingerprint(config),
            managed=False,
            payload_indexes=[index.model_dump() for index in (payload_indexes or [])],
        )
        await self._provision(record)  # validates components and creates the index before persisting
        await self.repository.upsert(record)
        log.info("collection.created", collection=name)
        return CollectionInfo(record=record, config=config, stale=False)

    async def update(
        self, name: str, *, description: str | None = None, pipeline: dict[str, Any] | None = None
    ) -> CollectionInfo:
        record = await self._record(name)
        if record.managed:
            raise ConflictError(f"Collection '{name}' is managed by the configuration file; change it there")
        if description is not None:
            record.description = description
        if pipeline is not None:
            merge_pipeline(self.settings.defaults, pipeline)  # validate
            record.pipeline_overrides = pipeline
        await self.repository.upsert(record)
        self._pipelines.pop(name, None)
        stale = self.is_stale(record)
        if not stale:
            await self._provision(record)
        return CollectionInfo(record=record, config=self.effective_config(record), stale=stale)

    async def delete(self, name: str) -> None:
        record = await self._record(name)
        if record.managed:
            raise ConflictError(f"Collection '{name}' is managed by the configuration file; remove it there first")
        await self.admin.drop_collection(name)
        await self.sources.delete_collection(name)
        await self.repository.delete(name)
        self._pipelines.pop(name, None)
        log.info("collection.deleted", collection=name)

    async def recreate_index(self, name: str) -> CollectionPipeline:
        """Drop and recreate the vector index with the *current* configuration (used by reindex)."""
        record = await self._record(name)
        self._pipelines.pop(name, None)
        await self.admin.drop_collection(name)
        record.embedding_fingerprint = embedding_fingerprint(self.effective_config(record))
        pipeline = await self._provision(record)
        await self.repository.upsert(record)
        log.info("collection.index_recreated", collection=name)
        return pipeline
