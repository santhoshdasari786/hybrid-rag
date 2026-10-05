"""The source-sync contract: re-sending a source_id re-embeds only what changed."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from rag_service.application.ingestion import ChunkInput, DocumentInput
from rag_service.container import Container
from rag_service.domain.filters import FilterSpec
from rag_service.domain.models import IngestAction, SourceStatus
from tests.conftest import CountingEmbedder, make_settings

PARAGRAPHS = [
    f"Paragraph {i}: the ventilation fan model {i} moves {60 + i} cubic feet per minute quietly." for i in range(6)
]


@pytest.fixture
async def svc() -> Any:
    settings = make_settings(
        defaults={
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 20,
                "chunk_overlap": 0,
                "length_unit": "words",
            },
            "embedding": {
                "dense": {"provider": "counting", "model": "counting", "dimension": 64},
                "sparse": {"provider": "lexical"},
            },
        },
    )
    c = Container.build(settings)
    await c.start()
    CountingEmbedder.calls.clear()
    yield c
    await c.stop()


def doc(content: str, **kw: Any) -> DocumentInput:
    return DocumentInput(source_id=kw.pop("source_id", "doc-1"), content=content, **kw)


async def sync(c: Container, d: DocumentInput, **kw: Any):  # type: ignore[no-untyped-def]
    (result,) = await c.ingestion.ingest("products", [d], **kw)
    return result


async def points(c: Container, source_id: str = "doc-1") -> list[Any]:
    pts, _ = await c.store.scroll("products", filter_=FilterSpec.for_source(source_id), limit=1000)
    return sorted(pts, key=lambda p: p.payload["chunk_index"])


async def test_create_then_unchanged(svc: Container) -> None:
    body = "\n\n".join(PARAGRAPHS)
    first = await sync(svc, doc(body, metadata={"brand": "Delta"}))
    assert first.action == IngestAction.CREATED and first.chunks_total == 6 and first.version == 1
    assert len(CountingEmbedder.calls) == 6

    CountingEmbedder.calls.clear()
    second = await sync(svc, doc(body, metadata={"brand": "Delta"}))
    assert second.action == IngestAction.UNCHANGED
    assert CountingEmbedder.calls == []  # nothing re-embedded
    assert len(await points(svc)) == 6


async def test_metadata_only_change_patches_without_embedding(svc: Container) -> None:
    body = "\n\n".join(PARAGRAPHS)
    await sync(svc, doc(body, metadata={"brand": "Delta"}))
    CountingEmbedder.calls.clear()
    result = await sync(svc, doc(body, metadata={"brand": "Broan", "cfm": 80}, title="New title"))
    assert result.action == IngestAction.METADATA_UPDATED and result.chunks_updated == 6 and result.version == 2
    assert CountingEmbedder.calls == []
    pts = await points(svc)
    assert {p.payload["metadata"]["brand"] for p in pts} == {"Broan"}
    assert {p.payload["title"] for p in pts} == {"New title"}


async def test_edit_reembeds_only_changed_chunks_and_removes_stale(svc: Container) -> None:
    await sync(svc, doc("\n\n".join(PARAGRAPHS)))
    old_ids = {p.id for p in await points(svc)}
    CountingEmbedder.calls.clear()

    edited = [*PARAGRAPHS]
    edited[2] = "Paragraph 2 was rewritten: this fan now has a humidity sensor."
    del edited[4]
    result = await sync(svc, doc("\n\n".join(edited)))

    assert result.action == IngestAction.UPDATED and result.version == 2
    assert result.chunks_added == 1 and result.chunks_removed == 2 and result.chunks_total == 5
    assert CountingEmbedder.calls == [edited[2]]  # exactly one chunk embedded (contextual header disabled: no title)
    pts = await points(svc)
    assert [p.payload["chunk_index"] for p in pts] == [0, 1, 2, 3, 4]  # moved chunks were re-numbered
    assert len(old_ids & {p.id for p in pts}) == 4  # unchanged chunks kept their IDs
    assert (await svc.store.count("products")) == 5


async def test_delete_removes_every_point_and_registry_row(svc: Container) -> None:
    await sync(svc, doc("\n\n".join(PARAGRAPHS)))
    result = await svc.ingestion.delete_source("products", "doc-1")
    assert result.action == IngestAction.DELETED and result.chunks_removed == 6
    assert await points(svc) == []
    assert await svc.sources.sources.get("products", "doc-1") is None
    again = await svc.ingestion.delete_source("products", "doc-1")
    assert again.action == IngestAction.NOT_FOUND


async def test_chunk_metadata_override_survives_resync(svc: Container) -> None:
    await sync(svc, doc("\n\n".join(PARAGRAPHS), metadata={"brand": "Delta"}))
    target = (await points(svc))[1]
    merged = await svc.ingestion.patch_chunk_metadata("products", target.id, {"verified": True})
    assert merged["verified"] is True and merged["brand"] == "Delta"

    # Re-sync with a metadata change and an unrelated content edit: the override must persist.
    edited = [*PARAGRAPHS]
    edited[5] = "Paragraph 5 changed."
    await sync(svc, doc("\n\n".join(edited), metadata={"brand": "Delta", "season": "2026"}))
    (point,) = await svc.store.fetch("products", [target.id])
    meta = point.payload["metadata"]
    assert (meta["brand"], meta["season"], meta["verified"]) == ("Delta", "2026", True)


async def test_force_reembeds_everything(svc: Container) -> None:
    body = "\n\n".join(PARAGRAPHS)
    await sync(svc, doc(body))
    CountingEmbedder.calls.clear()
    result = await sync(svc, doc(body), force=True)
    assert result.chunks_added == 6 and len(CountingEmbedder.calls) == 6
    assert await svc.store.count("products") == 6


async def test_prechunked_input_keeps_per_chunk_metadata(svc: Container) -> None:
    d = DocumentInput(
        source_id="faq",
        chunks=[
            ChunkInput("How loud is it? 0.3 sones.", {"qa_id": 1}),
            ChunkInput("Does it have a light? Yes, LED.", {"qa_id": 2}),
        ],
        metadata={"brand": "Broan"},
    )
    result = await sync(svc, d)
    assert result.chunks_total == 2
    pts = await points(svc, "faq")
    assert [p.payload["metadata"]["qa_id"] for p in pts] == [1, 2]
    assert all(p.payload["metadata"]["brand"] == "Broan" for p in pts)


async def test_failures_are_recorded_and_retry_succeeds(svc: Container) -> None:
    bad = DocumentInput(source_id="bad", content_base64="!!!not-base64!!!", mime_type="application/pdf")
    result = await sync(svc, bad)
    assert result.action == IngestAction.FAILED and result.error_code == "VALIDATION_FAILED"
    record = await svc.sources.sources.get("products", "bad")
    assert record is not None and record.status == SourceStatus.FAILED

    good = DocumentInput(source_id="bad", content=base64.b64decode(base64.b64encode(b"now valid text")).decode())
    retry = await sync(svc, good)
    assert retry.action == IngestAction.CREATED


async def test_chunking_config_change_rechunks_on_reindex(svc: Container) -> None:
    await sync(svc, doc("\n\n".join(PARAGRAPHS)))
    assert await svc.store.count("products") == 6
    # A per-document override changes the ingest fingerprint → the source is re-chunked.
    result = await sync(svc, doc("\n\n".join(PARAGRAPHS), chunking={"chunk_size": 200}))
    assert result.action == IngestAction.UPDATED and result.chunks_total == 1 and result.chunks_removed == 6
    reindexed = await svc.ingestion.reindex("products", force=False)
    assert [r.action for r in reindexed] == [IngestAction.UNCHANGED]  # stored override is re-applied


async def test_unsupported_mime_type(svc: Container) -> None:
    result = await sync(svc, DocumentInput(source_id="x", content="a", mime_type="application/x-foo"))
    assert result.action == IngestAction.FAILED and result.error_code == "UNSUPPORTED_MEDIA_TYPE"
