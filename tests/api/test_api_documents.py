from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx

from tests.conftest import client_for, make_settings

PREFIX = "/api/v1/collections/products"


async def ingest(client: httpx.AsyncClient, docs: list[dict[str, Any]], **options: Any) -> httpx.Response:
    return await client.post(f"{PREFIX}/documents", json={"documents": docs, "options": options})


async def test_sync_ingest_returns_envelope_with_results(
    app_client: httpx.AsyncClient, fan_docs: list[dict[str, Any]]
) -> None:
    r = await ingest(app_client, fan_docs, mode="sync")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"data", "meta"}
    assert body["meta"]["request_id"] and body["meta"]["api_version"] == "v1"
    job = body["data"]
    assert job["status"] == "succeeded" and job["total"] == 4 and job["succeeded"] == 4
    assert {x["action"] for x in job["results"]} == {"created"}
    assert r.headers["x-request-id"] == body["meta"]["request_id"]

    again = (await ingest(app_client, fan_docs, mode="sync")).json()["data"]
    assert {x["action"] for x in again["results"]} == {"unchanged"}


async def test_async_ingest_returns_202_and_job_completes(
    app_client: httpx.AsyncClient, fan_docs: list[dict[str, Any]]
) -> None:
    r = await ingest(app_client, fan_docs, mode="async")
    assert r.status_code == 202
    location = r.headers["location"]
    assert location.startswith("/api/v1/jobs/job_")
    for _ in range(50):
        job = (await app_client.get(location)).json()["data"]
        if job["status"] not in ("queued", "running"):
            break
        await asyncio.sleep(0.05)
    assert job["status"] == "succeeded" and job["processed"] == 4
    jobs = (await app_client.get("/api/v1/jobs", params={"collection": "products"})).json()
    assert jobs["meta"]["pagination"]["total"] >= 1


async def test_generated_source_id_and_put_upsert(app_client: httpx.AsyncClient) -> None:
    r = await ingest(app_client, [{"content": "A document without an id."}])
    source_id = r.json()["data"]["results"][0]["source_id"]
    assert source_id.startswith("src_")

    put = await app_client.put(
        f"{PREFIX}/documents/kb:faq:1", json={"content": "Version one.", "metadata": {"lang": "en"}}
    )
    assert put.status_code == 200 and put.json()["data"]["results"][0]["action"] == "created"
    put2 = await app_client.put(
        f"{PREFIX}/documents/kb:faq:1", json={"content": "Version two, changed.", "metadata": {"lang": "en"}}
    )
    assert put2.json()["data"]["results"][0]["action"] == "updated"
    src = (await app_client.get(f"{PREFIX}/documents/kb:faq:1")).json()["data"]
    assert src["version"] == 2 and src["metadata"] == {"lang": "en"} and src["status"] == "active"


async def test_list_sources_chunks_and_metadata_patch(
    app_client: httpx.AsyncClient, fan_docs: list[dict[str, Any]]
) -> None:
    await ingest(app_client, fan_docs)
    listing = (await app_client.get(f"{PREFIX}/documents", params={"limit": 2})).json()
    assert len(listing["data"]) == 2 and listing["meta"]["pagination"] == {
        "limit": 2,
        "offset": 0,
        "total": 4,
        "has_more": True,
    }

    chunks = (await app_client.get(f"{PREFIX}/documents/fan-panasonic-fv20vq3/chunks")).json()["data"]
    assert chunks and chunks[0]["metadata"]["brand"] == "Panasonic"
    assert chunks[0]["metadata"]["section"].startswith("Panasonic FV-20VQ3")

    patched = await app_client.patch(
        f"{PREFIX}/documents/fan-panasonic-fv20vq3/metadata", json={"metadata": {"in_stock": True}}
    )
    assert patched.status_code == 200 and patched.json()["data"]["action"] == "metadata_updated"
    chunk = (await app_client.get(f"{PREFIX}/chunks/{chunks[0]['chunk_id']}")).json()["data"]
    assert chunk["metadata"]["in_stock"] is True and chunk["metadata"]["brand"] == "Panasonic"

    chunk_patch = await app_client.patch(
        f"{PREFIX}/chunks/{chunks[0]['chunk_id']}/metadata", json={"metadata": {"reviewed_by": "qa"}}
    )
    assert chunk_patch.json()["data"]["metadata"]["reviewed_by"] == "qa"


async def test_delete_and_bulk_delete(app_client: httpx.AsyncClient, fan_docs: list[dict[str, Any]]) -> None:
    await ingest(app_client, fan_docs)
    r = await app_client.delete(f"{PREFIX}/documents/blender-ninja")
    assert r.status_code == 200 and r.json()["data"]["action"] == "deleted"
    missing = await app_client.delete(f"{PREFIX}/documents/blender-ninja")
    assert missing.status_code == 404 and missing.json()["code"] == "SOURCE_NOT_FOUND"

    bulk = await app_client.post(f"{PREFIX}/documents:delete", json={"source_ids": ["fan-delta-rad80", "nope"]})
    actions = {x["source_id"]: x["action"] for x in bulk.json()["data"]["results"]}
    assert actions == {"fan-delta-rad80": "deleted", "nope": "not_found"}
    info = (await app_client.get(PREFIX)).json()["data"]
    assert info["stats"]["sources"] == 2 and info["stats"]["points"] == info["stats"]["chunks"]


async def test_base64_and_html_documents(app_client: httpx.AsyncClient) -> None:
    html = "<html><head><title>Fan manual</title></head><body><h1>Setup</h1><p>Mount the fan.</p></body></html>"
    r = await ingest(
        app_client,
        [
            {
                "source_id": "manual",
                "content_base64": base64.b64encode(html.encode()).decode(),
                "mime_type": "text/html",
            }
        ],
    )
    assert r.json()["data"]["results"][0]["action"] == "created"
    src = (await app_client.get(f"{PREFIX}/documents/manual")).json()["data"]
    assert src["title"] == "Fan manual"


async def test_validation_errors_are_problem_json(app_client: httpx.AsyncClient) -> None:
    r = await ingest(app_client, [{"source_id": "x", "content": "a", "chunks": [{"text": "b"}]}])
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("application/problem+json")
    problem = r.json()
    assert problem["code"] == "VALIDATION_FAILED" and problem["type"].startswith("urn:rag-service:problem:")
    assert problem["errors"] and problem["request_id"]

    bad_key = await ingest(app_client, [{"source_id": "x", "content": "a", "metadata": {"a.b": 1}}])
    assert bad_key.status_code == 422

    dup = await ingest(app_client, [{"source_id": "x", "content": "a"}, {"source_id": "x", "content": "b"}])
    assert dup.status_code == 422 and "Duplicate" in dup.json()["detail"]

    unknown = await app_client.post("/api/v1/collections/nope/documents", json={"documents": [{"content": "a"}]})
    assert unknown.status_code == 404 and unknown.json()["code"] == "COLLECTION_NOT_FOUND"


async def test_all_failed_batch_is_422_and_partial_is_207(app_client: httpx.AsyncClient) -> None:
    bad = {"source_id": "bad", "content": "x", "mime_type": "application/x-unknown"}
    r = await ingest(app_client, [bad])
    assert r.status_code == 422 and r.json()["data"]["status"] == "failed"
    mixed = await ingest(app_client, [bad, {"source_id": "ok", "content": "fine"}])
    assert mixed.status_code == 207 and mixed.json()["data"]["status"] == "partially_succeeded"


async def test_large_batches_are_forced_async() -> None:
    settings = make_settings(ingestion={"default_mode": "sync", "sync_max_documents": 2})
    async with client_for(settings) as client:
        r = await ingest(client, [{"content": f"doc {i}"} for i in range(3)], mode="sync")
        assert r.status_code == 202
