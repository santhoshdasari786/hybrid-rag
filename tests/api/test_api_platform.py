"""Collections, security, middleware, health and system endpoints."""

from __future__ import annotations

import hashlib

import httpx

from tests.conftest import client_for, make_settings

READER_KEY = "reader-key-123"
WRITER_KEY = "writer-key-456"
ADMIN_KEY = "admin-key-789"


def secured_settings(**extra: object):  # type: ignore[no-untyped-def]
    return make_settings(
        security={
            "auth_enabled": True,
            "api_keys": [
                {"name": "reader", "key": READER_KEY, "scopes": ["query"], "collections": ["products"]},
                {
                    "name": "writer",
                    "key_sha256": hashlib.sha256(WRITER_KEY.encode()).hexdigest(),
                    "scopes": ["query", "ingest"],
                },
                {"name": "admin", "key": ADMIN_KEY, "scopes": ["admin"]},
            ],
            **extra,
        }
    )


async def test_health_and_metrics(app_client: httpx.AsyncClient) -> None:
    assert (await app_client.get("/health/live")).json() == {"status": "ok"}
    ready = await app_client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["checks"] == {"registry": True, "vector_store": True}
    await app_client.get("/api/v1/collections")
    metrics = await app_client.get("/metrics")
    assert "rag_http_requests_total" in metrics.text and 'route="/api/v1/collections"' in metrics.text


async def test_request_id_propagation_and_security_headers(app_client: httpx.AsyncClient) -> None:
    r = await app_client.get("/api/v1/collections", headers={"X-Request-ID": "client-supplied-id-1"})
    assert r.headers["x-request-id"] == "client-supplied-id-1"
    assert r.json()["meta"]["request_id"] == "client-supplied-id-1"
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store"
    bogus = await app_client.get("/api/v1/collections", headers={"X-Request-ID": "bad id with spaces"})
    assert bogus.headers["x-request-id"] != "bad id with spaces"


async def test_collection_lifecycle(app_client: httpx.AsyncClient) -> None:
    created = await app_client.post(
        "/api/v1/collections",
        json={
            "name": "manuals",
            "description": "Manuals",
            "pipeline": {"chunking": {"strategy": "markdown"}, "retrieval": {"top_k": 3}},
        },
    )
    assert created.status_code == 201 and created.headers["location"] == "/api/v1/collections/manuals"
    info = created.json()["data"]
    assert info["pipeline"]["chunking"]["strategy"] == "markdown" and info["managed"] is False

    dup = await app_client.post("/api/v1/collections", json={"name": "manuals"})
    assert dup.status_code == 409 and dup.json()["code"] == "COLLECTION_EXISTS"

    patched = await app_client.patch("/api/v1/collections/manuals", json={"description": "Updated"})
    assert patched.json()["data"]["description"] == "Updated" and patched.json()["data"]["stale"] is False

    # Changing the embedding model makes the collection stale until rebuilt.
    stale = await app_client.patch(
        "/api/v1/collections/manuals",
        json={"pipeline": {"embedding": {"dense": {"provider": "hash", "model": "hash-128"}}}},
    )
    assert stale.json()["data"]["stale"] is True
    await app_client.post("/api/v1/collections/manuals/documents", json={"documents": [{"content": "x"}]})
    blocked = await app_client.post("/api/v1/collections/manuals/retrieve", json={"query": "x"})
    assert blocked.status_code == 409 and blocked.json()["code"] == "EMBEDDING_CONFIG_CHANGED"
    rebuilt = await app_client.post("/api/v1/collections/manuals/reindex", params={"recreate": "true"})
    assert rebuilt.status_code == 202
    ok = await app_client.post("/api/v1/collections/manuals/retrieve", json={"query": "x"})
    assert ok.status_code == 200

    managed = await app_client.delete("/api/v1/collections/products")
    assert managed.status_code == 409
    assert (await app_client.delete("/api/v1/collections/manuals")).status_code == 204
    assert (await app_client.get("/api/v1/collections/manuals")).status_code == 404


async def test_invalid_collection_pipeline_rejected(app_client: httpx.AsyncClient) -> None:
    r = await app_client.post(
        "/api/v1/collections", json={"name": "bad", "pipeline": {"chunking": {"strategy": "nope"}}}
    )
    assert r.status_code == 500 and r.json()["code"] == "CONFIGURATION_ERROR"
    r2 = await app_client.post("/api/v1/collections", json={"name": "Bad Name"})
    assert r2.status_code == 422


async def test_authentication_scopes_and_collection_acl() -> None:
    async with client_for(secured_settings()) as client:
        assert (await client.get("/api/v1/collections")).status_code == 401
        assert (await client.get("/api/v1/collections", headers={"X-API-Key": "wrong"})).status_code == 401
        assert (await client.get("/health/live")).status_code == 200  # probes stay open

        reader = {"X-API-Key": READER_KEY}
        assert (await client.get("/api/v1/collections/products", headers=reader)).status_code == 200
        ingest = await client.post(
            "/api/v1/collections/products/documents", headers=reader, json={"documents": [{"content": "x"}]}
        )
        assert ingest.status_code == 403 and "ingest" in ingest.json()["detail"]

        writer = {"Authorization": f"Bearer {WRITER_KEY}"}  # hashed key + bearer form
        ok = await client.post(
            "/api/v1/collections/products/documents", headers=writer, json={"documents": [{"content": "x"}]}
        )
        assert ok.status_code == 200
        assert (await client.post("/api/v1/collections", headers=writer, json={"name": "x"})).status_code == 403

        admin = {"X-API-Key": ADMIN_KEY}
        assert (await client.post("/api/v1/collections", headers=admin, json={"name": "other"})).status_code == 201
        # The reader key is restricted to `products`.
        assert (await client.get("/api/v1/collections/other", headers=reader)).status_code == 403
        visible = (await client.get("/api/v1/collections", headers=reader)).json()["data"]
        assert [c["name"] for c in visible] == ["products"]

        info = await client.get("/api/v1/system/info", headers=admin)
        assert info.status_code == 200 and "chunkers" in info.json()["data"]["components"]
        assert (await client.get("/api/v1/system/info", headers=reader)).status_code == 403
        config = (await client.get("/api/v1/system/config", headers=admin)).text
        assert READER_KEY not in config and ADMIN_KEY not in config


async def test_rate_limiting() -> None:
    settings = make_settings(security={"rate_limit": {"enabled": True, "requests_per_minute": 60, "burst": 2}})
    async with client_for(settings) as client:
        codes = [(await client.get("/api/v1/collections")).status_code for _ in range(4)]
        assert codes[:2] == [200, 200] and 429 in codes
        limited = await client.get("/api/v1/collections")
        assert limited.headers.get("retry-after") and limited.json()["code"] == "RATE_LIMITED"


async def test_body_size_limit() -> None:
    settings = make_settings(security={"max_request_body_mb": 1})
    async with client_for(settings) as client:
        big = "x" * (1024 * 1024 + 10)
        r = await client.post("/api/v1/collections/products/documents", json={"documents": [{"content": big}]})
        assert r.status_code == 413 and r.json()["code"] == "PAYLOAD_TOO_LARGE"


async def test_openapi_documents_problem_responses(app_client: httpx.AsyncClient) -> None:
    spec = (await app_client.get("/openapi.json")).json()
    assert "Problem" in spec["components"]["schemas"]
    retrieve = spec["paths"]["/api/v1/collections/{collection}/retrieve"]["post"]
    assert "422" in retrieve["responses"] and "404" in retrieve["responses"]
