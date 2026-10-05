"""Result grouping (``retrieval.group_by`` / ``group_size``): product variants and pages collapse to one hit."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from rag_service.core.config import PayloadIndexConfig
from tests.conftest import client_for, make_settings

URL = "/api/v1/collections/products"


def sku(source_id: str, family: str, title: str, extra: str = "") -> dict[str, Any]:
    return {
        "source_id": source_id,
        "title": title,
        "content": f"{title}. Quiet bathroom exhaust fan, 80 CFM, ceiling mounted. {extra}",
        "metadata": {"family": family, "brand": "AirFlo"},
    }


DOCS = [
    sku("af80q-white", "af80q", "AirFlo AF-80Q White"),
    sku("af80q-black", "af80q", "AirFlo AF-80Q Black"),
    sku("af80q-grey", "af80q", "AirFlo AF-80Q Grey"),
    sku("af80qh", "af80qh", "AirFlo AF-80QH Humidity Sensor", "Turns on automatically when humidity rises."),
    sku("af110q", "af110q", "AirFlo AF-110Q", "110 CFM for larger bathrooms."),
]


@pytest.fixture
async def catalog(app_client: httpx.AsyncClient) -> httpx.AsyncClient:
    response = await app_client.post(f"{URL}/documents", json={"documents": DOCS, "options": {"mode": "sync"}})
    assert response.status_code == 200, response.text
    return app_client


async def search(client: httpx.AsyncClient, **body: Any) -> list[dict[str, Any]]:
    response = await client.post(f"{URL}/retrieve", json={"query": "airflo quiet bathroom fan", "top_k": 5, **body})
    assert response.status_code == 200, response.text
    return list(response.json()["data"]["results"])


async def test_variants_collapse_to_one_hit_per_family(catalog: httpx.AsyncClient) -> None:
    ungrouped = await search(catalog)
    assert len({r["metadata"]["family"] for r in ungrouped}) < len(ungrouped)  # variants crowd the page
    grouped = await search(catalog, retrieval={"group_by": "family"})
    families = [r["metadata"]["family"] for r in grouped]
    assert sorted(families) == ["af110q", "af80q", "af80qh"]


async def test_group_size_and_metadata_prefix(catalog: httpx.AsyncClient) -> None:
    results = await search(catalog, retrieval={"group_by": "metadata.family", "group_size": 2})
    families = [r["metadata"]["family"] for r in results]
    assert families.count("af80q") == 2 and len(results) == 4


async def test_empty_group_by_disables_a_configured_grouping() -> None:
    settings = make_settings(collections={"products": {"pipeline": {"retrieval": {"group_by": "family"}}}})
    async with client_for(settings) as client:
        response = await client.post(f"{URL}/documents", json={"documents": DOCS, "options": {"mode": "sync"}})
        assert response.status_code == 200, response.text
        assert len(await search(client)) == 3  # the collection groups by family
        assert len(await search(client, retrieval={"group_by": ""})) == 5  # a variant picker sees every SKU


async def test_group_by_source_id_limits_chunks_per_document(app_client: httpx.AsyncClient) -> None:
    long_doc = {
        "source_id": "guide",
        "content": "\n\n".join(f"Section {i}. The bathroom fan guide covers airflow and noise." for i in range(12)),
        "chunking": {"strategy": "fixed", "chunk_size": 16, "chunk_overlap": 0, "length_unit": "words"},
    }
    other = {"source_id": "faq", "content": "FAQ: how loud is a bathroom fan? Noise is measured in sones."}
    response = await app_client.post(
        f"{URL}/documents", json={"documents": [long_doc, other], "options": {"mode": "sync"}}
    )
    assert response.status_code == 200, response.text
    body = {"query": "bathroom fan noise", "top_k": 6, "retrieval": {"group_by": "source_id", "group_size": 2}}
    results = (await app_client.post(f"{URL}/retrieve", json=body)).json()["data"]["results"]
    assert [r["source_id"] for r in results].count("guide") == 2
    assert "faq" in {r["source_id"] for r in results}


def test_tenant_index_must_be_keyword() -> None:
    assert PayloadIndexConfig(field="site", tenant=True).tenant
    with pytest.raises(ValueError, match="keyword"):
        PayloadIndexConfig(field="price", type="float", tenant=True)
