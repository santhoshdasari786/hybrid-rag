from __future__ import annotations

import pytest

from rag_service.core.registry import RERANKERS  # ← your registry
from tests.conftest import client_for, make_settings


def test_component_is_registered() -> None:
    assert "example" in RERANKERS


@pytest.mark.skip(reason="template")
async def test_component_behaviour() -> None:
    # Unit-level: build the component with a config object and assert its behaviour.
    ...


@pytest.mark.skip(reason="template")
async def test_component_through_the_api() -> None:
    settings = make_settings(defaults={"retrieval": {"rerank": {"enabled": True, "provider": "example"}}})
    async with client_for(settings) as client:
        r = await client.post("/api/v1/collections/products/retrieve", json={"query": "fan"})
        assert r.status_code == 200
