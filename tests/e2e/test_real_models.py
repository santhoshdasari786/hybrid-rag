"""End-to-end with real models (opt-in: RAG_MODELS=1; downloads ~700 MB on first run).

Reproduces the class notebook scenario through the HTTP API: BGE dense + SPLADE++ sparse + RRF,
then a cross-encoder rerank, on the synthetic ESCI-style catalog in ``examples/``.

Optional real infrastructure:
  RAG_IT_QDRANT_URL=http://localhost:6399          → Qdrant server instead of the memory store
  RAG_IT_PG_DSN=postgresql://rag:rag@localhost:5439/rag → PostgreSQL source registry
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import client_for, make_settings

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(os.environ.get("RAG_MODELS") != "1", reason="set RAG_MODELS=1 to run real-model tests"),
]

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
CACHE = os.environ.get("FASTEMBED_CACHE", str(Path(__file__).resolve().parents[2] / "data" / "models"))


def settings() -> Any:
    vector_store: dict[str, Any] = {"provider": "memory"}
    if os.environ.get("RAG_IT_QDRANT_URL"):
        vector_store = {
            "provider": "qdrant",
            "qdrant": {"url": os.environ["RAG_IT_QDRANT_URL"], "collection_prefix": f"e2e{uuid.uuid4().hex[:6]}_"},
        }
    registry = {"url": "sqlite+aiosqlite:///:memory:"}
    if os.environ.get("RAG_IT_PG_DSN"):
        registry = {"url": os.environ["RAG_IT_PG_DSN"].replace("postgresql://", "postgresql+psycopg://")}
    opts = {"cache_dir": CACHE}
    return make_settings(
        vector_store=vector_store,
        registry=registry,
        defaults={
            "chunking": {"strategy": "markdown", "chunk_size": 256, "chunk_overlap": 32, "length_unit": "tokens"},
            "embedding": {
                "dense": {
                    "provider": "fastembed",
                    "model": "BAAI/bge-small-en-v1.5",
                    "query_prefix": "Represent this sentence for searching relevant passages: ",
                    "options": opts,
                },
                "sparse": {"provider": "fastembed", "model": "prithivida/Splade_PP_en_v1", "options": opts},
            },
            "retrieval": {
                "rerank": {"provider": "fastembed", "model": "Xenova/ms-marco-MiniLM-L-6-v2", "options": opts}
            },
        },
        collections={f"e2e{uuid.uuid4().hex[:6]}": {}},
    )


async def test_notebook_scenario_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    s = settings()
    collection = next(iter(s.collections))
    base = f"/api/v1/collections/{collection}"
    payload = json.loads((EXAMPLES / "ingest_products.json").read_text())
    ctx = client_for(s)
    async with ctx as client:
        r = await client.post(f"{base}/documents", json=payload, timeout=600)
        assert r.status_code == 200, r.text
        assert r.json()["data"]["succeeded"] == len(payload["documents"])

        # Exact product query — sparse (SPLADE) nails model names and numbers.
        r = await client.post(f"{base}/retrieve", json={"query": "revent 80 cfm", "top_k": 3, "include_debug": True})
        results = r.json()["data"]["results"]
        assert results[0]["source_id"] == "fan-revent-rg80", results

        # Conceptual query — dense understands "steamy shower" ≈ humidity sensor.
        r = await client.post(
            f"{base}/retrieve",
            json={
                "query": "fan that turns on automatically when the shower gets steamy",
                "top_k": 3,
                "retrieval": {"rerank": {"enabled": True}},
            },
        )
        top = [x["source_id"] for x in r.json()["data"]["results"]]
        assert "fan-delta-vfb25ach" in top[:2] or "fan-aeropure-ap120h" in top[:2], top

        # Re-sync with an edit re-embeds only the changed chunk.
        doc = payload["documents"][0] | {
            "content": payload["documents"][0]["content"] + "\n\nNow with a 10 year warranty."
        }
        r = await client.put(
            f"{base}/documents/{doc['source_id']}", json={k: v for k, v in doc.items() if k != "source_id"}
        )
        result = r.json()["data"]["results"][0]
        assert result["action"] == "updated" and result["version"] == 2

        evaluation = json.loads((EXAMPLES / "evaluate_products.json").read_text())
        r = await client.post(f"{base}/evaluate", json=evaluation, timeout=600)
        report = r.json()["data"]
        with capsys.disabled():
            print(f"\n{'variant':<16}{'hit':>7}{'recall':>8}{'mrr':>7}{'ndcg':>7}{'p50 ms':>9}")
            for v in report["variants"]:
                m = v["metrics"]
                print(
                    f"{v['name']:<16}{m['hit_rate']:>7.2f}{m['recall']:>8.2f}{m['mrr']:>7.2f}{m['ndcg']:>7.2f}{m['latency_ms_p50']:>9.1f}"
                )
        by_name = {v["name"]: v["metrics"] for v in report["variants"]}
        assert by_name["hybrid"]["ndcg"] >= min(by_name["dense"]["ndcg"], by_name["sparse"]["ndcg"])

        # Clean up (the collection is config-managed, so remove its data directly).
        container = ctx.app.state.container
        await container.store.drop_collection(collection)
        await container.ingestion.sources.delete_collection(collection)
        await container.collections.repository.delete(collection)
