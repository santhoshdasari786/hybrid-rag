from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

PREFIX = "/api/v1/collections/products"


@pytest.fixture
async def loaded(app_client: httpx.AsyncClient, fan_docs: list[dict[str, Any]]) -> httpx.AsyncClient:
    r = await app_client.post(f"{PREFIX}/documents", json={"documents": fan_docs, "options": {"mode": "sync"}})
    assert r.status_code == 200, r.text
    return app_client


def sse_events(text: str) -> list[tuple[str | None, Any]]:
    events = []
    for block in text.strip().split("\n\n"):
        event, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                raw = line[6:]
                data = raw if raw == "[DONE]" else json.loads(raw)
        events.append((event, data))
    return events


async def test_hybrid_retrieval_finds_exact_spec(loaded: httpx.AsyncClient) -> None:
    r = await loaded.post(f"{PREFIX}/retrieve", json={"query": "80 CFM bathroom exhaust fan", "top_k": 3})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["mode"] == "hybrid"
    top_sources = [x["source_id"] for x in data["results"]]
    assert "blender-ninja" not in top_sources
    first = data["results"][0]
    assert {"fused", "dense_rank", "sparse_rank"} <= set(first["scores"])
    assert first["rank"] == 1 and first["text"]


async def test_filters_modes_and_debug(loaded: httpx.AsyncClient) -> None:
    r = await loaded.post(
        f"{PREFIX}/retrieve",
        json={"query": "fan", "filter": {"brand": "Delta"}, "mode": "sparse", "include_debug": True},
    )
    data = r.json()["data"]
    assert data["mode"] == "sparse"
    assert {x["source_id"] for x in data["results"]} == {"fan-delta-rad80"}
    assert data["debug"]["lists"][0]["retriever"] == "sparse"
    assert data["config"]["mode"] == "sparse"

    structured = await loaded.post(
        f"{PREFIX}/retrieve",
        json={
            "query": "fan",
            "filter": {
                "must": [{"field": "tags", "op": "in", "value": ["quiet"]}],
                "must_not": [{"field": "brand", "op": "eq", "value": "Broan-NuTone"}],
            },
        },
    )
    assert {x["source_id"] for x in structured.json()["data"]["results"]} == {"fan-panasonic-fv20vq3"}

    bad = await loaded.post(
        f"{PREFIX}/retrieve",
        json={"query": "fan", "filter": {"must": [{"field": "x", "op": "in", "value": 3}]}},
    )
    assert bad.status_code == 422


async def test_retrieval_overrides_mmr_context_expansion_and_fusion(loaded: httpx.AsyncClient) -> None:
    r = await loaded.post(
        f"{PREFIX}/retrieve",
        json={
            "query": "quiet fan installation",
            "retrieval": {
                "fusion": {"method": "dbsf", "weights": {"dense": 1.0, "sparse": 2.0}},
                "mmr": {"enabled": True, "lambda_mult": 0.5},
                "context_expansion": {"enabled": True, "window": 1},
                "top_k": 4,
            },
        },
    )
    assert r.status_code == 200, r.text
    results = r.json()["data"]["results"]
    assert len(results) == 4
    assert all(x["context"] for x in results)
    assert any("mmr" in x["scores"] for x in results)

    invalid = await loaded.post(
        f"{PREFIX}/retrieve", json={"query": "fan", "retrieval": {"fusion": {"method": "magic"}}}
    )
    assert invalid.status_code == 500 or invalid.status_code == 422
    assert invalid.json()["code"] in ("CONFIGURATION_ERROR", "VALIDATION_FAILED")


async def test_chat_json_with_citations(loaded: httpx.AsyncClient) -> None:
    r = await loaded.post(
        f"{PREFIX}/chat",
        json={
            "messages": [{"role": "user", "content": "Does the Panasonic fan prevent backdraft?"}],
            "conversation_id": "c-1",
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["message"]["role"] == "assistant" and "damper" in data["message"]["content"].lower()
    assert data["citations"] and data["citations"][0]["source_id"] == "fan-panasonic-fv20vq3"
    assert f"[{data['citations'][0]['index']}]" in data["message"]["content"]
    assert data["conversation_id"] == "c-1" and data["finish_reason"] == "stop"
    assert data["sources"] and data["retrieval"]["standalone_query"]


async def test_chat_without_context_short_circuits(loaded: httpx.AsyncClient) -> None:
    r = await loaded.post(
        f"{PREFIX}/chat",
        json={"messages": [{"role": "user", "content": "anything"}], "filter": {"brand": "DoesNotExist"}},
    )
    data = r.json()["data"]
    assert data["finish_reason"] == "no_context" and data["citations"] == []


async def test_chat_streaming_sse(loaded: httpx.AsyncClient) -> None:
    r = await loaded.post(
        f"{PREFIX}/chat",
        json={
            "stream": True,
            "messages": [{"role": "user", "content": "How loud is the Broan fan in sones?"}],
        },
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    names = [e for e, _ in events]
    assert names[0] == "retrieval" and names[-1] == "done" and "delta" in names
    answer = "".join(d["content"] for e, d in events if e == "delta")
    assert "0.3 sones" in answer
    assert events[-1][1]["citations"]


async def test_stream_errors_before_streaming_are_http_errors(loaded: httpx.AsyncClient) -> None:
    r = await loaded.post(
        "/api/v1/collections/missing/chat",
        json={"stream": True, "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/problem+json")


async def test_openai_compatible_endpoints(loaded: httpx.AsyncClient) -> None:
    models = (await loaded.get("/v1/models")).json()
    assert {"id": "rag:products"}.items() <= models["data"][0].items()

    body = {
        "model": "rag:products",
        "messages": [{"role": "user", "content": "What heats the bathroom? Delta"}],
    }
    r = await loaded.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["object"] == "chat.completion" and data["choices"][0]["message"]["role"] == "assistant"
    assert data["citations"] and "usage" in data

    stream = await loaded.post("/v1/chat/completions", json={**body, "stream": True})
    events = sse_events(stream.text)
    assert events[-1][1] == "[DONE]"
    chunks = [d for _, d in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and "citations" in chunks[-1]

    missing = await loaded.post("/v1/chat/completions", json={**body, "model": "rag:nope"})
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "COLLECTION_NOT_FOUND"


async def test_evaluation_compares_variants(loaded: httpx.AsyncClient) -> None:
    r = await loaded.post(
        f"{PREFIX}/evaluate",
        json={
            "k": 3,
            "cases": [
                {"query": "80 cfm fan with heater", "relevant": ["fan-delta-rad80"]},
                {
                    "query": "whisper quiet ceiling fan damper",
                    "relevant": {"fan-panasonic-fv20vq3": 2, "fan-broan-quiet80": 1},
                },
                {"query": "smoothie blender", "relevant": ["blender-ninja"]},
            ],
            "include_cases": True,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert [v["name"] for v in data["variants"]] == ["dense", "sparse", "hybrid"]
    hybrid = next(v for v in data["variants"] if v["name"] == "hybrid")
    assert hybrid["metrics"]["hit_rate"] == 1.0
    assert set(hybrid["metrics"]) >= {"recall", "precision", "mrr", "ndcg", "latency_ms_p50"}
    assert len(hybrid["cases"]) == 3
    assert data["best"] is None or data["best"] in {v["name"] for v in data["variants"]}
