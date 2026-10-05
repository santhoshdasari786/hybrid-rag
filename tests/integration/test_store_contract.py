"""One behavioural contract, every vector store.

* ``memory`` and ``qdrant-embedded`` always run (no external services).
* ``qdrant-server`` runs when ``RAG_IT_QDRANT_URL`` is set (e.g. http://localhost:6333).
* ``pgvector`` runs when ``RAG_IT_PG_DSN`` is set (e.g. postgresql://rag:rag@localhost:5439/rag).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest

from rag_service.core.config import DistanceMetric, FusionConfig, VectorStoreConfig
from rag_service.core.errors import CapabilityNotSupportedError
from rag_service.core.registry import VECTOR_STORES
from rag_service.domain.filters import FilterSpec, parse_filter
from rag_service.domain.models import CollectionSpec, PayloadPatch, SparseVector, VectorRecord
from rag_service.domain.ports import VectorStore


def _params() -> list[Any]:
    params: list[Any] = [pytest.param(("memory", {}), id="memory")]
    try:
        import qdrant_client  # noqa: F401

        params.append(pytest.param(("qdrant", {"qdrant": {"location": ":memory:"}}), id="qdrant-embedded"))
    except ImportError:
        pass
    if os.environ.get("RAG_IT_QDRANT_URL"):
        params.append(
            pytest.param(
                (
                    "qdrant",
                    {
                        "qdrant": {
                            "url": os.environ["RAG_IT_QDRANT_URL"],
                            "collection_prefix": f"it{uuid.uuid4().hex[:6]}_",
                        }
                    },
                ),
                id="qdrant-server",
                marks=pytest.mark.integration,
            )
        )
    if os.environ.get("RAG_IT_PG_DSN"):
        params.append(
            pytest.param(
                (
                    "pgvector",
                    {
                        "pgvector": {
                            "dsn": os.environ["RAG_IT_PG_DSN"],
                            "table_prefix": f"it{uuid.uuid4().hex[:6]}_",
                        }
                    },
                ),
                id="pgvector",
                marks=pytest.mark.integration,
            )
        )
    return params


@pytest.fixture(params=_params())
async def store(request: pytest.FixtureRequest) -> AsyncIterator[VectorStore]:
    provider, options = request.param
    s: VectorStore = VECTOR_STORES.create(provider, config=VectorStoreConfig(provider=provider, **options))
    await s.ensure_collection(
        CollectionSpec(
            name="contract",
            dense_dim=3,
            distance=DistanceMetric.COSINE,
            sparse=True,
            sparse_idf=True,
            payload_indexes=[("tenant", "keyword"), ("price", "float")],
            tenant_fields=["tenant"],  # Qdrant server: tenant-optimised keyword index
        )
    )
    try:
        yield s
    finally:
        await s.drop_collection("contract")
        await s.close()


def _id(n: int) -> str:
    return str(uuid.UUID(int=n))


def _records() -> list[VectorRecord]:
    rows = [
        (1, "a", 0, [1.0, 0.0, 0.0], {10: 1.0, 11: 0.5}, {"brand": "Delta", "cfm": 80, "tags": ["quiet"]}),
        (2, "a", 1, [0.9, 0.1, 0.0], {11: 1.0}, {"brand": "Delta", "cfm": 110, "tags": ["led"]}),
        (3, "b", 0, [0.0, 1.0, 0.0], {12: 1.0}, {"brand": "Broan", "cfm": 80, "tags": ["quiet", "led"]}),
        (4, "c", 0, [0.0, 0.0, 1.0], {13: 1.0, 10: 0.2}, {"brand": "Ninja"}),
    ]
    return [
        VectorRecord(
            id=_id(n),
            payload={"text": f"chunk {n}", "source_id": src, "chunk_index": idx, "metadata": meta},
            dense=dense,
            sparse=SparseVector(indices=list(sparse), values=list(sparse.values())),
        )
        for n, src, idx, dense, sparse, meta in rows
    ]


async def test_upsert_count_and_dense_search(store: VectorStore) -> None:
    await store.upsert("contract", _records())
    assert await store.count("contract") == 4
    hits = await store.search_dense("contract", [1.0, 0.0, 0.0], limit=2)
    assert [h.id for h in hits] == [_id(1), _id(2)]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].payload["metadata"]["brand"] == "Delta"


async def test_sparse_search_ranks_by_weight(store: VectorStore) -> None:
    await store.upsert("contract", _records())
    hits = await store.search_sparse("contract", SparseVector(indices=[11], values=[1.0]), limit=5)
    assert [h.id for h in hits] == [_id(2), _id(1)]


@pytest.mark.parametrize(
    ("flt", "expected"),
    [
        ({"brand": "Delta"}, {1, 2}),
        ({"tags": ["led"]}, {2, 3}),
        ({"cfm": {"gte": 100}}, {2}),
        ({"must_not": [{"field": "brand", "op": "eq", "value": "Delta"}]}, {3, 4}),
        ({"source_id": "a", "chunk_index": {"gte": 1}}, {2}),
        ({"must": [{"field": "cfm", "op": "exists", "value": False}]}, {4}),
        (
            {
                "should": [
                    {"field": "brand", "op": "eq", "value": "Ninja"},
                    {"field": "cfm", "op": "eq", "value": 110},
                ]
            },
            {2, 4},
        ),
    ],
)
async def test_filters_have_identical_semantics(store: VectorStore, flt: dict[str, Any], expected: set[int]) -> None:
    await store.upsert("contract", _records())
    spec = parse_filter(flt)
    hits = await store.search_dense("contract", [0.5, 0.5, 0.5], limit=10, filter_=spec)
    assert {h.id for h in hits} == {_id(n) for n in expected}
    assert await store.count("contract", spec) == len(expected)


async def test_payload_patch_and_deletes(store: VectorStore) -> None:
    await store.upsert("contract", _records())
    await store.update_payloads(
        "contract", [PayloadPatch(id=_id(1), payload={"chunk_index": 7, "metadata": {"brand": "Changed"}})]
    )
    (point,) = await store.fetch("contract", [_id(1)], with_vectors=True)
    assert point.payload["chunk_index"] == 7 and point.payload["metadata"] == {"brand": "Changed"}
    assert point.payload["text"] == "chunk 1"  # untouched keys are preserved
    assert point.dense is not None and len(point.dense) == 3

    await store.delete_ids("contract", [_id(4)])
    await store.delete_by_filter("contract", FilterSpec.for_source("a"))
    remaining, _ = await store.scroll("contract", limit=10)
    assert {p.id for p in remaining} == {_id(3)}


async def test_scroll_paginates(store: VectorStore) -> None:
    await store.upsert("contract", _records())
    first, offset = await store.scroll("contract", limit=2)
    assert len(first) == 2 and offset is not None
    second, _ = await store.scroll("contract", limit=10, offset=offset)
    assert {p.id for p in first} | {p.id for p in second} == {r.id for r in _records()}


async def test_native_hybrid_or_explicit_unsupported(store: VectorStore) -> None:
    await store.upsert("contract", _records())
    try:
        hits = await store.search_hybrid(
            "contract",
            [1.0, 0.0, 0.0],
            SparseVector([11], [1.0]),
            limit=3,
            candidate_k=10,
            fusion=FusionConfig(),
        )
    except CapabilityNotSupportedError:
        assert not store.capabilities.native_hybrid
        return
    assert store.capabilities.native_hybrid
    assert {h.id for h in hits[:2]} == {_id(1), _id(2)}


async def test_upsert_is_idempotent(store: VectorStore) -> None:
    await store.upsert("contract", _records())
    await store.upsert("contract", _records())
    assert await store.count("contract") == 4
