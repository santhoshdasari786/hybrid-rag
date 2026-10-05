from __future__ import annotations

import math

import pytest

from rag_service.adapters.embedding.local import HashingDenseEmbedder, LexicalSparseEmbedder, cosine
from rag_service.adapters.fusion.mmr import mmr_select
from rag_service.core.config import FusionConfig
from rag_service.core.registry import FUSIONS
from rag_service.domain.models import RankedList, RetrievedChunk, ScoredPoint


def ranked(name: str, ids: list[str], weight: float = 1.0, scores: list[float] | None = None) -> RankedList:
    scores = scores or [1.0 / (i + 1) for i in range(len(ids))]
    return RankedList(
        name=name,
        weight=weight,
        items=[ScoredPoint(id=i, score=s, payload={}) for i, s in zip(ids, scores, strict=True)],
    )


def test_rrf_matches_class_notebook_example() -> None:
    # Same three lists as the class notebook; RRF rewards consensus.
    lists = [ranked("l1", ["A", "B", "C"]), ranked("l2", ["B", "C", "D"]), ranked("l3", ["D", "A", "E"])]
    fused = FUSIONS.create("rrf").fuse(lists, config=FusionConfig(k=60), limit=10)
    assert [f.id for f in fused] == ["A", "B", "D", "C", "E"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[-1].score == pytest.approx(1 / 63)
    assert fused[0].contributions["l1"]["rank"] == 1


def test_rrf_weights_shift_the_ranking() -> None:
    lists = [ranked("dense", ["A", "B"], weight=1.0), ranked("sparse", ["B", "A"], weight=3.0)]
    fused = FUSIONS.create("rrf").fuse(lists, config=FusionConfig(), limit=10)
    assert fused[0].id == "B"


@pytest.mark.parametrize("method", ["dbsf", "rsf"])
def test_score_based_fusions_handle_scale_mismatch(method: str) -> None:
    dense = ranked("dense", ["A", "B", "C"], scores=[0.90, 0.88, 0.10])  # cosine, bounded
    sparse = ranked("sparse", ["B", "A", "C"], scores=[40.0, 10.0, 5.0])  # unbounded dot products
    fused = FUSIONS.create(method).fuse([dense, sparse], config=FusionConfig(), limit=10)
    assert {f.id for f in fused} == {"A", "B", "C"}
    assert fused[0].id == "B"  # strong in both lists
    assert set(fused[0].contributions) == {"dense", "sparse"}


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingDenseEmbedder(dimension=128)
    a = emb._embed("quiet bathroom exhaust fan")
    b = emb._embed("quiet bathroom exhaust fan")
    assert a == b
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-5)
    assert cosine(a, emb._embed("quiet bathroom fans")) > cosine(a, emb._embed("chocolate cake recipe"))


async def test_lexical_sparse_bm25_shape() -> None:
    emb = LexicalSparseEmbedder()
    (doc,) = await emb.embed_documents(["fan fan fan the quiet"])
    weights = doc.as_dict()
    assert emb.term_index("the") not in weights  # stopword
    assert weights[emb.term_index("fan")] > weights[emb.term_index("quiet")]  # tf saturates but grows
    query = await emb.embed_query("Quiet FAN")
    assert set(query.indices) == {emb.term_index("quiet"), emb.term_index("fan")}
    assert emb.requires_idf


def test_mmr_prefers_diverse_results() -> None:
    q = [1.0, 0.0, 0.0]
    chunks = [
        RetrievedChunk("a", "s", "a", 0.9, dense=[1.0, 0.1, 0.0]),
        RetrievedChunk("b", "s", "b", 0.89, dense=[1.0, 0.11, 0.0]),  # near-duplicate of a
        RetrievedChunk("c", "s", "c", 0.7, dense=[0.7, 0.0, 0.7]),
    ]
    picked = mmr_select(q, chunks, k=2, lambda_mult=0.5)
    assert [c.chunk_id for c in picked] == ["a", "c"]
    assert "mmr" in picked[0].scores
