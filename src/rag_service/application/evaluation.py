"""Offline retrieval evaluation against labelled queries.

Run the same labelled queries through several retrieval *variants* (each a retrieval-config
override, e.g. ``{"mode": "dense"}`` vs ``{"mode": "hybrid", "rerank": {"enabled": true}}``) and
compare standard IR metrics:

* **Hit rate@k**  — share of queries with at least one relevant result in the top k.
* **Recall@k**    — share of relevant items found in the top k.
* **Precision@k** — share of the top k that is relevant.
* **MRR**         — mean reciprocal rank of the first relevant result.
* **nDCG@k**      — rank-aware graded relevance (uses the grades you supply, default 1).

Relevance is judged at *source* level (``relevant`` holds source IDs) or *chunk* level
(``level="chunk"``). Source-level results are de-duplicated by source, first occurrence wins.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from rag_service.application.retrieval import RetrievalService
from rag_service.core.errors import ValidationFailedError
from rag_service.domain.filters import FilterSpec

DEFAULT_VARIANTS: dict[str, dict[str, Any]] = {
    "dense": {"mode": "dense"},
    "sparse": {"mode": "sparse"},
    "hybrid": {"mode": "hybrid"},
}


@dataclass(slots=True)
class EvalCase:
    query: str
    relevant: dict[str, float]
    filter: FilterSpec | None = None


@dataclass(slots=True)
class CaseResult:
    query: str
    retrieved: list[str]
    first_relevant_rank: int | None
    recall: float
    precision: float
    ndcg: float


@dataclass(slots=True)
class VariantReport:
    name: str
    overrides: dict[str, Any]
    metrics: dict[str, float]
    cases: list[CaseResult] = field(default_factory=list)


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


class EvaluationService:
    def __init__(self, retrieval: RetrievalService) -> None:
        self.retrieval = retrieval

    async def evaluate(
        self,
        collection: str,
        cases: list[EvalCase],
        *,
        k: int = 5,
        level: Literal["source", "chunk"] = "source",
        variants: dict[str, dict[str, Any]] | None = None,
        include_cases: bool = False,
    ) -> list[VariantReport]:
        if not cases:
            raise ValidationFailedError("At least one evaluation case is required")
        reports = []
        for name, overrides in (variants or DEFAULT_VARIANTS).items():
            # Retrieve more than k chunks so source-level de-duplication can still fill k slots.
            fetch_k = k * 4 if level == "source" else k
            merged = {**overrides, "top_k": min(100, max(fetch_k, overrides.get("top_k", 0)))}
            results: list[CaseResult] = []
            latencies: list[float] = []
            for case in cases:
                started = time.perf_counter()
                retrieval = await self.retrieval.retrieve(collection, case.query, filter_=case.filter, overrides=merged)
                latencies.append((time.perf_counter() - started) * 1000)
                ids: list[str] = []
                for chunk in retrieval.chunks:
                    key = chunk.source_id if level == "source" else chunk.chunk_id
                    if key not in ids:
                        ids.append(key)
                results.append(self._score(case, ids[:k], k))
            reports.append(
                VariantReport(
                    name=name,
                    overrides=overrides,
                    metrics=self._aggregate(results, latencies),
                    cases=results if include_cases else [],
                )
            )
        return reports

    @staticmethod
    def _score(case: EvalCase, retrieved: list[str], k: int) -> CaseResult:
        relevant = case.relevant
        hits = [doc for doc in retrieved if doc in relevant]
        first = next((i + 1 for i, doc in enumerate(retrieved) if doc in relevant), None)
        ideal = sorted(relevant.values(), reverse=True)[:k]
        ndcg = _dcg([relevant.get(doc, 0.0) for doc in retrieved]) / (_dcg(ideal) or 1.0)
        return CaseResult(
            query=case.query,
            retrieved=retrieved,
            first_relevant_rank=first,
            recall=len(hits) / len(relevant) if relevant else 0.0,
            precision=len(hits) / k,
            ndcg=ndcg,
        )

    @staticmethod
    def _aggregate(results: list[CaseResult], latencies: list[float]) -> dict[str, float]:
        n = len(results)
        return {
            "hit_rate": round(sum(1 for r in results if r.first_relevant_rank) / n, 4),
            "recall": round(statistics.fmean(r.recall for r in results), 4),
            "precision": round(statistics.fmean(r.precision for r in results), 4),
            "mrr": round(
                statistics.fmean(1 / r.first_relevant_rank if r.first_relevant_rank else 0.0 for r in results),
                4,
            ),
            "ndcg": round(statistics.fmean(r.ndcg for r in results), 4),
            "latency_ms_p50": round(statistics.median(latencies), 2),
            "latency_ms_max": round(max(latencies), 2),
        }
