"""Rank-fusion strategies.

Dense cosine scores (≈0–1) and sparse dot products (0–∞) live on incompatible scales, so hybrid
search needs a fusion step:

* ``rrf``  — Reciprocal Rank Fusion (Cormack et al., 2009): ``Σ wᵢ / (k + rankᵢ)``. Score-free,
  robust default; rewards consensus across retrievers.
* ``dbsf`` — Distribution-Based Score Fusion: each list is normalised with mean ± 3σ, clipped to
  [0, 1], weighted and summed. Keeps score magnitude information.
* ``rsf``  — Relative Score Fusion: per-list min–max normalisation, weighted sum (convex
  combination when weights sum to 1; the classic ``alpha`` hybrid).

All strategies accept any number of ranked lists (e.g. several query variants × two retrievers),
which is what makes RAG-Fusion / multi-query retrieval work.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from rag_service.core.config import FusionConfig
from rag_service.core.registry import FUSIONS
from rag_service.domain.models import FusedItem, RankedList


def _weight(lst: RankedList) -> float:
    return lst.weight


def _record(out: dict[str, FusedItem], pid: str, contribution: float, name: str, rank: int, score: float) -> None:
    item = out.setdefault(pid, FusedItem(id=pid, score=0.0))
    item.score += contribution
    # Keep the best rank per retriever when the same retriever ran for several query variants.
    existing = item.contributions.get(name)
    if existing is None or rank < existing["rank"]:
        item.contributions[name] = {"rank": float(rank), "score": score}


def _sorted(out: dict[str, FusedItem], limit: int) -> list[FusedItem]:
    return sorted(out.values(), key=lambda i: (-i.score, i.id))[:limit]


class ReciprocalRankFusion:
    name = "rrf"

    def fuse(self, lists: Sequence[RankedList], *, config: FusionConfig, limit: int) -> list[FusedItem]:
        out: dict[str, FusedItem] = {}
        for lst in lists:
            for rank, point in enumerate(lst.items, start=1):
                _record(out, point.id, _weight(lst) / (config.k + rank), lst.name, rank, point.score)
        return _sorted(out, limit)


class DistributionBasedScoreFusion:
    name = "dbsf"

    def fuse(self, lists: Sequence[RankedList], *, config: FusionConfig, limit: int) -> list[FusedItem]:
        out: dict[str, FusedItem] = {}
        for lst in lists:
            scores = [p.score for p in lst.items]
            if not scores:
                continue
            mean = statistics.fmean(scores)
            std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            lo, hi = mean - 3 * std, mean + 3 * std
            for rank, point in enumerate(lst.items, start=1):
                norm = 1.0 if hi == lo else min(1.0, max(0.0, (point.score - lo) / (hi - lo)))
                _record(out, point.id, _weight(lst) * norm, lst.name, rank, point.score)
        return _sorted(out, limit)


class RelativeScoreFusion:
    name = "rsf"

    def fuse(self, lists: Sequence[RankedList], *, config: FusionConfig, limit: int) -> list[FusedItem]:
        out: dict[str, FusedItem] = {}
        for lst in lists:
            if not lst.items:
                continue
            scores = [p.score for p in lst.items]
            lo, hi = min(scores), max(scores)
            for rank, point in enumerate(lst.items, start=1):
                norm = 1.0 if hi == lo else (point.score - lo) / (hi - lo)
                _record(out, point.id, _weight(lst) * norm, lst.name, rank, point.score)
        return _sorted(out, limit)


@FUSIONS.register("rrf", aliases=("reciprocal_rank",), description="Weighted Reciprocal Rank Fusion")
def _rrf() -> ReciprocalRankFusion:
    return ReciprocalRankFusion()


@FUSIONS.register("dbsf", description="Distribution-Based Score Fusion (mean ± 3σ normalisation)")
def _dbsf() -> DistributionBasedScoreFusion:
    return DistributionBasedScoreFusion()


@FUSIONS.register("rsf", aliases=("relative_score", "linear"), description="Relative (min-max) Score Fusion")
def _rsf() -> RelativeScoreFusion:
    return RelativeScoreFusion()
