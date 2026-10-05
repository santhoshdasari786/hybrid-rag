"""Maximal Marginal Relevance (Carbonell & Goldstein, 1998) for result diversification."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag_service.domain.models import RetrievedChunk


def mmr_select(
    query_vector: Sequence[float],
    chunks: Sequence[RetrievedChunk],
    *,
    k: int,
    lambda_mult: float,
) -> list[RetrievedChunk]:
    """Greedily pick ``k`` chunks balancing query relevance against redundancy.

    Chunks without a dense vector keep their original order after the diversified ones.
    """
    with_vec = [c for c in chunks if c.dense is not None]
    without = [c for c in chunks if c.dense is None]
    if not with_vec:
        return list(chunks[:k])

    matrix = np.asarray([c.dense for c in with_vec], dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    query = np.asarray(query_vector, dtype=np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)

    relevance = matrix @ query
    selected: list[int] = []
    remaining = list(range(len(with_vec)))
    while remaining and len(selected) < k:
        redundancy = np.max(matrix[remaining] @ matrix[selected].T, axis=1) if selected else np.zeros(len(remaining))
        scores = lambda_mult * relevance[remaining] - (1 - lambda_mult) * redundancy
        pick = int(np.argmax(scores))
        best = remaining[pick]
        with_vec[best].scores["mmr"] = round(float(scores[pick]), 6)
        selected.append(best)
        remaining.remove(best)
    ordered = [with_vec[i] for i in selected]
    return (ordered + without)[:k]
