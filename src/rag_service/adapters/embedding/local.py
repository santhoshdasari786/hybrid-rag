"""Dependency-free embedders for local development, tests and air-gapped environments.

* ``hash`` (dense): signed feature hashing of word unigrams, bigrams and character trigrams into a
  fixed-size, L2-normalised vector. Captures lexical/morphological similarity — no semantics — but
  is deterministic, fast and needs no model download.
* ``lexical`` (sparse): BM25-style term-frequency weights over hashed term IDs. The vector store
  applies IDF at query time (``requires_idf = True``), which together gives true BM25 ranking.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from rag_service.core.config import DenseEmbeddingConfig, DistanceMetric, SparseEmbeddingConfig
from rag_service.core.registry import DENSE_EMBEDDERS, SPARSE_EMBEDDERS
from rag_service.core.utils import stable_hash32
from rag_service.domain.models import SparseVector

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    return [t for t in tokens if not (drop_stopwords and t in STOPWORDS)]


class HashingDenseEmbedder:
    def __init__(
        self,
        dimension: int = 384,
        distance: DistanceMetric = DistanceMetric.COSINE,
        model_id: str = "hash-384",
    ) -> None:
        self.dimension = dimension
        self.distance = distance
        self.model_id = model_id

    def _features(self, text: str) -> dict[str, float]:
        words = tokenize(text, drop_stopwords=False)
        feats: defaultdict[str, float] = defaultdict(float)
        for w in words:
            feats[f"w:{w}"] += 1.0
            padded = f"#{w}#"
            for i in range(len(padded) - 2):
                feats[f"c:{padded[i : i + 3]}"] += 0.3
        for a, b in pairwise(words):
            feats[f"b:{a}_{b}"] += 0.5
        return feats

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for feat, weight in self._features(text).items():
            h = stable_hash32(feat)
            vec[h % self.dimension] += weight if (h >> 31) & 1 else -weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LexicalSparseEmbedder:
    """BM25 document-side weighting over hashed vocabulary; pair with an IDF-applying store."""

    requires_idf = True

    def __init__(
        self,
        *,
        model_id: str = "lexical-bm25",
        k1: float = 1.2,
        b: float = 0.75,
        avg_doc_len: float = 256.0,
        hash_space: int = 1 << 20,
        drop_stopwords: bool = True,
    ) -> None:
        self.model_id = model_id
        self.k1, self.b, self.avg_doc_len = k1, b, avg_doc_len
        self.hash_space = hash_space
        self.drop_stopwords = drop_stopwords

    def term_index(self, term: str) -> int:
        return stable_hash32(term) % self.hash_space

    def _vector(self, weights: dict[int, float]) -> SparseVector:
        items = sorted(weights.items())
        return SparseVector(indices=[i for i, _ in items], values=[round(v, 6) for _, v in items])

    def _doc(self, text: str) -> SparseVector:
        tokens = tokenize(text, drop_stopwords=self.drop_stopwords)
        if not tokens:
            return SparseVector(indices=[], values=[])
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_doc_len)
        weights: dict[int, float] = {}
        for term, tf in Counter(tokens).items():
            idx = self.term_index(term)
            weights[idx] = weights.get(idx, 0.0) + tf * (self.k1 + 1) / (tf + norm)
        return self._vector(weights)

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._doc(t) for t in texts]

    async def embed_query(self, text: str) -> SparseVector:
        weights: dict[int, float] = {}
        for term in set(tokenize(text, drop_stopwords=self.drop_stopwords)):
            weights[self.term_index(term)] = 1.0
        return self._vector(weights)


@DENSE_EMBEDDERS.register("hash", description="Offline feature-hashing embedder (tests / local dev)")
def _hash(*, config: DenseEmbeddingConfig) -> HashingDenseEmbedder:
    dim = config.dimension or _dim_from_model(config.model) or 384
    return HashingDenseEmbedder(dimension=dim, distance=config.distance, model_id=f"hash-{dim}")


def _dim_from_model(model: str) -> int | None:
    match = re.search(r"(\d+)$", model)
    return int(match.group(1)) if match else None


@SPARSE_EMBEDDERS.register("lexical", aliases=("bm25-local",), description="Offline BM25-style sparse vectors")
def _lexical(*, config: SparseEmbeddingConfig) -> LexicalSparseEmbedder:
    opts: dict[str, Any] = dict(config.options)
    return LexicalSparseEmbedder(
        model_id=config.model,
        k1=float(opts.get("k1", 1.2)),
        b=float(opts.get("b", 0.75)),
        avg_doc_len=float(opts.get("avg_doc_len", 256.0)),
        hash_space=int(opts.get("hash_space", 1 << 20)),
        drop_stopwords=bool(opts.get("drop_stopwords", True)),
    )


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
