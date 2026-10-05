---
name: rag-retrieval-tuning
description: Measure and improve retrieval quality of a collection in the Advanced RAG Service using labelled queries, the /evaluate and /retrieve (debug) APIs, and single-variable experiments on fusion, reranking, query transforms and chunking. Use when results are poor, when choosing between dense/sparse/hybrid, or before promoting a pipeline change.
argument-hint: "[collection] [service url]"
---

# Retrieval tuning

Background: `doc/04-retrieval-and-tuning.md` (techniques, symptoms → fixes).

## 1. Labelled queries

Create a JSON file in the format of [eval-template.json](./eval-template.json) (also
`examples/evaluate_products.json`). Guidelines:

- 20–200 real or realistic queries; include exact-match (SKUs, error codes, numbers) and
  conceptual/paraphrased questions, and a few that should return nothing relevant.
- `relevant` lists source IDs (`level: "source"`) or chunk IDs (`level: "chunk"`). Use a mapping
  (`{"id": 2, "other": 1}`) for graded relevance (feeds nDCG).
- Confirm the set with the user before drawing conclusions from it.

## 2. Baseline

```bash
curl -s -X POST "$URL/api/v1/collections/$C/evaluate" -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d @eval.json | jq '.data.variants[] | {name, metrics}'
```

With no `variants`, dense / sparse / hybrid are compared.

## 3. Diagnose

- `include_cases: true` → per-query `retrieved` and `first_relevant_rank`.
- For a failing query: `POST /retrieve` with `include_debug: true` → `debug.lists` shows each
  retriever's ranking; `debug.fused` the fused one; `debug.reranked` after the cross-encoder.
  - Found by sparse, missed by dense → raise sparse weight / keep hybrid.
  - Found by neither within `candidate_k` → chunking/embedding problem, or raise `candidate_k`.
  - Found but ranked low after fusion → try `rerank`, fusion weights, `dbsf`.

## 4. Experiments (one change per variant)

```json
"variants": {
  "hybrid": {"mode": "hybrid"},
  "sparse-heavy": {"mode": "hybrid", "fusion": {"weights": {"dense": 1.0, "sparse": 1.5}}},
  "dbsf": {"mode": "hybrid", "fusion": {"method": "dbsf"}},
  "rerank": {"mode": "hybrid", "rerank": {"enabled": true, "candidates": 40}},
  "cand100": {"mode": "hybrid", "candidate_k": 100},
  "multi-query": {"mode": "hybrid", "query_transform": {"strategy": "multi_query"}},
  "hyde": {"mode": "hybrid", "query_transform": {"strategy": "hyde"}}
}
```

Query transforms call the utility LLM per query — note the cost/latency. Chunking or embedding
changes need `PATCH /collections/{c}` + `POST /collections/{c}/reindex` (`?recreate=true` for
embedding changes); ask before re-indexing a shared collection.

## 5. Report

A table per variant: hit rate, recall, MRR, nDCG, p50/max latency. Name the winner, state the
trade-offs, flag noise (small query sets: differences < ~0.03 nDCG are not meaningful), and give
the YAML to paste under `collections.<name>.pipeline`, e.g.

```yaml
retrieval:
  fusion: {method: rrf, weights: {dense: 1.0, sparse: 1.5}}
  rerank: {enabled: true, candidates: 40}
```

`scripts/smoke.py` runs the sample evaluation end-to-end against any running service.
