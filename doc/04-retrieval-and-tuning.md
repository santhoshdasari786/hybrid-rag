# 04 · Retrieval techniques & tuning guide

## Why hybrid

| Query | Dense (semantic) | Sparse (lexical / SPLADE) | Hybrid |
|---|---|---|---|
| `revent 80 cfm` — exact model / spec | ⚠️ returns "generic fans" | ✅ matches `ReVent`, `80`, `CFM` | ✅ |
| `fan that turns on when the shower gets steamy` — concept | ✅ understands humidity sensing | ⚠️ SPLADE expansion helps, BM25 misses | ✅ |
| `silent exhaust for small room` — mixed | ✅ silent ≈ quiet | ✅ "exhaust" + expansion | ✅✅ |

Dense cosine scores (≈0–1) and sparse dot products (0–∞) are on different scales, so they are
combined with **rank fusion**, not by adding scores.

Measured on the synthetic ESCI-style catalog in `examples/` (8 labelled queries, k=3, BGE-small +
SPLADE++ + MiniLM cross-encoder, real Qdrant; `make test-models`):

| variant | hit | recall | MRR | nDCG |
|---|---|---|---|---|
| dense | 1.00 | 0.79 | 0.88 | 0.77 |
| sparse (SPLADE) | 1.00 | 0.86 | 1.00 | 0.90 |
| hybrid (RRF) | 1.00 | 0.83 | 1.00 | 0.88 |
| hybrid + rerank | 1.00 | 0.86 | 1.00 | 0.90 |

This set is tiny and spec-heavy — use it to verify the setup, not to pick a winner. Measure on
your own labelled queries with `POST /collections/{c}/evaluate`.

## Building blocks

### Indexing
* **Dense → HNSW** (approximate nearest neighbour). Layered proximity graph: search starts in the
  sparse top layer and descends, giving ~log(N) latency. Knobs: `m` (links per node — memory vs
  recall), `ef_construct` (build quality), `ef_search` (query-time recall vs latency).
* **Sparse → inverted index**. Each vocabulary dimension maps to a posting list; only the query's
  non-zero terms are visited. With `Qdrant/bm25` or the offline `lexical` embedder, IDF is applied
  by the store (`requires_idf`); SPLADE weights are already learned and need no IDF.

### Embedding models

| Kind | Model | Dim | Notes |
|---|---|---|---|
| dense | `BAAI/bge-small-en-v1.5` | 384 | Fast CPU default; use the BGE query prefix |
| dense | `BAAI/bge-large-en-v1.5` | 1024 | Higher quality, ~3× slower (the class notebook's choice) |
| dense | `openai` provider | 256–3072 | Any OpenAI-compatible `/v1/embeddings` |
| sparse | `prithivida/Splade_PP_en_v1` | 30,522 vocab | Learned term weighting + **term expansion** (adds related terms like "software", "tool") |
| sparse | `Qdrant/bm25` | hashed | Classic BM25 (IDF in store) — tiny, exact-match only |
| sparse | `Qdrant/bm42-all-minilm-l6-v2-attentions` | vocab | Attention-weighted BM25 variant |
| sparse | `lexical` | 2^20 hashed | Offline BM25 (no download) |

### Fusion
* **RRF** (default) — `score = Σ wᵢ / (k + rankᵢ)`, `k = 60`. Score-free and robust; items ranked
  high by several retrievers rise. Lower `k` → top ranks dominate more.
* **DBSF** — each list normalised with mean ± 3σ, then weighted sum. Keeps score magnitudes.
* **RSF** — min–max normalisation, weighted sum (the classic `alpha` hybrid).
* `fusion.weights` biases toward a retriever (e.g. `sparse: 1.5` for SKU-heavy catalogs).
* `fusion.execution: native` runs fusion inside Qdrant (prefetch + RRF/DBSF) — one round trip,
  but no per-retriever debug detail, and embedded Qdrant weighs RRF differently from the server.
  Client-side fusion gives identical semantics on every store.

### Query transformation (needs an LLM)
* `multi_query` — N paraphrases, all searched and fused (RAG-Fusion). Helps vague queries; costs
  one LLM call plus N× search.
* `hyde` — the LLM writes a hypothetical answer, which is embedded for dense search while sparse
  keeps the literal query. Helps short or keyword-poor questions.
* `rewrite` — conversational question → keyword query.
* Chat also **condenses follow-ups** ("how loud is it?") into standalone queries using history.
* Transform failures fall back to the original query.

### Post-retrieval
* **Rerank** — a cross-encoder reads query and passage together; much more precise than
  bi-encoders but O(candidates). Rerank the fused top `candidates` (20–50).
* **Grouping** (`group_by`, `group_size`) — keeps the best `group_size` results per value of a field,
  after re-ranking. `group_by: source_id` stops one long document from filling the page;
  `group_by: family` collapses product variants (colours, sizes) into one hit. Pass
  `"group_by": ""` per request to see every member (a variant picker). Grouping removes results, so
  give it headroom with `candidate_k` / `rerank.candidates`. Evaluate relevance with grouping off;
  it is a presentation choice.
* **MMR** — Maximal Marginal Relevance trades relevance for diversity (`lambda_mult` 1.0 = pure
  relevance). Use when near-duplicate chunks crowd the top.
* **Context expansion** (sentence-window / small-to-big) — retrieve small precise chunks, return
  `context` with ±`window` neighbours (overlap is trimmed) for the LLM.
* **Contextual chunk headers** (default on) — title and section are prepended to the *embedded*
  text, so "It moves 80 CFM" is findable for "Panasonic WhisperCeiling airflow".
* **Contextual Retrieval** (`enrichment.llm_context`) — an LLM writes 1–2 situating sentences per
  chunk before embedding. Best quality gain for long documents; costs one LLM call per *new* chunk
  (prompt-cached document, runs only for chunks the diff says are new).

### Chunking strategies

| Strategy | Use for |
|---|---|
| `recursive` | General prose (default) |
| `markdown` | Docs/wikis with headings — adds `section` / `headers` metadata |
| `sentence` | FAQ-like or short-sentence content |
| `fixed` | Uniform windows (logs, transcripts) |
| `semantic` | Long documents mixing topics; cuts where adjacent sentences diverge (costs embeddings at ingest) |
| pre-chunked `chunks[]` | You already have the right units (Q&A pairs, product rows, tickets) |

Rules of thumb: 256–512 tokens with 10–15 % overlap; keep the chunk plus header under the dense
model's window (512 tokens for BGE).

## Tuning workflow

1. **Collect 30–200 labelled queries** with the source IDs that should be found — log real
   queries and judge them, or have domain experts write them. Include exact-match (SKUs, error
   codes) and conceptual questions.
2. **Baseline**: `POST /collections/{c}/evaluate` with the default variants (dense / sparse / hybrid).
3. **Change one thing per variant** and compare nDCG@k / MRR / recall, e.g.

```json
{
  "k": 5,
  "cases": [{"query": "revent 80 cfm", "relevant": {"fan-revent-rg80": 2, "fan-revent-rg110": 1}}],
  "variants": {
    "hybrid": {"mode": "hybrid"},
    "hybrid-sparse1.5": {"mode": "hybrid", "fusion": {"weights": {"dense": 1, "sparse": 1.5}}},
    "hybrid-dbsf": {"mode": "hybrid", "fusion": {"method": "dbsf"}},
    "hybrid+rerank": {"mode": "hybrid", "rerank": {"enabled": true, "candidates": 40}},
    "multi-query": {"mode": "hybrid", "query_transform": {"strategy": "multi_query"}}
  }
}
```

4. **Diagnose with debug**: `POST /retrieve {"include_debug": true}` returns each retriever's ranked
   list, the fused list and the reranked list, so you see *which* retriever found (or missed) a
   document.
5. **Chunking/embedding changes** need a re-index: update the collection's pipeline, then
   `POST /reindex` (or `?recreate=true` for embedding changes), then re-evaluate.
6. Promote the winner into the collection's `pipeline` config.

| Symptom | Try |
|---|---|
| Exact SKUs / codes missed | raise `fusion.weights.sparse`; use SPLADE or BM25 sparse; ensure codes are in title/metadata |
| Paraphrased questions missed | stronger dense model; `hyde` or `multi_query`; raise `weights.dense` |
| Right document, wrong chunk | smaller chunks + `context_expansion`; `markdown` strategy; rerank |
| Top results near-duplicates | `mmr.enabled` or `dedupe`; larger chunks |
| Relevant item just below the cut | raise `candidate_k`; enable rerank |
| Answers lack surrounding context | `context_expansion.window: 1-2`; raise `generation.max_context_tokens` |
| Latency too high | disable rerank or lower `candidates`; `bge-small`; `fusion.execution: native`; scalar quantization |

## Metadata filtering

Filters are applied inside each retriever (pre-filtering), so `top_k` is always filled from matching
documents. Shorthand `{"brand": "Delta", "tags": ["quiet"], "cfm": {"gte": 80}}` or the structured
`must` / `should` / `must_not` form; see [05-api-reference.md](05-api-reference.md#filters).
Declare `payload_indexes` for frequently filtered fields (Qdrant builds keyword/integer indexes;
pgvector uses a GIN index on the whole payload).

Worked, measured examples of these techniques on realistic data, including dense vs sparse vs hybrid on
near-identical product names, site-scoped multi-tenant search, relevance gates and chunker comparisons,
are in the [cookbook](../cookbook/README.md).
