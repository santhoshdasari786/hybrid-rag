# ADR 0002 — Client-side rank fusion by default

* Status: accepted · Date: 2026-09-27

## Context
Qdrant can fuse dense and sparse results natively (prefetch + RRF/DBSF); pgvector and the memory
store cannot. Embedded Qdrant also weighs RRF differently from the server.

## Decision
Run dense and sparse searches separately and fuse in the service (`fusion.execution: client`,
default). Native fusion is opt-in (`execution: native`) for Qdrant when a single query variant is
used.

## Consequences
+ Identical ranking semantics on every store; per-retriever ranks and scores are available for
  debugging (`include_debug`) and evaluation; multi-query (RAG-Fusion) works everywhere.
− Two round trips instead of one on Qdrant; native mode exists for latency-critical cases.
