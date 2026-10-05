---
description: Evaluate and tune a collection's retrieval pipeline with labelled queries
agent: agent
argument-hint: "collection name and service URL"
---

Tune retrieval for collection `${input:collection:collection name}` on `${input:url:http://localhost:8080}`.

Use the `rag-retrieval-tuning` skill. In short:
1. Gather or draft 20+ labelled queries (`{query, relevant: [source_ids]}`) covering exact-match and
   conceptual questions; confirm them with me before running.
2. Run `POST /api/v1/collections/${input:collection}/evaluate` with a baseline (dense, sparse, hybrid).
3. Propose 3–5 single-change variants (fusion weights/method, rerank, candidate_k, query transform,
   chunking) based on the failures (`include_cases: true`, and `/retrieve` with `include_debug`).
4. Report a table of hit rate, recall, MRR, nDCG and p50 latency per variant, name the winner, and
   give the exact YAML to put under `collections.${input:collection}.pipeline`.
Do not change configuration files without my confirmation.
