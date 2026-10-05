# ADR 0005 — Offline-first defaults

* Status: accepted · Date: 2026-09-27

## Decision
Built-in defaults use the in-memory store, a feature-hashing dense embedder, a BM25-style lexical
sparse embedder, the SQLite registry and an extractive "LLM". Real models and databases are
enabled by configuration profiles.

## Consequences
+ The full service (including chat and evaluation) runs and is tested with no downloads, network
  or secrets; CI is fast and deterministic.
− Default retrieval quality is lexical only — production must layer a profile
  (`qdrant-hybrid.yaml`); the base config says so explicitly.
