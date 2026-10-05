# ADR 0003 — SQL source registry and deterministic chunk IDs

* Status: accepted · Date: 2026-09-27

## Context
Clients re-send documents under a stable processing ID (`source_id`). Updates must re-embed only
what changed, deletes must remove every chunk, and configuration changes must be re-applicable
without clients re-sending content.

## Decision
* Keep a relational **source registry** (SQLite for development, PostgreSQL in production) with
  content hash, metadata hash, pipeline fingerprint, version, status, chunk refs and (optionally)
  normalised content.
* Chunk IDs are `uuid5(collection/source_id/chunk_hash/occurrence)` where `chunk_hash` covers the
  pipeline fingerprint, the embedded text and the chunk text.
* Sync = diff registry chunk refs vs new chunk IDs; write order upsert → registry → delete stale.
* Optimistic concurrency via the source `version` column; per-source async locks in-process.

## Consequences
+ Idempotent ingestion, minimal embedding cost on edits, metadata-only updates without
  re-embedding, reindex from stored content, auditable job history.
− An extra datastore to operate (can share the pgvector PostgreSQL). Stored content costs space
  (`registry.store_content` can be disabled).
