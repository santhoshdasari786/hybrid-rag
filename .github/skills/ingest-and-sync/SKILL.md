---
name: ingest-and-sync
description: Ingest, update and delete documents in the Advanced RAG Service over HTTP using stable source IDs (processing IDs) — raw text, base64 files (PDF), pre-chunked content with per-chunk metadata, metadata-only updates, folder mirroring and troubleshooting failed sources. Use when integrating a content system or loading data.
argument-hint: "[collection] [what to ingest]"
---

# Ingest & keep sources in sync

Full semantics: `doc/03-ingestion-and-sync.md`. API: `doc/05-api-reference.md`.

## Choose the source_id first

The `source_id` is the document's stable processing ID in the upstream system (page ID, SKU, file
path). Re-sending it updates in place; deleting it removes all its chunks. Allowed characters:
`A-Z a-z 0-9 _ - . : @ = + ~` (no `/` — use `:` for hierarchy, e.g. `wiki:hr:leave-policy`).

## Recipes

| Need | Request |
|---|---|
| New or changed text/Markdown/HTML | `PUT /api/v1/collections/{c}/documents/{source_id}` with `{"content", "mime_type", "title", "metadata"}` |
| Many documents | `POST /api/v1/collections/{c}/documents` `{"documents": [...], "options": {"mode": "async"}}` → poll `Location` |
| PDF / binary | `"content_base64": "<base64>", "mime_type": "application/pdf"` (one section per page → `metadata.page`) |
| Your own chunks | `"chunks": [{"text": "...", "metadata": {...}}]` (per-chunk metadata) |
| Different chunking for one doc | `"chunking": {"strategy": "markdown", "chunk_size": 256}` |
| Metadata only (no re-embed) | `PATCH …/documents/{source_id}/metadata` `{"metadata": {...}, "merge": true}` |
| Tag one chunk | `PATCH …/chunks/{chunk_id}/metadata` (survives re-sync while the chunk text is unchanged) |
| Delete | `DELETE …/documents/{source_id}` or `POST …/documents:delete {"source_ids": [...]}` |
| Force re-embed | `"options": {"force": true}` |
| Mirror a folder (incl. deletions) | `uv run python scripts/sync_folder.py --dir <path> --collection <c> --delete-missing` |

Result `action` per source: `created`, `updated`, `unchanged`, `metadata_updated`, `deleted`,
`not_found`, `failed`. Batch status: 200 all ok, 207 partial, 422 all failed, 202 accepted (async).

## Metadata tips

- Put filterable facts in `metadata` (brand, category, language, tenant, dates as ISO strings,
  numbers as numbers). Keys: no `.` and no leading `$`.
- Declare frequently filtered fields in `collections.<c>.payload_indexes` (Qdrant).
- Filters: shorthand `{"brand": "Delta", "tags": ["quiet"], "price": {"lte": 100}}`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `failed` with `UNSUPPORTED_MEDIA_TYPE` | MIME not in `ingestion.allowed_mime_types` or no parser |
| `failed` with `VALIDATION_FAILED` "no extractable text" | empty/scanned document — OCR first |
| `409 EMBEDDING_CONFIG_CHANGED` | collection stale after an embedding change → `POST …/reindex?recreate=true` (admin) |
| `409 CONCURRENT_MODIFICATION` | two writers on one source — retry |
| Always `updated`, never `unchanged` | content differs each time (timestamps, generated IDs in text) — strip volatile parts or move them to metadata |
| Source listed as `failed` | `GET …/documents/{id}` → `error`; re-send after fixing |
