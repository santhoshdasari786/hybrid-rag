# 01 · Quickstart — the whole lifecycle in eight steps

```bash
uv run python cookbook/run.py quickstart -v
```

## What you learn

* How a collection is declared in configuration, and what it inherits.
* That `source_id` is your **processing ID**: re-sending it re-syncs that document.
* Why chunking by section makes edits cheap.

## The configuration

```yaml
collections:
  notes:
    payload_indexes: [{field: team}]     # index what you filter on
    pipeline:
      chunking: {strategy: markdown, chunk_size: 200, chunk_overlap: 20}
```

Everything else comes from the layers below it. `config/rag.yaml` gives hybrid retrieval (dense +
sparse, fused with RRF), five results and a citation-first prompt. The stack layer decides the
embedding models and the vector store. A collection only states what differs.

## What the scenario proves

| Step | Request | Result |
|---|---|---|
| Ingest four notes | `POST /collections/notes/documents` | four sources, six chunks |
| Search | `POST …/retrieve {"query": "how do I deploy to production"}` | the deploy checklist first |
| Filter | `… "filter": {"team": "platform"}` | only platform notes |
| Chat | `POST …/chat` | an answer with `[1]` citations |
| Edit one section | `PUT …/documents/note-deploy-checklist` | `updated`: 1 chunk added, 1 removed, 2 unchanged |
| Re-send unchanged | same `PUT` again | `unchanged`, nothing re-embedded |
| Delete | `POST …/documents:delete` | every chunk of the note is gone |

Two details:

* **Edits are cheap.** With `markdown` chunking, each heading section is its own chunk. Changing the
  *Rollback* paragraph re-embeds that one chunk. With the default chunk size the whole note would be
  a single chunk and every edit would re-embed all of it.
* **Something always comes back.** The last search ("espresso machine error E4", after the coffee
  note is deleted) still returns the closest remaining notes, because dense search always has a
  nearest neighbour. Recipe 04 shows how to turn "nothing relevant" into a fallback answer.

## Try next

* Run with real models: `--stack models`.
* Serve it and click through Swagger: `uv run python cookbook/run.py quickstart --serve`, then open
  <http://localhost:8080/docs>.
