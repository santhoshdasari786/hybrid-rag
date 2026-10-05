# 03 · A CMS with many sites

```bash
uv run python cookbook/run.py cms -v
uv run python cookbook/run.py cms --stack qdrant     # real Qdrant: creates the tenant index (make it-up starts one)
```

## The problem

A CMS hosts hundreds or thousands of sites, and visitors of each site must get answers from that site
only. Here, three of the four sites have a "reset your password" page, and a blog post on a fourth
also talks about password resets. Sites have locales. Editors publish, edit, move and unpublish
pages all day, and sometimes a whole site is removed.

The data has four fictional sites and 18 pages:

| Site | Content | Format |
|---|---|---|
| `northwind-bank` | support pages in **en + de**, fees, FAQ | HTML (with nav, footer and scripts the parser drops) and Markdown |
| `contoso-blog` | articles, one about rebuilding password reset | Markdown |
| `fabrikam-docs` | product docs, incl. resetting the admin password | Markdown |
| `tailspin-travel` | travel help in **en + fr** | HTML |

## Choosing a tenancy model

| | One shared collection + `site` filter (**this recipe**) | One collection per site | Hybrid |
|---|---|---|---|
| Scales to | thousands of sites, millions of pages | tens to low hundreds of sites | many sites, a few large ones |
| Per-site tuning | no: one pipeline for all | yes: own chunking, prompt, models | yes, for the big ones |
| Isolation | the **caller** adds `filter: {site: …}` | API keys can be limited to collections | both |
| Remove a site | delete its page IDs | `DELETE /collections/{site}` | either |
| Operational cost | one index | an index per site (HNSW graph, segments, tables); thousands of them cost memory, file handles and start-up time | moderate |

Shared is the default for a CMS. It is efficient when the store knows the partition key:

```yaml
payload_indexes:
  - {field: site, tenant: true}   # Qdrant: tenant-optimised keyword index (is_tenant) — co-locates each site's points
  - {field: locale}
  - {field: content_type}
  - {field: area}
  - {field: published_at, type: datetime}
```

`tenant: true` maps to Qdrant's `is_tenant` index, which stores each site's points together and makes
filtered search per site fast. The `qdrant` stack run verifies it was created. pgvector and the
memory store ignore the flag; pgvector's GIN index on the payload already serves the filter.

### Security: who adds the site filter?

In the shared model, **your CMS backend** calls this service and adds `filter: {"site": <the
visitor's site>}` to every `/retrieve` and `/chat` request. Visitors never call the service directly.
API keys enforce *which collections* a key may use and *what* it may do (`query`, `ingest`, `admin`).
They don't enforce a per-key metadata filter. So:

* **Trusted backend** (the usual CMS setup): a `cms-backend` key with `ingest` + `query` on `cms`.
  The backend derives `site` from its own routing and auth, never from user input.
* **Sites call the service directly** (untrusted): use a **collection per site** and give each site
  a key limited to its collection. See the commented `security` block in `config.yaml`.

## The configuration, line by line

```yaml
chunking: {strategy: markdown, chunk_size: 200, chunk_overlap: 30}
```
HTML is converted to Markdown first. `<nav>`, `<footer>`, `<script>` and `<head>` are dropped, so
menus and cookie banners never become chunks. Headings become chunk boundaries and a heading path
(`Account fees and charges > International transfers`) is stored on each chunk.

```yaml
enrichment: {contextual_header: {template: "{title}\n{section}\n\n{text}"}}
```
A chunk that says "costs 0.5% of the amount" is embedded together with "Account fees and charges >
International transfers", so it is found by "international transfer fee".

```yaml
retrieval: {group_by: source_id, group_size: 2, context_expansion: {enabled: true, window: 1}}
```
At most two chunks per page, so one long page cannot fill the result list. Each hit also carries
its neighbouring chunks, which is what the LLM reads.

`area` is the CMS's own section tree. Don't call it `section`: the Markdown chunker writes the
heading path into `section`.

### `source_id` conventions

Use the CMS's **stable content ID**, prefixed with the site: `northwind-bank:page-103`. IDs may
contain letters, digits and `_ - . : @ = + ~`, but not `/`, so keep URLs in metadata (`url`), not in
the ID. A stable ID is what makes every update incremental.

## The publishing workflow (CMS event → API call)

| CMS event | Call | What happens |
|---|---|---|
| Site onboarded / nightly backfill | `POST /collections/cms/documents` with `"options": {"mode": "async"}` (+ `callback_url`) | background job; poll `GET /jobs/{id}` |
| Page published or edited | `PUT /collections/cms/documents/{site}:{page}` | unchanged chunks are kept; only changed sections are re-embedded |
| Page moved, tags or area changed | `PATCH …/documents/{id}/metadata` | payload update only, **no re-embedding** |
| Page unpublished | `DELETE …/documents/{id}` | every chunk removed |
| Site deleted | `POST …/documents:delete {"source_ids": [...]}` (up to 10,000 per call) | the CMS knows its page IDs |

The scenario plays each row. Editing one section of the four-chunk fees page re-embedded **one**
chunk and kept three.

## What the checks prove

| Check | Result |
|---|---|
| "how do I reset my password" without a filter | pages from ≥ 3 sites compete. This is why the filter is mandatory |
| …with `{site: northwind-bank, locale: en}` | only that site, its reset page in the top 2 |
| "Passwort vergessen" with `locale: de` | the German page |
| FAQ facet (`content_type: faq`) | FAQ chunks only |
| `group_by: source_id`, `group_size: 2` | never more than two chunks of a page |
| edit / move / unpublish / remove a site | as in the table above, each verified by a search |
| site-scoped chat | every citation comes from `fabrikam-docs` |
| 14 labelled site-scoped queries, nDCG@5 | offline 0.91 · models 0.99 |

The evaluation is high on every retriever because the site filter shrinks the candidate set to a
handful of pages. With a few thousand pages per site, measure again with your own queries.

## Scaling notes

* **Thousands of sites, one collection.** Qdrant with `{field: site, tenant: true}`. Add
  `quantization: scalar` when memory matters, and keep `candidate_k` moderate (filters run inside
  the search).
* **Big backfills** use async jobs. `ingestion.worker_concurrency` and `embed_concurrency` bound the
  load; a `callback_url` (allow-listed in `ingestion.callbacks.allowed_hosts`) tells the CMS when a
  job is done.
* **Locales.** BGE-small is English-first. For de/fr content in production, use a multilingual dense
  model such as `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d) or
  `intfloat/multilingual-e5-large` (1024-d), always filter by `locale`, and reindex with
  `?recreate=true` after the model change.
* **Fresh content first.** `published_at` is indexed as a datetime, so a "recent articles" widget can
  filter `{"published_at": {"gte": "2026-06-01T00:00:00Z"}}` (a scenario step checks it on every
  store). Ranking boosts by recency are not built in.
