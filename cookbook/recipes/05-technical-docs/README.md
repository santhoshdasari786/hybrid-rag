# 05 · Technical documentation — identifiers, code and versions

```bash
uv run python cookbook/run.py technical -v
uv run python cookbook/run.py technical --stack models -v
```

## The problem

Developers search docs in two very different ways:

* **Identifiers**: `E1042`, `QUASAR_TIMEOUT_MS`, `client.resume`, `max_retries`. They must match
  exactly, and the hit should be the right *section*, not just the right page.
* **How-to questions**: "my upload was interrupted halfway, how do I continue it?" Here the words
  differ from the docs ("resumable uploads").

Code samples must come back whole, and "what changed" questions belong to the changelog.

The content is five Markdown pages of a fictional *Quasar SDK*: install, configuration reference,
error codes, uploads (with code) and changelog.

## The configuration

```yaml
chunking: {strategy: markdown, chunk_size: 300, chunk_overlap: 40, markdown_header_levels: [1, 2, 3]}
```
Headings become chunk boundaries, so each error code (`## E1042 — upload checksum mismatch`) is its
own chunk and its metadata `section` is the heading path. The Markdown chunker never splits inside a
fenced code block; the scenario checks that every returned chunk has balanced fences.

```yaml
enrichment: {contextual_header: {template: "{title}\n{section}\n\n{text}"}}
```
The heading path is embedded with the text. "Error codes > E1042 — upload checksum mismatch" is what
the retrievers see, so the code and its meaning travel together.

```yaml
retrieval: {group_by: source_id, group_size: 2, context_expansion: {enabled: true, window: 1}}
```
At most two sections per page. The section before and after come along, so the prose that explains
a code sample, and the sample after the prose, reach the LLM together.

Metadata `kind: guide | reference | changelog` turns "what changed?" into a filter:
`{"kind": "changelog"}`.

## What the checks prove

| Check | Result |
|---|---|
| `E1042` | the error-codes page; the rank-1 chunk *is* the E1042 section (`section` contains "E1042") |
| `QUASAR_TIMEOUT_MS` | the configuration reference |
| "upload interrupted halfway…" | the uploads guide in the top 2 |
| "python example upload a file" | chunks with complete code blocks, including the `qs.connect(` sample |
| context expansion | hits carry their neighbouring sections |
| "timeout setting renamed" + `kind: changelog` | the 2.4.0 entry (`timeout` → `timeout_ms`) |

### Measurements (nDCG@5, 12 queries: identifiers + how-to)

| Variant | offline | models |
|---|---|---|
| dense | 0.78 | 0.94 |
| sparse | **0.96** | **0.99** |
| hybrid | 0.93 | **0.99** |

Identifiers are where sparse retrieval shines: an error code is a rare, exact token. With real models
hybrid matches sparse and keeps the dense side for paraphrased questions. Offline, the hashing dense
embedder costs a little on identifier queries.

## Variations

* **Versioned docs.** Index each version with `metadata.version` (`"3.x"`, `"2.x"`) and filter by the
  version the reader selected. A `PUT` per page keeps each version in sync.
* **API references generated from code.** Send one pre-chunked document per class, with one chunk
  per method and `symbol` metadata.
* **Code-heavy corpora.** If most queries are identifiers, try `mode: sparse` or
  `fusion.weights: {dense: 1.0, sparse: 2.0}` and compare with `/evaluate`.
