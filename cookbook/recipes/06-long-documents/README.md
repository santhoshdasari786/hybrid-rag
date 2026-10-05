# 06 · Long documents — choosing a chunker and giving the LLM enough context

```bash
uv run python cookbook/run.py long -v
uv run python cookbook/run.py long --stack models -v
uv run python cookbook/run.py long --llm anthropic -v     # + LLM contextual retrieval
```

## The problem

Handbooks, policies, contracts and PDF reports arrive as long text, often without Markdown structure
(text extracted from a PDF looks like `data/handbook.json`). Retrieval returns *chunks*. If the
answer was cut in half at a chunk boundary, or sits in a chunk whose words don't match the question,
neither the search nor the LLM can recover it.

There is only one document here, so source-level evaluation would be meaningless. This recipe
measures what matters for long documents instead: **is the answer inside the chunk we return?**
`facts.json` holds 15 questions, each with the exact phrase that answers it ("16 weeks", "70 EUR",
"31 March").

## One document, three chunkers

`config.yaml` declares three collections with identical retrieval settings, shared through a YAML
anchor (`x-handbook-retrieval`):

| Collection | Chunking | Idea |
|---|---|---|
| `handbook_fixed` | `fixed`, 100 tokens, no overlap | baseline: windows that ignore sentences and sections |
| `handbook_recursive` | `recursive`, 100 tokens, 20 overlap | split on paragraphs, then sentences, then words; overlap repairs edges |
| `handbook_semantic` | `semantic`, ≤ 160 tokens, breakpoint percentile 85 | split where consecutive sentences change topic (embedding distance) |

Chunking is decided at ingestion, so comparing chunkers means comparing collections. With
`/evaluate` you can compare *retrieval* settings on one collection without re-ingesting.

## Results: answer inside the top chunk (of 15)

| | fixed | recursive | semantic | recursive + neighbours |
|---|---|---|---|---|
| offline | 12 | 14 | **15** | **15** |
| models (BGE + SPLADE + cross-encoder) | 15 | 15 | 15 | 15 |

* **Offline**, fixed windows lose three answers. The windows straddle sections: the chunk with
  "28 days" is half *Remote work* and half *Annual leave*, and the one with "three months" is half
  *Pension*. Mixed-topic chunks match neither topic well, so other chunks outrank them. Recursive
  splitting follows paragraph boundaries and loses one answer; semantic splitting loses none.
* **With real models and a cross-encoder** every chunker reaches 15/15 on this short, clean
  handbook. A strong retriever hides chunking mistakes until documents get long and messy: tables,
  headers and footers from PDFs, multi-page sections. Measure with your own documents. `facts.json`
  is an easy format to write: question plus the phrase that must be found.
* **Small-to-big.** `context_expansion: {enabled: true, window: 1}` attaches the neighbouring chunks
  to every hit. Precise small chunks are matched, and the LLM reads the larger window. That rescues
  the answer recursive chunking missed (14 → 15).

## LLM contextual retrieval

With `--llm anthropic` (or `openai`, `gemini`, `claude-lite`), the scenario creates
`handbook_contextual` with:

```yaml
enrichment: {llm_context: {enabled: true}}
```

Before embedding, an LLM reads the whole document and writes one or two sentences situating each
chunk, e.g. *"This excerpt is from the Parental leave section of the Halcyon Labs handbook"*. It is
the most effective fix for chunks that make no sense on their own ("It is replaced every three
years."). It costs one LLM call per new or changed chunk at ingestion time, never per query. With
Anthropic the document is sent as a cached system prompt, so the calls after the first are cheaper.
The service skips it, with a warning, when the LLM cannot write (offline `extractive`).

## Choosing in practice

| Content | Start with |
|---|---|
| Structured Markdown / HTML | `markdown` (recipes 03 and 05) |
| Long prose, policies, PDF text | `recursive`, 300–500 tokens, ~15% overlap, plus context expansion |
| Topic-dense prose with weak structure | try `semantic`; compare with `facts` checks |
| Chunks that need the document to make sense | add `llm_context` if the ingestion budget allows |

For broad questions over redundant collections (news, reviews, meeting notes), also enable MMR
(`retrieval.mmr: {enabled: true, lambda_mult: 0.7}`) so the top results aren't near-duplicates.
