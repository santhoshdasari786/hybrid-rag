# 04 · Support helpdesk — Q&A chunks, conversations and a safe fallback

```bash
uv run python cookbook/run.py helpdesk -v
uv run python cookbook/run.py helpdesk --stack models -v       # cross-encoder relevance gate
uv run python cookbook/run.py helpdesk --llm anthropic -v      # + rewriting follow-ups, multi-query
```

## The problem

A help centre for a fictional ISP, Lumen Fibre, has three kinds of content:

* **FAQs**: already one answer per question. Splitting them would break pairs apart.
* **Guides**: prose with headings ("No internet connection", "Slow Wi-Fi").
* **Curation**: support leads retire outdated answers without waiting for a re-import.

Customers ask follow-ups ("Can I take it with me?") and ask things the help centre doesn't cover.
A wrong answer is worse than "I don't know".

## Pre-chunked documents with per-chunk metadata

Send the chunks yourself and the chunker is skipped. Each chunk carries its own metadata, and
document metadata applies to all of them:

```json
{"source_id": "faq-billing", "metadata": {"area": "billing", "kind": "faq"},
 "chunks": [{"text": "Q: Why was I charged twice this month?\nA: …",
             "metadata": {"question": "Why was I charged twice this month?", "intent": "duplicate-charge"}}]}
```

Filters combine both levels: `{"area": "billing"}` (document) plus `first_metadata: {intent: refund}`
(chunk). Re-sending the FAQ re-embeds only the Q&A pairs whose text changed.

**Retire an answer without deleting it.** Use `PATCH /chunks/{id}/metadata {"metadata":
{"deprecated": true}}`, then exclude retired chunks with
`{"must_not": [{"field": "deprecated", "op": "eq", "value": true}]}`. The chunk stays for audit and
can be restored with one call.

The troubleshooting guide uses `strategy: sentence` with small chunks (90 tokens), so "router light
is red" lands on the two sentences that answer it. `context_expansion` then hands the LLM the
surrounding steps.

## Conversations

```yaml
generation:
  condense_history: true     # needs an LLM that writes
  history_turns: 6
```

With a generative LLM, the follow-up "Can I take it with me?" after "I'm moving house next month" is
rewritten to a standalone question before searching. It then finds the moving-house answer. With the
offline `extractive` LLM the rewrite is skipped (the service logs `pipeline.feature_skipped`), and
those steps only run with `--llm …`.

The same collection is also an OpenAI-compatible model, `rag:helpdesk`, so chat widgets and SDKs
that speak `/v1/chat/completions` work unchanged.

## The off-topic gate: thresholds belong to a model

A dense retriever always returns *something*. To turn "nothing relevant" into the fallback answer,
drop weak results before generation. When no chunk survives, chat returns
`finish_reason: no_context` with `generation.no_context_answer`.

The scores below are measured on this help centre for the top hit of each question:

| Signal | On-topic questions | Off-topic ("best pizza in Naples", "football world cup", "Mount Everest") |
|---|---|---|
| cross-encoder (MiniLM) score | +3.3 … +7.5 | ≈ −11 |
| BGE-small cosine | 0.74 … 0.78 | 0.29 … 0.38 |
| offline hash-embedding cosine | 0.28 … 0.40 | 0.17 … 0.21 |

* **With a cross-encoder** (the model stacks), `rerank.min_score: 0.0` separates them by a wide
  margin, so it is set in `config.yaml`. When the only relevant answer has been retired, the gate
  returns nothing rather than a near miss, which is the right behaviour for support.
* **Offline**, the scenario uses `mode: dense` with `dense_score_threshold: 0.25`. It works, but the
  margin is thin and the value is meaningless for any other embedding model.

Calibrate any threshold with `POST /retrieve {"include_debug": true}` on your own on-topic and
off-topic questions, and re-calibrate when you change models.

## What the checks prove

| Check | Result |
|---|---|
| "I was charged twice" | the `duplicate-charge` Q&A pair first |
| `area: billing` + "how do I get my money back" | the `refund` pair, billing only |
| "router light is red" | the guide's sentences about the red light |
| retire the "first bill" answer | never returned again |
| chat, OpenAI-compatible chat | answers with `[n]` citations |
| "What's the best pizza in Naples?" | `no_context` fallback (offline and models) |
| follow-up and multi-query (`--llm`) | the moving-house answer; the troubleshooting guide for "internet keeps dropping" |

Verified with Claude (`--llm claude-lite`, 2026-10-02). It answered "Can I take it with me?" with
*"Yes, you can take your Lumen Fibre service with you when you move [1]. Tell us your new address at
least 3 weeks before the move…"*, citing the moving-house FAQ.
