# 02 · E-commerce — similar product names (Amazon-style search)

```bash
uv run python cookbook/run.py ecommerce -v                 # offline: mechanics in seconds
uv run python cookbook/run.py ecommerce --stack models -v  # BGE + SPLADE + cross-encoder
```

## The problem

Catalog search fails in ways document search doesn't:

| Trap | Example in `data/catalog.json` |
|---|---|
| Names that differ by one word | Nimbus X9 · X9 **Pro** · X9 **Pro Max** · X9 **Lite** · X8 Pro |
| Model numbers that differ by one letter | AirFlo AF-80Q · AF-80Q**H** (humidity) · AF-80Q**L** (light) · AF-110Q |
| The same model in several variants | X9 Pro in 256 GB Graphite and 512 GB Silver; QC700 in Black and Silver |
| Accessories that name the product | "Vela Clear Case for Nimbus **X9 Pro Max**" mentions the phone more often than the phone does |
| Shoppers use different words | "earphones for jogging", "stop the mirror fogging up after a shower" |

The 26 SKUs from five fictional brands are synthetic, built to contain exactly these traps.

## How the pieces solve it

**Index one SKU per document, one chunk per SKU.** A product page is small, so the title, specs and
description stay together:

```yaml
chunking: {strategy: markdown, chunk_size: 256, chunk_overlap: 0}
enrichment:
  contextual_header: {template: "{title}\n\n{text}"}   # the product name is embedded twice
```

**Give every SKU its facets as metadata**, and index the ones you filter on: `brand`, `category`,
`family`, `price`, `rating`, `in_stock`. Facets become filters:
`{"category": ["headphones", "earbuds"], "price": {"lte": 250}, "in_stock": true}`.

**Hybrid retrieval: exact tokens and meaning together.**

* The sparse side (BM25, or SPLADE with real models) separates "Pro" from "Pro Max" and "AF-80Q"
  from "AF-80QH". Those are exact-token differences.
* The dense side catches "earphones for jogging" → *open-ear earbuds for running*.
* RRF fuses the two ranked lists.

**Write down the spellings shoppers type.** The lexical tokenizer splits "AF-80QH" into `af` + `80qh`, so a
shopper typing `af80qh` matches nothing lexically. Each fan's specs therefore say
`Model: AF-80QH (also written AF80QH)`. Real catalogs keep such aliases in a search-keywords field.

**Collapse variants with `group_by`.** Shoppers want one tile per product, not three colours of the
same headphones:

```yaml
retrieval:
  group_by: family      # keeps the best-scoring SKU of each family
  group_size: 1
```

A variant picker (or "show all sizes") turns it off per request: `"retrieval": {"group_by": ""}`.

**Departments are filters.** Shoppers browsing *Phones* see phones only:
`"filter": {"category": "phone"}`.

## What the checks prove

| Check | offline | models |
|---|---|---|
| "nimbus x9 pro max" → rank 1 | **the case** for the X9 Pro Max; the phone is #3 | the phone (the cross-encoder reads the whole title) |
| …with `filter: {category: phone}` | the phone | the phone |
| "Nimbus X9 Pro 512 GB Silver" → that exact SKU | ✔ | ✔ |
| "AF-80QH" and "af80qh" → the humidity model, not AF-80Q | ✔ | ✔ |
| "case for nimbus x9 pro max" → the Pro **Max** case | ✔ | ✔ |
| facets, out-of-stock exclusion, one result per family, variant picker | ✔ | ✔ |

### The accessory trap, and the three ways out

A short accessory title that repeats the product name beats the product on pure term statistics.
That's the offline result above, and it happens with BM25 in production too. The three ways out:

1. **A cross-encoder re-ranker** reads query and title together and puts the device first (the
   models stack does this).
2. **Department / category filters** when the shopper is in a department, or when your query
   classifier decides the intent is "device".
3. **Business boosts or behavioural signals.** Amazon ranks with clicks and sales. This service has
   no boosting today; apply such signals in your storefront, or ask for a `boosts` feature.

### Measurements (nDCG@5, rerank off unless stated)

| Variant | offline exact (9) | offline shopper (11) | models exact (9) | models shopper (11) |
|---|---|---|---|---|
| dense | 0.88 | 0.53 | 0.95 | 0.78 |
| sparse | **0.92** | **0.74** | 0.92 | **0.90** |
| hybrid 1:1 | 0.90 | 0.68 | **0.98** | 0.87 |
| hybrid sparse×1.5 | 0.91 | 0.68 | 0.94 | 0.88 |
| hybrid + cross-encoder | – | – | – | **0.93** (+85 ms p50) |

The eval sets are `eval-exact.json` (names, model numbers, variants) and `eval-natural.json`
(shopper language).

What the numbers say:

* With real models, **hybrid 1:1 is the most robust choice**: best on exact names, close to the
  best on shopper language.
* The sparse×1.5 weighting I expected to help did not, so the recipe ships 1:1.
* The cross-encoder adds the most of any single change (+0.06 nDCG) and fixes the accessory trap.
* SPLADE alone is strong on shopper language here. It is a *learned* sparse model that expands
  terms ("jogging" → "running").

These sets are small (9 and 11 queries), so treat differences under ~0.03 as noise. **Measure on
your own queries** before changing weights: copy the eval files, label 50–200 real search-log
queries, and compare variants with `POST /collections/shop/evaluate`. No restart is needed.

## Production notes

* **Scale.** One chunk per SKU means a million SKUs is a million points. Use Qdrant with
  `quantization: scalar`, and index every facet you filter on.
* **Price and stock change constantly.** Update them with `PATCH /documents/{sku}/metadata` (or
  `PUT` with the same content). That changes payload only and never re-embeds.
* **New variants.** Ingest them as new SKUs with the same `family`, and grouping picks them up.
* **Assistant answers.** The system prompt tells the LLM to name exact model, storage and colour.
  Run with `--llm anthropic` (or `claude-lite` for local use) to see generated answers instead of
  extracts.
