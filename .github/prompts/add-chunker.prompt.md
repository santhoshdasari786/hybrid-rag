---
description: Add a new chunking strategy
agent: agent
argument-hint: "strategy name and splitting rule"
---

Add a chunking strategy named `${input:name:strategy name}` that ${input:rule:describe how it splits}.

1. Subclass `BaseChunker` in `src/rag_service/adapters/chunking/text_splitters.py` (or a new module in
   the same package). Implement `split_text` (or `split_section` if you need per-chunk metadata);
   reuse `self.pack()` and `self.hard_split()` so size limits and overlap are honoured in the
   configured `length_unit`.
2. Register: `@CHUNKERS.register("${input:name}", description="…")` with the standard factory
   signature `(*, config, length_fn, dense_embedder=None)`.
3. Strategy-specific settings come from `ChunkingConfig.options`.
4. Tests in `tests/unit/test_chunkers.py`: size limits, coverage of the input, offsets map back to the
   source text, metadata propagation, edge cases (empty text, one huge token).
5. Document in `doc/02-configuration.md` (chunking table) and `doc/04-retrieval-and-tuning.md`
   (when to use it). Run `make lint test`.
