"""Chunking strategies.

All chunkers share :class:`BaseChunker`, which handles per-section iteration, character offsets,
section metadata propagation and merging of undersized chunks. Subclasses only implement
``split_text``.

=============  ==========================================================================
strategy       behaviour
=============  ==========================================================================
fixed          Sliding window over words with overlap (hard-splits over-long words).
recursive      Split on the highest-priority separator that fits, recurse on oversized
               pieces, then pack pieces with overlap (paragraph → line → sentence → word).
sentence       Packs whole sentences up to ``chunk_size`` with sentence overlap.
markdown       Splits on Markdown headers (code-fence aware), records the header path in
               ``metadata.section``, then applies the recursive splitter inside sections.
semantic       Embeds sentence windows and cuts where the cosine distance between
               neighbours exceeds the configured percentile (Kamradt-style).
=============  ==========================================================================
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from rag_service.core.config import ChunkingConfig
from rag_service.core.errors import ConfigurationError
from rag_service.core.registry import CHUNKERS
from rag_service.domain.models import ChunkDraft, ParsedDocument
from rag_service.domain.ports import DenseEmbedder, LengthFunction

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])|\n{2,}")
_WORDS = re.compile(r"\S+\s*")


class BaseChunker:
    name = "base"

    def __init__(self, config: ChunkingConfig, length_fn: LengthFunction) -> None:
        self.config = config
        self.length = length_fn
        self.size = config.chunk_size
        self.overlap = config.chunk_overlap

    # ------------------------------------------------------------------ template method

    async def split(self, document: ParsedDocument) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for section in document.sections:
            if not section.text.strip():
                continue
            pieces = await self.split_section(section.text, section.metadata)
            drafts.extend(self._with_offsets(section.text, pieces, section.metadata))
        return self._merge_small(drafts)

    async def split_section(self, text: str, metadata: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        return [(piece, {}) for piece in self.split_text(text)]

    def split_text(self, text: str) -> list[str]:
        raise NotImplementedError

    # ------------------------------------------------------------------ shared helpers

    def _with_offsets(
        self, source: str, pieces: list[tuple[str, dict[str, Any]]], base_meta: dict[str, Any]
    ) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        search_from = (
            0  # chunks advance monotonically; overlap means the next one may start earlier than the previous end
        )
        for raw, extra in pieces:
            text = raw.strip()
            if not text:
                continue
            start = source.find(text, search_from)
            meta = {**base_meta, **extra}
            if start >= 0:
                meta["start_char"] = start
                meta["end_char"] = start + len(text)
                search_from = start + 1
            drafts.append(ChunkDraft(text=text, metadata=meta))
        return drafts

    def _merge_small(self, drafts: list[ChunkDraft]) -> list[ChunkDraft]:
        minimum = self.config.min_chunk_size
        if minimum <= 0 or len(drafts) < 2:
            return drafts
        merged: list[ChunkDraft] = []
        for draft in drafts:
            if (
                merged
                and self.length(draft.text) < minimum
                and self.length(merged[-1].text) + self.length(draft.text) <= self.size
                and merged[-1].metadata.get("section") == draft.metadata.get("section")
            ):
                prev = merged[-1]
                prev.text = f"{prev.text}\n{draft.text}"
                if "end_char" in draft.metadata:
                    prev.metadata["end_char"] = draft.metadata["end_char"]
            else:
                merged.append(draft)
        # A tiny leading chunk has no predecessor: merge it forward instead.
        if (
            len(merged) > 1
            and self.length(merged[0].text) < minimum
            and self.length(merged[0].text) + self.length(merged[1].text) <= self.size
            and merged[0].metadata.get("section") == merged[1].metadata.get("section")
        ):
            first, second = merged[0], merged[1]
            second.text = f"{first.text}\n{second.text}"
            if "start_char" in first.metadata:
                second.metadata["start_char"] = first.metadata["start_char"]
            merged = merged[1:]
        return merged

    def pack(self, pieces: Sequence[str]) -> list[str]:
        """Greedily pack pieces into chunks of at most ``size`` units with ``overlap`` carried over."""
        chunks: list[str] = []
        window: list[str] = []
        lengths: list[int] = []
        total = 0
        for piece in pieces:
            plen = self.length(piece)
            if plen > self.size:
                if window:
                    chunks.append("".join(window))
                    window, lengths, total = [], [], 0
                chunks.extend(self.hard_split(piece))
                continue
            if window and total + plen > self.size:
                chunks.append("".join(window))
                while window and (total > self.overlap or total + plen > self.size):
                    total -= lengths.pop(0)
                    window.pop(0)
            window.append(piece)
            lengths.append(plen)
            total += plen
        if window:
            chunks.append("".join(window))
        return [c for c in chunks if c.strip()]

    def hard_split(self, text: str) -> list[str]:
        """Split text with no usable separators into windows that fit (binary search on length)."""
        out: list[str] = []
        start = 0
        while start < len(text):
            lo, hi = 1, len(text) - start
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.length(text[start : start + mid]) <= self.size:
                    lo = mid
                else:
                    hi = mid - 1
            out.append(text[start : start + lo])
            if start + lo >= len(text):
                break
            # Step back to create overlap, measured in the same unit.
            back = 0
            if self.overlap:
                while back < lo - 1 and self.length(text[start + lo - back - 1 : start + lo]) <= self.overlap:
                    back += 1
            start += max(1, lo - back)
        return out


class FixedSizeChunker(BaseChunker):
    name = "fixed"

    def split_text(self, text: str) -> list[str]:
        return self.pack(_WORDS.findall(text))


class RecursiveChunker(BaseChunker):
    name = "recursive"

    def __init__(self, config: ChunkingConfig, length_fn: LengthFunction) -> None:
        super().__init__(config, length_fn)
        self.separators = config.separators or DEFAULT_SEPARATORS

    def split_text(self, text: str) -> list[str]:
        return self.pack(self._pieces(text, self.separators))

    def _pieces(self, text: str, separators: list[str]) -> list[str]:
        if self.length(text) <= self.size:
            return [text]
        separator, rest = "", []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator, rest = sep, separators[i + 1 :]
                break
        if separator == "":
            return self.hard_split(text)
        parts = text.split(separator)
        # Keep the separator attached so the original text is reproducible.
        pieces = [p + separator for p in parts[:-1]] + [parts[-1]]
        out: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            if self.length(piece) <= self.size:
                out.append(piece)
            else:
                out.extend(self._pieces(piece, rest) if rest else self.hard_split(piece))
        return out


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    last = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        parts.append(text[last : match.end()])
        last = match.end()
    parts.append(text[last:])
    return [p for p in parts if p.strip()]


class SentenceChunker(BaseChunker):
    name = "sentence"

    def __init__(self, config: ChunkingConfig, length_fn: LengthFunction) -> None:
        super().__init__(config, length_fn)
        self._fallback = RecursiveChunker(config, length_fn)

    def split_text(self, text: str) -> list[str]:
        sentences: list[str] = []
        for sentence in split_sentences(text):
            if self.length(sentence) > self.size:
                sentences.extend(self._fallback.split_text(sentence))
            else:
                sentences.append(sentence)
        return self.pack(sentences)


_HEADER = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


class MarkdownChunker(BaseChunker):
    name = "markdown"

    def __init__(self, config: ChunkingConfig, length_fn: LengthFunction) -> None:
        super().__init__(config, length_fn)
        self.levels = set(config.markdown_header_levels)
        self._inner = RecursiveChunker(config, length_fn)

    async def split_section(self, text: str, metadata: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []
        for body, headers in self._sections(text):
            extra: dict[str, Any] = {}
            if headers:
                extra["section"] = " > ".join(title for _, title in headers)
                extra["headers"] = {f"h{level}": title for level, title in headers}
            out.extend((piece, extra) for piece in self._inner.split_text(body))
        return out

    def _sections(self, text: str) -> list[tuple[str, list[tuple[int, str]]]]:
        sections: list[tuple[str, list[tuple[int, str]]]] = []
        stack: list[tuple[int, str]] = []
        buffer: list[str] = []
        in_fence = False

        def flush() -> None:
            body = "\n".join(buffer).strip()
            if body:
                sections.append((body, list(stack)))
            buffer.clear()

        for line in text.split("\n"):
            if _FENCE.match(line):
                in_fence = not in_fence
            match = None if in_fence else _HEADER.match(line)
            if match and len(match.group(1)) in self.levels:
                flush()
                level, title = len(match.group(1)), match.group(2).strip()
                stack = [(lvl, t) for lvl, t in stack if lvl < level]
                stack.append((level, title))
            buffer.append(line)
        flush()
        return sections


class SemanticChunker(BaseChunker):
    name = "semantic"

    def __init__(self, config: ChunkingConfig, length_fn: LengthFunction, embedder: DenseEmbedder) -> None:
        super().__init__(config, length_fn)
        self.embedder = embedder
        self.percentile = config.semantic.breakpoint_percentile
        self.buffer = config.semantic.buffer_size
        self._fallback = RecursiveChunker(config, length_fn)

    async def split_section(self, text: str, metadata: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        sentences = split_sentences(text)
        if len(sentences) < 3:
            return [(p, {}) for p in self._fallback.split_text(text)]
        windows = ["".join(sentences[max(0, i - self.buffer) : i + self.buffer + 1]) for i in range(len(sentences))]
        vectors = np.asarray(await self.embedder.embed_documents(windows), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit = vectors / np.where(norms == 0, 1, norms)
        distances = 1.0 - np.sum(unit[:-1] * unit[1:], axis=1)
        threshold = float(np.percentile(distances, self.percentile))
        groups: list[list[str]] = [[sentences[0]]]
        for sentence, distance in zip(sentences[1:], distances, strict=True):
            if distance > threshold:
                groups.append([sentence])
            else:
                groups[-1].append(sentence)
        out: list[tuple[str, dict[str, Any]]] = []
        for group in groups:
            joined = "".join(group)
            if self.length(joined) > self.size:
                out.extend((p, {}) for p in self._fallback.split_text(joined))
            else:
                out.append((joined, {}))
        return out


# ------------------------------------------------------------------------------ registration


@CHUNKERS.register("fixed", description="Fixed-size word windows with overlap")
def _fixed(
    *, config: ChunkingConfig, length_fn: LengthFunction, dense_embedder: DenseEmbedder | None = None
) -> FixedSizeChunker:
    return FixedSizeChunker(config, length_fn)


@CHUNKERS.register("recursive", description="Recursive separator splitting (paragraph → sentence → word)")
def _recursive(
    *, config: ChunkingConfig, length_fn: LengthFunction, dense_embedder: DenseEmbedder | None = None
) -> RecursiveChunker:
    return RecursiveChunker(config, length_fn)


@CHUNKERS.register("sentence", description="Sentence packing with overlap")
def _sentence(
    *, config: ChunkingConfig, length_fn: LengthFunction, dense_embedder: DenseEmbedder | None = None
) -> SentenceChunker:
    return SentenceChunker(config, length_fn)


@CHUNKERS.register("markdown", description="Markdown header-aware sections + recursive splitting")
def _markdown(
    *, config: ChunkingConfig, length_fn: LengthFunction, dense_embedder: DenseEmbedder | None = None
) -> MarkdownChunker:
    return MarkdownChunker(config, length_fn)


@CHUNKERS.register("semantic", description="Embedding-similarity breakpoints between sentences")
def _semantic(
    *, config: ChunkingConfig, length_fn: LengthFunction, dense_embedder: DenseEmbedder | None = None
) -> SemanticChunker:
    if dense_embedder is None:
        raise ConfigurationError("The semantic chunker requires a dense embedding model (embedding.dense).")
    return SemanticChunker(config, length_fn, dense_embedder)
