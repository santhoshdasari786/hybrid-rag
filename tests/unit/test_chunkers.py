from __future__ import annotations

import pytest

from rag_service.adapters.chunking.length import length_function_for
from rag_service.adapters.embedding.local import HashingDenseEmbedder
from rag_service.core.config import ChunkingConfig
from rag_service.core.registry import CHUNKERS
from rag_service.domain.models import ParsedDocument, Section

TEXT = " ".join(f"Sentence number {i} talks about ventilation fans and airflow." for i in range(60))


def chunker(**cfg: object):  # type: ignore[no-untyped-def]
    config = ChunkingConfig(**cfg)  # type: ignore[arg-type]
    return CHUNKERS.create(
        config.strategy,
        config=config,
        length_fn=length_function_for(config),
        dense_embedder=HashingDenseEmbedder(64),
    )


def doc(text: str, **meta: object) -> ParsedDocument:
    return ParsedDocument(sections=[Section(text=text, metadata=dict(meta))])


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "sentence"])
async def test_chunks_respect_size_and_cover_text(strategy: str) -> None:
    c = chunker(strategy=strategy, chunk_size=40, chunk_overlap=8, length_unit="words")
    drafts = await c.split(doc(TEXT))
    assert len(drafts) > 3
    assert all(len(d.text.split()) <= 40 for d in drafts)
    # Every sentence survives chunking.
    joined = " ".join(d.text for d in drafts)
    assert "Sentence number 0 " in joined and "Sentence number 59 " in joined


async def test_overlap_repeats_boundary_text() -> None:
    c = chunker(strategy="fixed", chunk_size=20, chunk_overlap=5, length_unit="words")
    drafts = await c.split(doc(TEXT))
    first_tail = drafts[0].text.split()[-3:]
    assert " ".join(first_tail) in drafts[1].text


async def test_offsets_point_into_source() -> None:
    c = chunker(strategy="recursive", chunk_size=30, chunk_overlap=0, length_unit="words")
    drafts = await c.split(doc(TEXT))
    for d in drafts:
        start, end = d.metadata["start_char"], d.metadata["end_char"]
        assert TEXT[start:end] == d.text


async def test_markdown_records_header_path() -> None:
    md = "# Fans\n\nIntro text.\n\n## Installation\n\nMount it.\n\n```\n# not a header\n```\n\n## Noise\n\nQuiet."
    drafts = await chunker(strategy="markdown", chunk_size=200, chunk_overlap=0, length_unit="words").split(doc(md))
    sections = [d.metadata.get("section") for d in drafts]
    assert sections == ["Fans", "Fans > Installation", "Fans > Noise"]
    assert "# not a header" in drafts[1].text  # fenced code is not split
    assert drafts[1].metadata["headers"] == {"h1": "Fans", "h2": "Installation"}


async def test_section_metadata_propagates() -> None:
    drafts = await chunker(chunk_size=100, chunk_overlap=0, length_unit="words").split(doc("Page one text.", page=3))
    assert drafts[0].metadata["page"] == 3


async def test_hard_split_for_unbreakable_text() -> None:
    drafts = await chunker(strategy="recursive", chunk_size=50, chunk_overlap=10, length_unit="chars").split(
        doc("x" * 400)
    )
    assert all(len(d.text) <= 50 for d in drafts)
    assert sum(len(d.text) for d in drafts) >= 400


async def test_min_chunk_size_merges_tiny_chunks() -> None:
    parsed = ParsedDocument(
        sections=[
            Section(text="Tiny."),
            Section(text="This paragraph has exactly twelve words in it for the test."),
            Section(text="End."),
        ]
    )
    drafts = await chunker(
        strategy="recursive", chunk_size=16, chunk_overlap=0, length_unit="words", min_chunk_size=3
    ).split(parsed)
    assert len(drafts) == 1
    assert drafts[0].text.startswith("Tiny.") and drafts[0].text.endswith("End.")


async def test_semantic_chunker_splits_topics() -> None:
    text = (
        "Bathroom fans move air. Exhaust fans remove humidity. Fans are rated in CFM. "
        "Blenders crush ice. Kitchen blenders make smoothies. A blender has sharp blades."
    )
    drafts = await chunker(strategy="semantic", chunk_size=200, chunk_overlap=0, length_unit="words").split(doc(text))
    assert len(drafts) >= 2
    assert "Blenders" in "".join(d.text for d in drafts)


def test_semantic_requires_dense_embedder() -> None:
    from rag_service.core.errors import ConfigurationError

    config = ChunkingConfig(strategy="semantic")
    with pytest.raises(ConfigurationError):
        CHUNKERS.create("semantic", config=config, length_fn=length_function_for(config), dense_embedder=None)


def test_token_length_approximation() -> None:
    fn = length_function_for(ChunkingConfig(length_unit="tokens", tokenizer="approx"))
    assert fn("abcd" * 10) == 10
