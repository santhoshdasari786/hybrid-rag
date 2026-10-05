"""Chunk enrichers — they change the text that is *embedded*, never the text that is returned.

* :class:`ContextualHeaderEnricher` prepends title / section path. Cheap and consistently helpful:
  a chunk that says "It draws 80 CFM" becomes findable for "Panasonic WhisperCeiling airflow".
* :class:`LLMContextEnricher` implements Anthropic's *Contextual Retrieval*: an LLM reads the whole
  document and writes 1–2 sentences situating each chunk; that context is prepended before
  embedding. It runs only for new/changed chunks (the ingestion diff happens first), and the
  document is sent as a cacheable system prompt so repeated calls hit the prompt cache.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence

from rag_service.core.config import ContextualHeaderConfig, LLMContextConfig
from rag_service.core.logging import get_logger
from rag_service.domain.models import ChatMessage, Chunk, LLMRequest
from rag_service.domain.ports import LLMClient

log = get_logger(__name__)

_BLANKS = re.compile(r"\n{3,}")


class _SafeDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""


class ContextualHeaderEnricher:
    def __init__(self, config: ContextualHeaderConfig) -> None:
        self.config = config

    def embed_text(self, text: str, *, title: str | None, section: str | None, source_id: str) -> str:
        if not self.config.enabled:
            return text
        rendered = self.config.template.format_map(
            _SafeDict(title=title or "", section=section or "", text=text, source_id=source_id)
        )
        lines = rendered.split("\n")
        # Drop leading empty header lines and collapse gaps left by empty placeholders.
        while lines and not lines[0].strip():
            lines.pop(0)
        return _BLANKS.sub("\n\n", "\n".join(lines)).strip() or text


CONTEXT_SYSTEM = (
    "You situate document excerpts for a search index. Here is the full document:\n<document>\n{document}\n</document>"
)
CONTEXT_USER = (
    "Here is an excerpt from the document:\n<excerpt>\n{chunk}\n</excerpt>\n"
    "Write one or two sentences that situate this excerpt within the overall document to improve search "
    "retrieval of the excerpt (mention the product/topic and section). Answer with the context only."
)


class LLMContextEnricher:
    def __init__(self, config: LLMContextConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

    async def enrich(self, document_text: str, chunks: Sequence[Chunk]) -> None:
        if not self.config.enabled or not chunks:
            return
        system = CONTEXT_SYSTEM.format(document=document_text[: self.config.max_document_chars])
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def one(chunk: Chunk) -> None:
            async with semaphore:
                try:
                    response = await self.llm.generate(
                        LLMRequest(
                            system=system,
                            messages=[ChatMessage(role="user", content=CONTEXT_USER.format(chunk=chunk.text))],
                            max_tokens=self.config.max_tokens,
                            cache_system=True,
                        )
                    )
                except Exception as exc:  # enrichment is best-effort
                    log.warning("enrichment.llm_context.failed", chunk_id=chunk.id, error=str(exc))
                    return
                context = response.text.strip()
                if context and context != chunk.text:
                    chunk.embed_text = f"{context}\n\n{chunk.embed_text}"
                    chunk.metadata["llm_context"] = context

        await asyncio.gather(*(one(c) for c in chunks))
