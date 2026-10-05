"""Pre-retrieval query transformations.

* ``none``        — search with the user query as-is.
* ``multi_query`` — the LLM writes N paraphrases; every variant is searched and all ranked lists
  are fused (RAG-Fusion). Improves recall for ambiguous or under-specified questions.
* ``hyde``        — Hypothetical Document Embeddings (Gao et al., 2022): the LLM drafts an answer
  passage which is embedded for *dense* search (documents look more like answers than questions);
  sparse search keeps the literal query so exact terms still match.
* ``rewrite``     — the LLM rewrites a conversational query into a keyword-rich search query.

Transform failures never fail the request: they fall back to the original query.
"""

from __future__ import annotations

from rag_service.core.config import QueryTransformConfig
from rag_service.core.logging import get_logger
from rag_service.core.registry import QUERY_TRANSFORMS
from rag_service.domain.models import ChatMessage, LLMRequest, QueryVariant
from rag_service.domain.ports import LLMClient

log = get_logger(__name__)

MULTI_QUERY_PROMPT = (
    "Generate {n} different search queries that would retrieve documents answering the question below. "
    "Vary vocabulary and specificity (synonyms, product terms, broader and narrower phrasings). "
    "Return one query per line with no numbering or commentary.\n\nQuestion: {query}"
)
HYDE_PROMPT = (
    "Write a short, factual passage (3-5 sentences) that directly answers the question below, as it "
    "might appear in a product description or documentation page. Do not mention the question.\n\n"
    "Question: {query}"
)
REWRITE_PROMPT = (
    "Rewrite the question below into a concise search query containing the key entities, product names, "
    "numbers and technical terms. Return only the query.\n\nQuestion: {query}"
)


class NoTransform:
    name = "none"

    async def transform(self, query: str) -> list[QueryVariant]:
        return [QueryVariant(text=query)]


class _LLMTransform:
    name = "llm"
    prompt = ""
    max_tokens = 400

    def __init__(self, config: QueryTransformConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

    async def _ask(self, query: str) -> str:
        prompt = self.prompt.format(query=query, n=self.config.num_queries)
        response = await self.llm.generate(
            LLMRequest(messages=[ChatMessage(role="user", content=prompt)], max_tokens=self.max_tokens)
        )
        return response.text.strip()

    def _variants(self, query: str, generated: str) -> list[QueryVariant]:
        raise NotImplementedError

    async def transform(self, query: str) -> list[QueryVariant]:
        try:
            generated = await self._ask(query)
        except Exception as exc:  # degrade gracefully — retrieval must still work
            log.warning("query_transform.failed", strategy=self.name, error=str(exc))
            return [QueryVariant(text=query)]
        variants = self._variants(query, generated)
        seen: set[tuple[str, bool, bool]] = set()
        unique = []
        for v in variants:
            key = (v.text.strip().lower(), v.embed_dense, v.embed_sparse)
            if v.text.strip() and key not in seen:
                seen.add(key)
                unique.append(v)
        return unique or [QueryVariant(text=query)]


class MultiQueryTransform(_LLMTransform):
    name = "multi_query"
    prompt = MULTI_QUERY_PROMPT

    def _variants(self, query: str, generated: str) -> list[QueryVariant]:
        lines = [line.strip(" -*\t0123456789.)") for line in generated.splitlines()]
        out = [QueryVariant(text=query)] if self.config.include_original else []
        out += [QueryVariant(text=line, kind="multi_query") for line in lines if line][: self.config.num_queries]
        return out


class HyDETransform(_LLMTransform):
    name = "hyde"
    prompt = HYDE_PROMPT

    def _variants(self, query: str, generated: str) -> list[QueryVariant]:
        out = [QueryVariant(text=generated, kind="hyde", embed_dense=True, embed_sparse=False)]
        if self.config.include_original:
            out.insert(0, QueryVariant(text=query))
        else:
            out.append(QueryVariant(text=query, embed_dense=False, embed_sparse=True))
        return out


class RewriteTransform(_LLMTransform):
    name = "rewrite"
    prompt = REWRITE_PROMPT
    max_tokens = 120

    def _variants(self, query: str, generated: str) -> list[QueryVariant]:
        out = [QueryVariant(text=generated.splitlines()[0] if generated else query, kind="rewrite")]
        if self.config.include_original:
            out.insert(0, QueryVariant(text=query))
        return out


@QUERY_TRANSFORMS.register("none", description="Use the query as-is")
def _none(*, config: QueryTransformConfig, llm: LLMClient | None = None) -> NoTransform:
    return NoTransform()


def _needs_llm(llm: LLMClient | None, name: str) -> LLMClient:
    if llm is None:
        from rag_service.core.errors import ConfigurationError  # noqa: PLC0415

        raise ConfigurationError(f"query_transform '{name}' requires an LLM (llm / utility_llm)")
    return llm


@QUERY_TRANSFORMS.register(
    "multi_query", aliases=("rag_fusion",), description="LLM paraphrases fused with RRF (RAG-Fusion)"
)
def _multi(*, config: QueryTransformConfig, llm: LLMClient | None = None) -> MultiQueryTransform:
    return MultiQueryTransform(config, _needs_llm(llm, "multi_query"))


@QUERY_TRANSFORMS.register("hyde", description="Hypothetical Document Embeddings for dense search")
def _hyde(*, config: QueryTransformConfig, llm: LLMClient | None = None) -> HyDETransform:
    return HyDETransform(config, _needs_llm(llm, "hyde"))


@QUERY_TRANSFORMS.register("rewrite", description="LLM keyword-oriented query rewrite")
def _rewrite(*, config: QueryTransformConfig, llm: LLMClient | None = None) -> RewriteTransform:
    return RewriteTransform(config, _needs_llm(llm, "rewrite"))
