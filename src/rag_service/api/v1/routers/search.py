"""Retrieval (raw RAG results), chat (grounded answers with citations) and evaluation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from rag_service.api.deps import ContainerDep, QueryPrincipal
from rag_service.api.metrics import LLM_TOKENS, RETRIEVAL_RESULTS, RETRIEVALS
from rag_service.api.responses import envelope, request_id
from rag_service.api.schemas.common import Envelope
from rag_service.api.schemas.v1 import (
    AssistantMessage,
    ChatRequest,
    ChatResponse,
    ChatRetrievalOut,
    EvalCaseOut,
    EvaluateRequest,
    EvaluateResponse,
    RetrieveRequest,
    RetrieveResponse,
    UsageOut,
    VariantReportOut,
)
from rag_service.api.sse import error_event, sse_event, sse_response
from rag_service.api.v1.mappers import citation_out, retrieve_response
from rag_service.application.evaluation import EvalCase
from rag_service.core.utils import new_id
from rag_service.domain.models import ChatMessage

router = APIRouter(prefix="/collections/{collection}", tags=["Retrieval & Chat"])

CHAT_SSE_DOC = """
**Streaming (`stream: true`)** returns `text/event-stream` with these events:

| event | data |
|-------|------|
| `retrieval` | `{"id", "conversation_id", "standalone_query", "sources": [Citation]}` |
| `delta` | `{"content": "<text fragment>"}` (many) |
| `done` | `{"citations": [Citation], "finish_reason", "usage", "model"}` |
| `error` | problem details — sent if generation fails after the stream started |
"""


@router.post(
    "/retrieve",
    response_model=Envelope[RetrieveResponse],
    summary="Retrieve relevant chunks (raw RAG results)",
    description=(
        "Runs the collection's retrieval pipeline (dense / sparse / hybrid + fusion, optional query transform, "
        "re-ranking, MMR and context expansion) with optional per-request overrides and metadata filters. "
        "Set `include_debug` to see each retriever's ranked list and the fused list."
    ),
)
async def retrieve(
    request: Request, collection: str, body: RetrieveRequest, container: ContainerDep, _: QueryPrincipal
) -> Envelope[RetrieveResponse]:
    result = await container.retrieval.retrieve(
        collection, body.query, filter_=body.filter, overrides=body.overrides(), debug=body.include_debug
    )
    RETRIEVALS.labels(collection, result.mode.value).inc()
    RETRIEVAL_RESULTS.labels(collection).observe(len(result.chunks))
    return envelope(request, retrieve_response(collection, result, debug=body.include_debug))


@router.post(
    "/chat",
    response_model=Envelope[ChatResponse],
    summary="Grounded chat answer with citations",
    description="Retrieves context for the last user message (condensing follow-ups with the history) and "
    "generates an answer citing sources as [n].\n" + CHAT_SSE_DOC,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat(
    request: Request, collection: str, body: ChatRequest, container: ContainerDep, _: QueryPrincipal
) -> Envelope[ChatResponse] | StreamingResponse:
    messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    retrieval_overrides = body.retrieval.as_overrides() if body.retrieval else None
    generation_overrides = body.generation.as_overrides() if body.generation else None
    chat_id = new_id("chat")

    if body.stream:
        rid = request_id(request)
        prepared = await container.chat.prepare(
            collection,
            messages,
            filter_=body.filter,
            retrieval_overrides=retrieval_overrides,
            generation_overrides=generation_overrides,
        )

        async def events() -> AsyncIterator[bytes]:
            try:
                async for event in container.chat.stream(prepared):
                    data: dict[str, Any]
                    if event.type == "retrieval":
                        data = {
                            "id": chat_id,
                            "conversation_id": body.conversation_id,
                            "standalone_query": event.data["standalone_query"],
                            "sources": [citation_out(s).model_dump() for s in event.data["sources"]]
                            if body.include_sources
                            else [],
                        }
                    elif event.type == "done":
                        usage = event.data["usage"]
                        LLM_TOKENS.labels(container.settings.llm.provider, "output").inc(usage.output_tokens)
                        data = {
                            "id": chat_id,
                            "citations": [citation_out(c).model_dump() for c in event.data["citations"]],
                            "finish_reason": event.data["finish_reason"],
                            "model": event.data["model"],
                            "usage": {
                                "input_tokens": usage.input_tokens,
                                "output_tokens": usage.output_tokens,
                                "total_tokens": usage.total_tokens,
                            },
                        }
                    else:
                        data = event.data
                    yield sse_event(event.type, data)
            except Exception as exc:
                yield error_event(exc, rid)

        return sse_response(events())

    result = await container.chat.answer(
        collection,
        messages,
        filter_=body.filter,
        retrieval_overrides=retrieval_overrides,
        generation_overrides=generation_overrides,
    )
    LLM_TOKENS.labels(container.settings.llm.provider, "input").inc(result.usage.input_tokens)
    LLM_TOKENS.labels(container.settings.llm.provider, "output").inc(result.usage.output_tokens)
    return envelope(
        request,
        ChatResponse(
            id=chat_id,
            conversation_id=body.conversation_id,
            collection=collection,
            model=result.model,
            message=AssistantMessage(content=result.answer),
            citations=[citation_out(c) for c in result.citations],
            sources=[citation_out(c) for c in result.sources] if body.include_sources else None,
            finish_reason=result.finish_reason,
            usage=UsageOut(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                total_tokens=result.usage.total_tokens,
            ),
            retrieval=ChatRetrievalOut(
                standalone_query=result.standalone_query,
                mode=result.retrieval.mode.value,
                results=len(result.retrieval.chunks),
                timings_ms=result.retrieval.timings,
            ),
        ),
    )


@router.post(
    "/evaluate",
    response_model=Envelope[EvaluateResponse],
    summary="Evaluate retrieval quality on labelled queries",
    description="Compares retrieval variants (e.g. dense vs sparse vs hybrid vs hybrid+rerank) with Hit-rate, "
    "Recall@k, Precision@k, MRR and nDCG@k.",
)
async def evaluate(
    request: Request, collection: str, body: EvaluateRequest, container: ContainerDep, _: QueryPrincipal
) -> Envelope[EvaluateResponse]:
    cases = [
        EvalCase(
            query=c.query,
            relevant=c.relevant if isinstance(c.relevant, dict) else dict.fromkeys(c.relevant, 1.0),
            filter=c.filter,
        )
        for c in body.cases
    ]
    reports = await container.evaluation.evaluate(
        collection,
        cases,
        k=body.k,
        level=body.level,
        variants=body.variants,
        include_cases=body.include_cases,
    )
    variants = [
        VariantReportOut(
            name=r.name,
            overrides=r.overrides,
            metrics=r.metrics,
            cases=[
                EvalCaseOut(
                    query=c.query,
                    retrieved=c.retrieved,
                    first_relevant_rank=c.first_relevant_rank,
                    recall=round(c.recall, 4),
                    precision=round(c.precision, 4),
                    ndcg=round(c.ndcg, 4),
                )
                for c in r.cases
            ],
        )
        for r in reports
    ]
    ranking = sorted(reports, key=lambda r: (r.metrics["ndcg"], r.metrics["mrr"]), reverse=True)
    top = [(r.metrics["ndcg"], r.metrics["mrr"]) for r in ranking[:2]]
    best = ranking[0].name if ranking and (len(top) == 1 or top[0] != top[1]) else None  # null on a tie
    return envelope(
        request,
        EvaluateResponse(collection=collection, k=body.k, level=body.level, variants=variants, best=best),
    )
