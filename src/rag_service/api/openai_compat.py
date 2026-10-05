"""OpenAI-compatible Chat Completions facade.

Point any OpenAI SDK or chat UI (Open WebUI, LibreChat, LangChain ``ChatOpenAI``…) at
``http(s)://<host>/v1`` and use ``rag:<collection>`` (or just ``<collection>``) as the model name.
Answers are grounded in the collection; citations are returned in the extra ``citations`` field
(and on the final streamed chunk). RAG options can be passed in an optional ``rag`` object.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from rag_service.api.deps import AnyQueryPrincipal, ContainerDep
from rag_service.api.schemas.v1 import GenerationOverrides, RetrievalOverrides
from rag_service.api.sse import error_event, sse_event, sse_response
from rag_service.api.v1.mappers import citation_out
from rag_service.core.errors import ValidationFailedError
from rag_service.core.utils import new_id
from rag_service.domain.filters import FilterSpec
from rag_service.domain.models import ChatMessage

router = APIRouter(prefix="/v1", tags=["OpenAI-compatible"])

_FINISH = {
    "stop": "stop",
    "length": "length",
    "refusal": "content_filter",
    "no_context": "stop",
    "tool_calls": "tool_calls",
}


class OAIMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: str | list[dict[str, Any]] | None = None

    def text(self) -> str:
        if isinstance(self.content, list):
            return "".join(str(part.get("text", "")) for part in self.content if part.get("type") == "text")
        return self.content or ""


class RagOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filter: FilterSpec | None = None
    retrieval: RetrievalOverrides | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerate unused OpenAI params (n, top_p, user, …)

    model: str = Field(..., examples=["rag:products"])
    messages: list[OAIMessage] = Field(..., min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(None, ge=16)
    max_completion_tokens: int | None = Field(None, ge=16)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    rag: RagOptions | None = None


def _collection(model: str) -> str:
    return model.split(":", 1)[1] if model.startswith("rag:") else model


@router.get("/models", summary="List collections as models")
async def list_models(container: ContainerDep, principal: AnyQueryPrincipal) -> dict[str, Any]:
    infos = await container.collections.list_all()
    return {
        "object": "list",
        "data": [
            {
                "id": f"rag:{i.record.name}",
                "object": "model",
                "created": int(i.record.created_at.timestamp()) if i.record.created_at else 0,
                "owned_by": container.settings.app.name,
            }
            for i in infos
            if principal.can_access(i.record.name)
        ],
    }


@router.post(
    "/chat/completions",
    summary="OpenAI-compatible chat completion grounded in a collection",
    response_model=None,
)
async def chat_completions(
    request: Request, body: ChatCompletionRequest, container: ContainerDep, principal: AnyQueryPrincipal
) -> dict[str, Any] | StreamingResponse:
    collection = _collection(body.model)
    principal.require_collection(collection)
    messages = [
        ChatMessage(role="system" if m.role == "developer" else m.role, content=m.text())
        for m in body.messages
        if m.role in ("system", "developer", "user", "assistant")
    ]
    if not messages or messages[-1].role != "user":
        raise ValidationFailedError("The last message must be a user message")
    generation: dict[str, Any] = {}
    max_tokens = body.max_completion_tokens or body.max_tokens
    if max_tokens:
        generation["max_tokens"] = max_tokens
    if body.temperature is not None:
        generation["temperature"] = body.temperature
    system_prompts = [m.content for m in messages if m.role == "system"]
    if system_prompts:
        base = (await container.collections.pipeline(collection)).config.generation.system_prompt
        generation["system_prompt"] = base + "\n\n" + "\n\n".join(system_prompts)
    generation = GenerationOverrides(**generation).as_overrides()
    chat_messages = [m for m in messages if m.role != "system"]
    rag = body.rag or RagOptions()
    retrieval_overrides = rag.retrieval.as_overrides() if rag.retrieval else None
    completion_id = new_id("chatcmpl")
    created = int(time.time())

    if body.stream:
        prepared = await container.chat.prepare(
            collection,
            chat_messages,
            filter_=rag.filter,
            retrieval_overrides=retrieval_overrides,
            generation_overrides=generation or None,
        )
        rid = str(getattr(request.state, "request_id", ""))

        def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> bytes:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **extra,
            }
            return sse_event(None, payload)

        async def events() -> AsyncIterator[bytes]:
            try:
                yield chunk({"role": "assistant", "content": ""})
                async for event in container.chat.stream(prepared):
                    if event.type == "delta":
                        yield chunk({"content": event.data["content"]})
                    elif event.type == "done":
                        usage = event.data["usage"]
                        yield chunk(
                            {},
                            _FINISH.get(event.data["finish_reason"], "stop"),
                            usage={
                                "prompt_tokens": usage.input_tokens,
                                "completion_tokens": usage.output_tokens,
                                "total_tokens": usage.total_tokens,
                            },
                            citations=[citation_out(c).model_dump() for c in event.data["citations"]],
                        )
                yield sse_event(None, "[DONE]")
            except Exception as exc:
                yield error_event(exc, rid)

        return sse_response(events())

    result = await container.chat.answer(
        collection,
        chat_messages,
        filter_=rag.filter,
        retrieval_overrides=retrieval_overrides,
        generation_overrides=generation or None,
    )
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": _FINISH.get(result.finish_reason, "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": result.usage.input_tokens,
            "completion_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "citations": [citation_out(c).model_dump() for c in result.citations],
    }
