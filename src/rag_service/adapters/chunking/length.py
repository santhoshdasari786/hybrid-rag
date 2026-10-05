"""Length functions used to size chunks and context windows."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from rag_service.core.config import ChunkingConfig, LengthUnit
from rag_service.core.errors import ConfigurationError, DependencyMissingError
from rag_service.core.registry import LENGTH_FUNCTIONS
from rag_service.domain.ports import LengthFunction


class CharLength:
    name = "chars"

    def __call__(self, text: str) -> int:
        return len(text)


class WordLength:
    name = "words"

    def __call__(self, text: str) -> int:
        return len(text.split())


class ApproxTokenLength:
    """~4 characters per token for English text — fast and dependency-free."""

    name = "approx"

    def __call__(self, text: str) -> int:
        return math.ceil(len(text) / 4)


class TiktokenLength:
    def __init__(self, encoding: str) -> None:
        try:
            import tiktoken  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise DependencyMissingError.for_extra("tiktoken", "tokenizers") from exc
        self.name = f"tiktoken:{encoding}"
        self._encoding: Any = tiktoken.get_encoding(encoding)

    def __call__(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


@LENGTH_FUNCTIONS.register("chars", description="Characters")
def _chars(**_: Any) -> CharLength:
    return CharLength()


@LENGTH_FUNCTIONS.register("words", description="Whitespace-separated words")
def _words(**_: Any) -> WordLength:
    return WordLength()


@LENGTH_FUNCTIONS.register("approx", description="Approximate tokens (chars / 4)")
def _approx(**_: Any) -> ApproxTokenLength:
    return ApproxTokenLength()


@LENGTH_FUNCTIONS.register("tiktoken", description="Exact OpenAI BPE token counts (extra: tokenizers)")
def _tiktoken(encoding: str = "cl100k_base", **_: Any) -> TiktokenLength:
    return TiktokenLength(encoding)


@lru_cache(maxsize=16)
def _cached(spec: str) -> LengthFunction:
    name, _, arg = spec.partition(":")
    kwargs = {"encoding": arg} if arg else {}
    fn: LengthFunction = LENGTH_FUNCTIONS.create(name, **kwargs)
    return fn


def length_function_for(config: ChunkingConfig) -> LengthFunction:
    if config.length_unit == LengthUnit.CHARS:
        return _cached("chars")
    if config.length_unit == LengthUnit.WORDS:
        return _cached("words")
    spec = config.tokenizer or "approx"
    if spec.split(":", 1)[0] not in LENGTH_FUNCTIONS:
        raise ConfigurationError(f"Unknown tokenizer '{spec}'. Use 'approx' or 'tiktoken:<encoding>'.")
    return _cached(spec)
