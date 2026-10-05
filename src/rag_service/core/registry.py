"""Named component registries — the Open/Closed backbone of the service.

Adapters register a *factory* under a name; configuration selects the name. Adding a new vector
store, chunker, embedder, fusion method, re-ranker, query transform, parser or LLM therefore never
requires editing existing code:

.. code-block:: python

    from rag_service.core.registry import CHUNKERS

    @CHUNKERS.register("paragraph")
    def build_paragraph_chunker(*, config, length_fn, dense_embedder=None):
        return ParagraphChunker(config, length_fn)

External packages can register components through the ``rag_service.plugins`` entry-point group,
and extra modules can be imported at startup via the ``plugins`` configuration key.

Factory keyword arguments per registry are documented in ``doc/08-extending.md``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, Generic, TypeVar

from rag_service.core.errors import ConfigurationError
from rag_service.core.logging import get_logger

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

log = get_logger(__name__)


class ComponentRegistry(Generic[T]):
    """Maps a name (and optional aliases) to a factory that builds a component of type ``T``."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, name: str, *, aliases: tuple[str, ...] = (), description: str = "") -> Callable[[F], F]:
        def decorator(factory: F) -> F:
            for alias in (name, *aliases):
                key = alias.lower()
                if key in self._factories and self._factories[key] is not factory:
                    log.warning("registry.override", kind=self.kind, name=key)
                self._factories[key] = factory
            self._descriptions[name.lower()] = description or (factory.__doc__ or "").strip().split("\n")[0]
            return factory

        return decorator

    def create(self, name: str, /, **kwargs: Any) -> T:
        try:
            factory = self._factories[name.lower()]
        except KeyError:
            raise ConfigurationError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names()) or '(none)'}"
            ) from None
        return factory(**kwargs)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._factories

    def names(self) -> list[str]:
        return sorted(self._descriptions)

    def describe(self) -> dict[str, str]:
        return dict(sorted(self._descriptions.items()))


# Registry instances. The concrete component types are Protocols in `rag_service.domain.ports`;
# `Any` keeps this module free of a dependency on the domain layer.
PARSERS: ComponentRegistry[Any] = ComponentRegistry("parser")
LENGTH_FUNCTIONS: ComponentRegistry[Any] = ComponentRegistry("length function")
CHUNKERS: ComponentRegistry[Any] = ComponentRegistry("chunker")
DENSE_EMBEDDERS: ComponentRegistry[Any] = ComponentRegistry("dense embedder")
SPARSE_EMBEDDERS: ComponentRegistry[Any] = ComponentRegistry("sparse embedder")
VECTOR_STORES: ComponentRegistry[Any] = ComponentRegistry("vector store")
FUSIONS: ComponentRegistry[Any] = ComponentRegistry("fusion method")
RERANKERS: ComponentRegistry[Any] = ComponentRegistry("reranker")
QUERY_TRANSFORMS: ComponentRegistry[Any] = ComponentRegistry("query transform")
LLMS: ComponentRegistry[Any] = ComponentRegistry("llm provider")

ALL_REGISTRIES: dict[str, ComponentRegistry[Any]] = {
    "parsers": PARSERS,
    "length_functions": LENGTH_FUNCTIONS,
    "chunkers": CHUNKERS,
    "dense_embedders": DENSE_EMBEDDERS,
    "sparse_embedders": SPARSE_EMBEDDERS,
    "vector_stores": VECTOR_STORES,
    "fusions": FUSIONS,
    "rerankers": RERANKERS,
    "query_transforms": QUERY_TRANSFORMS,
    "llms": LLMS,
}

_loaded = False


def load_plugins(extra_modules: list[str] | None = None) -> None:
    """Import built-in adapters, entry-point plugins and configured modules (idempotent)."""
    global _loaded  # noqa: PLW0603 - one-time process initialisation
    if not _loaded:
        importlib.import_module("rag_service.adapters")
        for ep in entry_points(group="rag_service.plugins"):
            try:
                register = ep.load()
                if callable(register):
                    register(ALL_REGISTRIES)
                log.info("plugin.loaded", plugin=ep.name)
            except Exception as exc:  # a broken plugin must not take the service down silently
                raise ConfigurationError(f"Failed to load plugin '{ep.name}': {exc}") from exc
        _loaded = True
    for module in extra_modules or []:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise ConfigurationError(f"Cannot import plugin module '{module}': {exc}") from exc
