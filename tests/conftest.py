"""Shared fixtures. The default stack is fully offline: memory store, hash/lexical embedders,
in-memory SQLite registry and the extractive LLM — no downloads, no network, no Docker."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag_service.adapters.embedding.local import HashingDenseEmbedder
from rag_service.api.app import create_app
from rag_service.container import Container
from rag_service.core.config import DenseEmbeddingConfig, Settings
from rag_service.core.registry import DENSE_EMBEDDERS, load_plugins

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

os.environ.pop("RAG_CONFIG_FILE", None)
load_plugins()


class CountingEmbedder(HashingDenseEmbedder):
    """Hash embedder that records how many texts were embedded (to assert incremental re-sync)."""

    calls: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        CountingEmbedder.calls.extend(texts)
        return await super().embed_documents(texts)


@DENSE_EMBEDDERS.register("counting", description="test-only")
def _counting(*, config: DenseEmbeddingConfig) -> CountingEmbedder:
    return CountingEmbedder(dimension=config.dimension or 128, distance=config.distance, model_id="counting")


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "app": {"warmup_models": False, "environment": "test"},
        "logging": {"level": "WARNING", "format": "console", "access_log": False},
        "registry": {"url": "sqlite+aiosqlite:///:memory:"},
        "vector_store": {"provider": "memory"},
        "ingestion": {"default_mode": "sync"},
        "defaults": {
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 60,
                "chunk_overlap": 10,
                "length_unit": "words",
            },
            "embedding": {
                "dense": {"provider": "hash", "model": "hash-256"},
                "sparse": {"provider": "lexical"},
            },
        },
        "collections": {"products": {"description": "Test catalog", "payload_indexes": [{"field": "brand"}]}},
    }
    for key, value in overrides.items():
        if key != "collections" and isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return Settings.load([], **base)


@pytest.fixture
def settings_factory() -> Callable[..., Settings]:
    return make_settings


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
async def container(settings: Settings) -> AsyncIterator[Container]:
    c = Container.build(settings)
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


@pytest.fixture
async def app_client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async with client_for(settings) as client:
        yield client


class client_for:  # noqa: N801 - reads naturally as a context manager
    """``async with client_for(settings) as client`` — app with lifespan + httpx client."""

    def __init__(self, settings: Settings, headers: dict[str, str] | None = None) -> None:
        self.app = create_app(settings)
        self.headers = headers or {}

    async def __aenter__(self) -> httpx.AsyncClient:
        await self.app.state.container.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://test",
            headers=self.headers,
            timeout=30,
        )
        return self.client

    async def __aexit__(self, *exc: object) -> None:
        await self.client.aclose()
        await self.app.state.container.stop()


FAN_DOCS = [
    {
        "source_id": "fan-panasonic-fv20vq3",
        "title": "Panasonic FV-20VQ3 WhisperCeiling 190 CFM Ceiling Mounted Fan",
        "mime_type": "text/markdown",
        "chunking": {"strategy": "markdown"},
        "metadata": {
            "brand": "Panasonic",
            "cfm": 190,
            "category": "bathroom-fan",
            "tags": ["quiet", "ceiling"],
        },
        "content": (
            "# Panasonic FV-20VQ3 WhisperCeiling\n\n"
            "WhisperCeiling fans feature a totally enclosed condenser motor and a double-tapered, dolphin-shaped "
            "blower wheel to quietly move air. It moves 190 CFM.\n\n"
            "## Installation\n\nDetachable adaptors, adjustable mounting brackets and uncomplicated wiring make "
            "installation user friendly. A built-in damper prevents backdraft."
        ),
    },
    {
        "source_id": "fan-delta-rad80",
        "title": "Delta BreezRadiance RAD80 80 CFM Fan with Heater",
        "mime_type": "text/markdown",
        "chunking": {"strategy": "markdown"},
        "metadata": {"brand": "Delta", "cfm": 80, "category": "bathroom-fan", "tags": ["heater"]},
        "content": (
            "# Delta BreezRadiance RAD80\n\nThe Delta BreezRadiance series 80 CFM bathroom exhaust fan includes a "
            "heater and a light. Energy efficient DC motor.\n\n## Noise\n\nRated at 1.5 sones."
        ),
    },
    {
        "source_id": "fan-broan-quiet80",
        "title": "Broan Very Quiet Ceiling Bathroom Exhaust Fan 80 CFM",
        "mime_type": "text/plain",
        "metadata": {
            "brand": "Broan-NuTone",
            "cfm": 80,
            "category": "bathroom-fan",
            "tags": ["quiet", "energy-star"],
        },
        "content": (
            "Broan very quiet ceiling bathroom exhaust fan, ENERGY STAR certified, 0.3 sones, 80 CFM. "
            "Covers bathrooms up to 75 sq. ft. Humidity and moisture control for small rooms."
        ),
    },
    {
        "source_id": "blender-ninja",
        "title": "Ninja Professional Kitchen Blender",
        "mime_type": "text/plain",
        "metadata": {"brand": "Ninja", "category": "kitchen", "tags": ["blender"]},
        "content": "Ninja professional countertop blender with 1000 watt motor for smoothies and crushing ice.",
    },
]


@pytest.fixture
def fan_docs() -> list[dict[str, Any]]:
    return [dict(d) for d in FAN_DOCS]
