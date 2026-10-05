"""The cookbook is executable documentation: every recipe must pass on the offline stack, and the steps that
need a generative LLM must pass with a scripted one. A recipe that drifts from the product fails CI."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from rag_service.core.config import LLMConfig
from rag_service.core.registry import LLMS
from rag_service.domain.models import LLMRequest, LLMResponse, LLMStreamEvent, TokenUsage

ROOT = Path(__file__).resolve().parents[2]


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cookbook_run", ROOT / "cookbook" / "run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["cookbook_run"] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
RECIPES = runner.all_recipes()


def failures(report: Any) -> list[str]:
    return [f"{s.name}: {s.detail} {s.lines}" for s in report.steps if s.status == "fail"]


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda r: r.name)
async def test_recipe_passes_on_the_offline_stack(recipe: Any) -> None:
    report = await runner.run_recipe(recipe, "offline", "extractive")
    assert not failures(report), "\n".join(failures(report))
    assert sum(s.status == "pass" for s in report.steps) >= 5


class ScriptedLLM:
    """A generative stand-in: rewrites follow-ups, expands queries, situates chunks, answers with a citation."""

    provider = "cookbook-scripted"
    model = "scripted-1"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @staticmethod
    def _reply(request: LLMRequest) -> str:
        prompt = "\n".join([request.system or "", *(m.content for m in request.messages)])
        if "standalone search query" in prompt:
            return "Can I take my Lumen Fibre service with me when I move house?"
        if "different search queries" in prompt:
            return (
                "1. internet connection keeps dropping\n2. router light red no internet\n3. restart the router safely"
            )
        if "situate this excerpt" in prompt:
            return "This excerpt is part of the Halcyon Labs Employee Handbook."
        return "Here is what the help centre says [1]."

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text=self._reply(request), model=self.model, usage=TokenUsage(10, 5))

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        yield LLMStreamEvent(delta=self._reply(request))
        yield LLMStreamEvent(done=True, finish_reason="stop", usage=TokenUsage(10, 5), model=self.model)

    async def close(self) -> None:
        return None


@LLMS.register("cookbook-scripted", description="test-only")
def _scripted(*, config: LLMConfig) -> ScriptedLLM:
    return ScriptedLLM(config)


@pytest.mark.parametrize("name", ["04-support-helpdesk", "06-long-documents"])
async def test_generative_steps_run_with_a_writing_llm(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = tmp_path / "llm-scripted.yaml"
    overlay.write_text("llm: {provider: cookbook-scripted, model: scripted-1}\n")
    monkeypatch.setitem(runner.LLM_LAYERS, "scripted", (str(overlay),))
    recipe = next(r for r in RECIPES if r.name == name)
    report = await runner.run_recipe(recipe, "offline", "scripted")
    assert not failures(report), "\n".join(failures(report))
    generative = [
        s
        for s in report.steps
        if s.status == "pass"
        and s.name in {st.get("name") for st in recipe.steps if (st.get("when") or {}).get("llm") == "generative"}
    ]
    assert generative, "the generative steps should have run"
