"""Run the cookbook: compose configuration layers, start the service, play each recipe's scenario and check
what it promises.

    uv run python cookbook/run.py --list
    uv run python cookbook/run.py                               # every recipe on the offline stack
    uv run python cookbook/run.py ecommerce -v                  # one recipe, show the ranked results
    uv run python cookbook/run.py ecommerce cms --stack models  # real models (BGE + SPLADE + reranker), in memory
    uv run python cookbook/run.py cms --stack qdrant --llm anthropic
    uv run python cookbook/run.py cms --serve                   # serve that configuration on :8080 (Swagger at /docs)
    uv run python cookbook/run.py cms --base-url http://localhost:8080   # check a service you started yourself

Configuration layers, lowest to highest priority (see cookbook/README.md):
    config/rag.yaml → stack → recipes/<recipe>/config.yaml → LLM overlay → RAG__* env → per-request overrides
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "cookbook" / "recipes"
API = "/api/v1"


@dataclass(frozen=True)
class Stack:
    files: tuple[str, ...]
    external: bool
    help: str


STACKS = {
    "offline": Stack((), False, "in-memory store, hash + BM25-style embedders, extractive answers (no downloads)"),
    "models": Stack(
        ("config/profiles/qdrant-hybrid.yaml", "cookbook/stacks/memory-store.yaml"),
        False,
        "BGE-small + SPLADE++ + MiniLM cross-encoder, in-memory store (models download once, ~0.7 GB)",
    ),
    "qdrant": Stack(("config/profiles/qdrant-hybrid.yaml",), True, "Qdrant server ($QDRANT_URL) + the models above"),
    "pgvector": Stack(
        ("config/profiles/pgvector.yaml", "cookbook/stacks/reranker.yaml"),
        True,
        "PostgreSQL + pgvector ($PG_DSN, $REGISTRY_URL) + the same models and reranker",
    ),
}
LLM_LAYERS = {
    "extractive": (),
    "anthropic": ("config/profiles/llm-anthropic.yaml",),
    "openai": ("config/profiles/llm-openai.yaml",),
    "gemini": ("config/profiles/llm-gemini.yaml",),
    "claude-lite": ("config/profiles/llm-claude-lite.yaml",),
}


# ============================================================================== recipes


@dataclass
class Recipe:
    path: Path
    title: str
    summary: str
    collection: str
    steps: list[dict[str, Any]]

    @property
    def name(self) -> str:
        return self.path.name

    @classmethod
    def load(cls, path: Path) -> Recipe:
        try:
            data = yaml.safe_load((path / "scenario.yaml").read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # usually an unquoted ':', ',' or '?' inside {…}
            raise SystemExit(f"{path / 'scenario.yaml'}: invalid YAML — {exc}") from None
        return cls(path, data["title"], data.get("summary", ""), data["collection"], data["steps"])

    def data(self, relative: str) -> Any:
        return json.loads((self.path / relative).read_text(encoding="utf-8"))


def all_recipes() -> list[Recipe]:
    return [Recipe.load(p) for p in sorted(RECIPES.iterdir()) if (p / "scenario.yaml").exists()]


def select(names: list[str]) -> list[Recipe]:
    recipes = all_recipes()
    if not names:
        return recipes
    chosen = [r for r in recipes if any(n in r.name for n in names)]
    unknown = [n for n in names if not any(n in r.name for r in recipes)]
    if unknown:
        raise SystemExit(f"Unknown recipe(s): {', '.join(unknown)}. Try --list.")
    return chosen


def layers(recipe: Recipe, stack: str, llm: str) -> list[str]:
    return [
        "config/rag.yaml",
        *STACKS[stack].files,
        str((recipe.path / "config.yaml").relative_to(ROOT)),
        *LLM_LAYERS[llm],
    ]


# ============================================================================== results


@dataclass
class StepResult:
    name: str
    status: str  # pass | fail | skip
    detail: str = ""
    lines: list[str] = field(default_factory=list)


@dataclass
class RecipeReport:
    recipe: str
    stack: str
    llm: str
    steps: list[StepResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(s.status != "fail" for s in self.steps)


class CheckFailedError(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailedError(message)


# ============================================================================== expectations


def _value(result: dict[str, Any], key: str) -> Any:
    if key in ("source_id", "title", "chunk_index", "text"):
        return result.get(key)
    current: Any = result.get("metadata", {})
    for part in key.removeprefix("metadata.").split("."):
        current = current.get(part) if isinstance(current, dict) else None
    return current


def _matches(value: Any, rule: Any) -> bool:
    if isinstance(rule, dict):
        ops: dict[str, Callable[[Any, Any], bool]] = {
            "eq": lambda v, x: v == x,
            "ne": lambda v, x: v != x,
            "lt": lambda v, x: v is not None and v < x,
            "lte": lambda v, x: v is not None and v <= x,
            "gt": lambda v, x: v is not None and v > x,
            "gte": lambda v, x: v is not None and v >= x,
            "in": lambda v, x: v in x,
            "exists": lambda v, x: (v is not None) == x,
            "contains": lambda v, x: isinstance(v, str) and x in v,
        }
        return all(ops[op](value, arg) for op, arg in rule.items())
    if isinstance(rule, list):
        return value in rule
    return bool(value == rule)


def expect_results(results: list[dict[str, Any]], exp: dict[str, Any]) -> None:
    ids = [r["source_id"] for r in results]
    if "count" in exp:
        bounds = exp["count"]
        check(len(results) >= bounds.get("min", 0), f"expected at least {bounds.get('min')} results, got {len(ids)}")
        check(len(results) <= bounds.get("max", 10**9), f"expected at most {bounds.get('max')} results, got {len(ids)}")
    if "first" in exp:
        allowed = exp["first"] if isinstance(exp["first"], list) else [exp["first"]]
        check(bool(ids) and ids[0] in allowed, f"rank 1 should be {' or '.join(allowed)}, got {ids[:3]}")
    if "includes" in exp:
        window = ids[: exp.get("within", len(ids))]
        missing = [i for i in exp["includes"] if i not in window]
        check(not missing, f"missing from the top {len(window)}: {missing} (got {window})")
    if "excludes" in exp:
        present = [i for i in exp["excludes"] if i in ids]
        check(not present, f"should not be returned: {present}")
    for key, rule in (exp.get("every") or {}).items():
        bad = [(r["source_id"], _value(r, key)) for r in results if not _matches(_value(r, key), rule)]
        check(not bad, f"every result should have {key} {rule}; these do not: {bad[:3]}")
    if "distinct" in exp:
        values = [_value(r, exp["distinct"]) for r in results]
        repeated = [v for v, n in Counter(map(str, values)).items() if n > 1]
        check(not repeated, f"{exp['distinct']} repeats: {repeated}")
    for key, limit in (exp.get("max_per") or {}).items():
        worst = Counter(str(_value(r, key)) for r in results).most_common(1)
        check(not worst or worst[0][1] <= limit, f"at most {limit} results per {key}, got {worst}")
    for key, minimum in (exp.get("distinct_min") or {}).items():
        seen = {str(_value(r, key)) for r in results}
        check(len(seen) >= minimum, f"expected ≥ {minimum} different {key} values, got {sorted(seen)}")
    if "first_text_contains" in exp:
        needles = exp["first_text_contains"]
        needles = needles if isinstance(needles, list) else [needles]
        text = results[0]["text"] if results else ""
        check(all(n in text for n in needles), f"rank-1 text should contain {needles}: {text[:160]!r}")
    for key, rule in (exp.get("first_metadata") or {}).items():
        check(bool(results), "no results")
        check(_matches(_value(results[0], key), rule), f"rank 1 {key} should be {rule}, got {_value(results[0], key)}")
    if "any_text_contains" in exp:
        needle = exp["any_text_contains"]
        check(any(needle in r["text"] for r in results), f"no result text contains {needle!r}")
    if "balanced" in exp:  # e.g. "```": a code block is never cut in half by the chunker
        token = exp["balanced"]
        split = [r["source_id"] for r in results if r["text"].count(token) % 2]
        check(not split, f"unbalanced {token!r} in chunks of {split}")
    if exp.get("has_context"):
        check(any(r.get("context") for r in results), "results should carry neighbouring context")


def expect_evaluation(report: dict[str, Any], exp: dict[str, Any]) -> None:
    by_name = {v["name"]: v["metrics"] for v in report["variants"]}
    name = exp["variant"]
    check(name in by_name, f"variant {name} not evaluated (have {sorted(by_name)})")
    metrics = by_name[name]
    for metric in ("ndcg", "mrr", "recall", "hit_rate", "precision"):
        if f"{metric}_min" in exp:
            check(
                metrics[metric] >= exp[f"{metric}_min"],
                f"{name} {metric} {metrics[metric]:.3f} < {exp[f'{metric}_min']}",
            )
    for other in exp.get("not_worse_than", []):
        check(
            metrics["ndcg"] >= by_name[other]["ndcg"],
            f"{name} nDCG {metrics['ndcg']:.3f} < {other} {by_name[other]['ndcg']:.3f}",
        )
    for other in exp.get("better_than", []):
        check(
            metrics["ndcg"] > by_name[other]["ndcg"],
            f"{name} nDCG {metrics['ndcg']:.3f} ≤ {other} {by_name[other]['ndcg']:.3f}",
        )


# ============================================================================== steps


class Runner:
    def __init__(self, client: httpx.AsyncClient, recipe: Recipe, stack: str, llm: str, verbose: bool) -> None:
        self.client, self.recipe, self.stack, self.llm, self.verbose = client, recipe, stack, llm, verbose

    def url(self, step: dict[str, Any], suffix: str = "") -> str:
        return f"{API}/collections/{step.get('collection', self.recipe.collection)}{suffix}"

    async def call(self, method: str, url: str, ok: tuple[int, ...] = (200,), **kwargs: Any) -> dict[str, Any]:
        response = await self.client.request(method, url, **kwargs)
        check(response.status_code in ok, f"{method} {url} → {response.status_code}: {response.text[:300]}")
        return response.json() if response.content else {}

    def applies(self, step: dict[str, Any]) -> bool:
        when = step.get("when") or {}
        if "stacks" in when and self.stack not in when["stacks"]:
            return False
        wanted = when.get("llm")
        if wanted == "generative":
            return self.llm != "extractive"
        return not (isinstance(wanted, list) and self.llm not in wanted)

    async def fresh(self, collection: str, external: bool, *, allow_reset: bool = True) -> None:
        """Start from an empty collection (matters on persistent stacks)."""
        base = f"{API}/collections/{collection}"
        if (await self.client.get(base)).status_code == 404:
            return  # created later by the scenario itself
        if not allow_reset:
            page = await self.call("GET", f"{base}/documents", params={"limit": 1})
            check(
                not page["data"],
                f"collection {collection!r} on the target service is not empty; pass --reset to delete its documents",
            )
            return
        while True:
            page = await self.call("GET", f"{base}/documents", params={"limit": 500})
            ids = [s["source_id"] for s in page["data"]]
            if not ids:
                break
            await self.call("POST", f"{base}/documents:delete", json={"source_ids": ids})
        if external:
            job = await self.call("POST", f"{base}/reindex", ok=(202,), params={"recreate": "true"})
            await self.wait(job["data"])

    async def wait(self, job: dict[str, Any], timeout_s: float = 600) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while job["status"] in ("queued", "running"):
            check(time.monotonic() < deadline, f"job {job['id']} did not finish in {timeout_s}s")
            await asyncio.sleep(0.2)
            job = (await self.call("GET", f"{API}/jobs/{job['id']}"))["data"]
        return job

    async def run(self, step: dict[str, Any]) -> StepResult:
        name = step.get("name", next(k for k in step if k not in ("when", "expect", "collection")))
        if not self.applies(step):
            return StepResult(name, "skip", f"needs {step['when']}")
        kind = next(k for k in STEP_KINDS if k in step)
        try:
            detail, lines = await getattr(self, f"step_{kind}")(step, step.get("expect") or {})
            shown = self.verbose or kind == "evaluate"  # metric tables are the point of an evaluate step
            return StepResult(name, "pass", detail, lines if shown else [])
        except CheckFailedError as exc:
            return StepResult(name, "fail", str(exc), list(getattr(exc, "lines", [])))

    # ---------------------------------------------------------------- writes

    async def step_ingest(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        spec = step["ingest"] if isinstance(step["ingest"], dict) else {"file": step["ingest"]}
        body = self.recipe.data(spec["file"])
        mode = spec.get("mode", "sync")
        body["options"] = {**body.get("options", {}), "mode": mode}
        job = (await self.call("POST", self.url(step, "/documents"), ok=(200, 202, 207), json=body))["data"]
        if mode == "async":
            job = await self.wait(job)
            job = (await self.call("GET", f"{API}/jobs/{job['id']}"))["data"]  # includes per-source results
        results = job.get("results") or []
        actions = Counter(r["action"] for r in results)
        failed = [r for r in results if r["action"] == "failed"]
        check(not failed, f"failed sources: {[(r['source_id'], r['error']) for r in failed][:3]}")
        chunks = sum(r["chunks_total"] for r in results)
        if "chunks_min" in exp:
            check(chunks >= exp["chunks_min"], f"expected ≥ {exp['chunks_min']} chunks, got {chunks}")
        return f"{mode} job {job['status']}: {dict(actions)}, {chunks} chunks", []

    async def step_upsert(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        spec = step["upsert"]
        body = self.recipe.data(spec["file"])
        data = await self.call("PUT", self.url(step, f"/documents/{spec['source_id']}"), json=body)
        result = data["data"]["results"][0]
        for key, rule in exp.items():
            field_name, _, bound = key.rpartition("_")
            if bound in ("min", "max") and field_name in result:
                ok = result[field_name] >= rule if bound == "min" else result[field_name] <= rule
                check(ok, f"{field_name} should be {bound} {rule}, got {result[field_name]}")
            else:
                check(result.get(key) == rule, f"{key} should be {rule}, got {result.get(key)}")
        keys = ("action", "version", "chunks_added", "chunks_removed", "chunks_unchanged")
        return ", ".join(f"{k}={result[k]}" for k in keys), []

    async def step_patch_metadata(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        spec = step["patch_metadata"]
        url = self.url(step, f"/documents/{spec['source_id']}/metadata")
        result = (await self.call("PATCH", url, json={"metadata": spec["metadata"]}))["data"]
        check(result.get("chunks_added", 0) == 0, f"a metadata change should not re-embed: {result}")
        return f"{spec['source_id']}: {result.get('action')}, nothing re-embedded", []

    async def step_patch_chunk(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        spec = step["patch_chunk"]
        chunks = (await self.call("GET", self.url(step, f"/documents/{spec['source_id']}/chunks")))["data"]
        target = next((c for c in chunks if c["chunk_index"] == spec["chunk_index"]), None)
        check(target is not None, f"{spec['source_id']} has no chunk {spec['chunk_index']}")
        assert target is not None
        url = self.url(step, f"/chunks/{target['chunk_id']}/metadata")
        await self.call("PATCH", url, json={"metadata": spec["metadata"]})
        return f"chunk {spec['chunk_index']} of {spec['source_id']}: {spec['metadata']}", []

    async def step_delete(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        spec = step["delete"]
        if isinstance(spec, dict):  # every source of a payload file, e.g. a whole site
            ids = [d["source_id"] for d in self.recipe.data(spec["file"])["documents"]]
        else:
            ids = list(spec)
        await self.call("POST", self.url(step, "/documents:delete"), json={"source_ids": ids})
        return f"deleted {len(ids)} source(s)", []

    async def step_create_collection(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        body = step["create_collection"]
        await self.client.delete(f"{API}/collections/{body['name']}")
        await self.call("POST", f"{API}/collections", ok=(201,), json=body)
        return f"collection {body['name']} created", []

    async def step_drop_collection(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        await self.call("DELETE", f"{API}/collections/{step['drop_collection']}", ok=(204,))
        return f"collection {step['drop_collection']} dropped", []

    # ---------------------------------------------------------------- reads

    async def step_search(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        body = dict(step["search"])
        data = (await self.call("POST", self.url(step, "/retrieve"), json=body))["data"]
        results = data["results"]
        lines = [
            f"#{r['rank']} {r['score']:.4f}  {r['source_id']:<34} {(r.get('title') or r['text'])[:60]!r}"
            for r in results
        ]
        try:
            expect_results(results, exp)
        except CheckFailedError as exc:
            exc.lines = lines  # type: ignore[attr-defined]
            raise
        top = ", ".join(r["source_id"] for r in results[:3])
        return f"{body['query']!r} → {top}", lines

    async def step_facts(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        """For long documents: is the answer inside the chunk(s) we return? (`use_context` also counts the
        neighbouring chunks that context expansion attaches — what an LLM would actually read.)"""
        spec = step["facts"] if isinstance(step["facts"], dict) else {"file": step["facts"]}
        within, use_context = spec.get("within", 1), spec.get("use_context", False)
        found, lines = 0, []
        for case in self.recipe.data(spec["file"]):
            body: dict[str, Any] = {"query": case["query"], "top_k": max(within, 3)}
            if "retrieval" in spec:
                body["retrieval"] = spec["retrieval"]
            results = (await self.call("POST", self.url(step, "/retrieve"), json=body))["data"]["results"][:within]
            texts = [(r.get("context") or r["text"]) if use_context else r["text"] for r in results]
            hit = any(case["contains"] in t for t in texts)
            found += hit
            lines.append(f"{'✓' if hit else '✗'} {case['query']!r} needs {case['contains']!r}")
        detail = f"{found}/{len(lines)} answers inside the top {within} chunk(s){' + context' if use_context else ''}"
        if "found_min" in exp and found < exp["found_min"]:
            failure = CheckFailedError(f"{detail}; expected ≥ {exp['found_min']}")
            failure.lines = lines  # type: ignore[attr-defined]
            raise failure
        return detail, lines

    async def step_evaluate(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        spec = step["evaluate"] if isinstance(step["evaluate"], dict) else {"file": step["evaluate"]}
        body = self.recipe.data(spec["file"])
        if "variants" in spec:  # the same labelled cases, different configurations
            body["variants"] = spec["variants"]
        data = (await self.call("POST", self.url(step, "/evaluate"), json=body))["data"]
        lines = [f"{'variant':<22}{'hit':>6}{'recall':>8}{'mrr':>7}{'ndcg':>7}{'p50 ms':>8}"]
        for v in data["variants"]:
            m = v["metrics"]
            lines.append(
                f"{v['name']:<22}{m['hit_rate']:>6.2f}{m['recall']:>8.2f}{m['mrr']:>7.2f}{m['ndcg']:>7.2f}"
                f"{m['latency_ms_p50']:>8.1f}"
            )
        try:
            if exp:
                expect_evaluation(data, exp)
        except CheckFailedError as exc:
            exc.lines = lines  # type: ignore[attr-defined]
            raise
        return f"{len(body['cases'])} cases, best: {data['best']}", lines  # tables are always shown

    async def step_chat(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        body = dict(step["chat"])
        if "question" in body:
            body["messages"] = [{"role": "user", "content": body.pop("question")}]
        data = (await self.call("POST", self.url(step, "/chat"), json=body))["data"]
        citations = data["citations"]
        answer = data["message"]["content"]
        lines = [f"answer: {answer[:300]!r}", f"citations: {[c['source_id'] for c in citations]}"]
        if "citations_min" in exp:
            check(len(citations) >= exp["citations_min"], f"expected ≥ {exp['citations_min']} citations: {lines}")
        if "finish_reason" in exp:
            check(data["finish_reason"] == exp["finish_reason"], f"finish_reason {data['finish_reason']}: {lines}")
        for key, rule in (exp.get("cited_every") or {}).items():
            bad = [c["source_id"] for c in citations if not _matches(_value(c, key), rule)]
            check(not bad, f"citations outside {key}={rule}: {bad}")
        if "answer_contains" in exp:
            check(exp["answer_contains"].lower() in answer.lower(), f"answer lacks {exp['answer_contains']!r}")
        return f"{data['finish_reason']}, {len(citations)} citation(s), model {data['model']}", lines

    async def step_openai_chat(self, step: dict[str, Any], exp: dict[str, Any]) -> tuple[str, list[str]]:
        spec = step["openai_chat"]
        body = {"model": f"rag:{spec.get('collection', self.recipe.collection)}", "messages": spec["messages"]}
        data = await self.call("POST", "/v1/chat/completions", json=body)
        content = data["choices"][0]["message"]["content"]
        check(bool(content.strip()), "empty completion")
        return f"{data['model']}: {content[:80]!r}", [content] if self.verbose else []


STEP_KINDS = [k.removeprefix("step_") for k in vars(Runner) if k.startswith("step_")]


# ============================================================================== service


def _settings_overrides() -> dict[str, Any]:
    """In-process runs get a throwaway registry and their own namespace in shared databases, so a cookbook
    run can never touch the collections of a real deployment on the same Qdrant or PostgreSQL."""
    return {
        "logging": {"level": "WARNING", "access_log": False},
        "registry": {"url": "sqlite+aiosqlite:///:memory:"},
        "vector_store": {"qdrant": {"collection_prefix": "cookbook_"}, "pgvector": {"db_schema": "cookbook"}},
    }


@asynccontextmanager
async def in_process(files: list[str], stack: str) -> AsyncIterator[httpx.AsyncClient]:
    sys.path.insert(0, str(ROOT / "src"))
    from rag_service.api.app import create_app
    from rag_service.core.config import Settings

    settings = Settings.load([ROOT / f for f in files], **_settings_overrides())
    app = create_app(settings)
    container = app.state.container
    await container.start()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://cookbook", timeout=600) as client:
            yield client
    finally:
        await container.stop()


@asynccontextmanager
async def remote(base_url: str, api_key: str | None) -> AsyncIterator[httpx.AsyncClient]:
    headers = {"X-API-Key": api_key} if api_key else {}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=600) as client:
        yield client


async def run_recipe(
    recipe: Recipe,
    stack: str = "offline",
    llm: str = "extractive",
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    reset: bool = False,
    verbose: bool = False,
) -> RecipeReport:
    files = layers(recipe, stack, llm)
    report = RecipeReport(recipe.name, stack, llm)
    service = remote(base_url, api_key) if base_url else in_process(files, stack)
    async with service as client:
        runner = Runner(client, recipe, stack, llm, verbose)
        collections = {recipe.collection, *(s["collection"] for s in recipe.steps if "collection" in s)}
        for name in sorted(collections):
            await runner.fresh(name, STACKS[stack].external or base_url is not None, allow_reset=reset or not base_url)
        for step in recipe.steps:
            report.steps.append(await runner.run(step))
    return report


# ============================================================================== CLI


ICONS = {"pass": "✔", "fail": "✘", "skip": "·"}


def print_report(report: RecipeReport, recipe: Recipe) -> None:
    print(f"\n━━ {recipe.title}  [{report.recipe} · stack={report.stack} · llm={report.llm}]")
    for step in report.steps:
        print(f"  {ICONS[step.status]} {step.name}  — {step.detail}")
        for line in step.lines:
            print(f"      {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipes", nargs="*", help="Recipe names or fragments (default: all)")
    parser.add_argument("--stack", choices=sorted(STACKS), default="offline")
    parser.add_argument("--llm", choices=sorted(LLM_LAYERS), default="extractive")
    parser.add_argument("--list", action="store_true", help="List recipes, stacks and LLM overlays")
    parser.add_argument("--serve", action="store_true", help="Serve the composed configuration of one recipe")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--base-url", help="Check a running service instead of starting one in process")
    parser.add_argument("--api-key")
    parser.add_argument("--reset", action="store_true", help="With --base-url: delete existing documents first")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show ranked results for every search")
    args = parser.parse_args()

    if args.list:
        print("Recipes:")
        for r in all_recipes():
            print(f"  {r.name:<34} {r.summary}")
        print("\nStacks (--stack):")
        for name, s in STACKS.items():
            print(f"  {name:<10} {s.help}")
        print(f"\nLLM overlays (--llm): {', '.join(LLM_LAYERS)}")
        return

    recipes = select(args.recipes)
    if args.serve:
        if len(recipes) != 1:
            raise SystemExit("--serve needs exactly one recipe")
        files = ",".join(layers(recipes[0], args.stack, args.llm))
        env = {**os.environ, "RAG_CONFIG_FILE": files, "RAG__SERVER__PORT": str(args.port)}
        print(f"RAG_CONFIG_FILE={files} RAG__SERVER__PORT={args.port} uv run rag-service")
        raise SystemExit(subprocess.call([sys.executable, "-m", "rag_service"], cwd=ROOT, env=env))

    failed = 0
    for recipe in recipes:
        try:
            report = asyncio.run(
                run_recipe(
                    recipe,
                    args.stack,
                    args.llm,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    reset=args.reset,
                    verbose=args.verbose,
                )
            )
        except CheckFailedError as exc:  # setup problems: unreachable service, non-empty remote collection…
            print(f"\n━━ {recipe.title}\n  ✘ {exc}")
            failed += 1
            continue
        print_report(report, recipe)
        failed += not report.passed
    print(f"\n{len(recipes) - failed}/{len(recipes)} recipe(s) passed on stack={args.stack}, llm={args.llm}")
    sys.stdout.flush()
    # Skip interpreter teardown: onnxruntime (fastembed) can abort while its thread pools are finalised,
    # after all work is done. The service was already stopped cleanly above.
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
