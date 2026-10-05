"""``provider: claude_lite_llm`` against a fake ``claude`` executable: the real ``claude-lite-llm``
package runs end to end (subprocess, JSON parsing, error classes) without network or a subscription."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from rag_service.application.ingestion import DocumentInput
from rag_service.container import Container
from rag_service.core.config import LLMConfig
from rag_service.core.errors import ConfigurationError, ProviderError, ProviderUnavailableError
from rag_service.core.registry import LLMS
from rag_service.domain.models import ChatMessage, LLMRequest
from tests.conftest import make_settings

pytest.importorskip("claude_lite_llm")

from rag_service.adapters.llm.claude_lite import ClaudeLiteLLM, render_prompt

FAKE_CLI = """#!{python}
import json, os, sys, time
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
start = time.time()
if mode == "slow":
    time.sleep(0.3)
if mode == "hang":
    time.sleep(10)
env = {{k: v for k, v in os.environ.items() if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "ENABLE_CLAUDEAI"))}}
record = {{"argv": sys.argv, "cwd": os.getcwd(), "cwd_entries": os.listdir("."), "start": start, "end": time.time(),
          "env": env}}
with open(os.environ["FAKE_CLAUDE_LOG"], "a") as log:
    log.write(json.dumps(record) + "\\n")
if mode == "rate_limit":
    print(json.dumps({{"is_error": True, "api_error_status": 429, "result": "Claude AI usage limit reached"}}))
    sys.exit(1)
if mode == "auth":
    print(json.dumps({{"is_error": True, "api_error_status": 401, "result": "Not logged in"}}))
    sys.exit(1)
if mode == "crash":
    print("segfault-ish failure", file=sys.stderr)
    sys.exit(2)
prompt = sys.argv[-1]
answer = "The Delta RAD80 is rated at 1.5 sones [1]." if "<source" in prompt else "ok"
print(json.dumps({{
    "type": "result", "subtype": "success", "is_error": False, "result": answer, "stop_reason": "end_turn",
    "session_id": "s-1", "total_cost_usd": 0.01,
    "usage": {{"input_tokens": 100, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 5,
              "output_tokens": 7}},
    "modelUsage": {{"claude-haiku-4-5": {{"inputTokens": 900, "outputTokens": 14}},
                   "claude-opus-5": {{"inputTokens": 100, "outputTokens": 7}}}},
}}))
"""


@pytest.fixture
def fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "bin" / "claude"
    path.parent.mkdir()
    path.write_text(FAKE_CLI.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(tmp_path / "calls.jsonl"))
    return path


def calls(fake_cli: Path) -> list[dict[str, Any]]:
    log = Path(os.environ["FAKE_CLAUDE_LOG"])
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def is_private_dir(path: str) -> bool:
    """Exists, sits outside the project tree (no CLAUDE.md / .claude / .mcp.json above it)."""
    return Path(path).is_dir() and Path.cwd().resolve() not in Path(path).resolve().parents


def make_llm(fake_cli: Path, **overrides: Any) -> ClaudeLiteLLM:
    claude_lite = {"cli_path": str(fake_cli), **overrides.pop("claude_lite", {})}
    return ClaudeLiteLLM(LLMConfig(provider="claude_lite_llm", model="opus", claude_lite=claude_lite, **overrides))


async def test_command_is_pinned_to_text_generation(fake_cli: Path) -> None:
    llm = make_llm(fake_cli)
    response = await llm.generate(
        LLMRequest(system="Be precise.", messages=[ChatMessage("system", "Cite."), ChatMessage("user", "--help me")])
    )
    argv = calls(fake_cli)[0]["argv"]
    assert argv[1] == "-p" and argv[-2:] == ["--", "--help me"]  # a leading '-' can never become a flag
    assert argv[argv.index("--system-prompt") + 1] == "Be precise.\n\nCite."
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--disallowedTools") + 1] == "*"
    assert {"--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"} <= set(argv)
    assert argv[argv.index("--model") + 1] == "opus"
    assert response.text == "ok" and response.finish_reason == "stop"
    assert response.model == "claude-opus-5"  # not the helper model that also appears in modelUsage
    assert (response.usage.input_tokens, response.usage.output_tokens) == (115, 7)
    await llm.close()


async def test_environment_keeps_the_call_on_the_subscription(fake_cli: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")
    llm = make_llm(fake_cli, claude_lite={"oauth_token": "sk-ant-oat01-test"})
    await llm.generate(LLMRequest(messages=[ChatMessage("user", "hi")]))
    record = calls(fake_cli)[0]
    assert "ANTHROPIC_API_KEY" not in record["env"] and "ANTHROPIC_BASE_URL" not in record["env"]
    assert record["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-test"
    assert record["env"]["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    assert record["cwd_entries"] == [] and is_private_dir(record["cwd"])
    await llm.close()
    assert not is_private_dir(record["cwd"])  # the private working directory is removed on close


@pytest.mark.parametrize(
    ("mode", "error", "fragment"),
    [
        ("rate_limit", ProviderUnavailableError, "usage limit"),
        ("auth", ProviderError, "claude login"),
        ("crash", ProviderError, "Claude CLI failed"),
    ],
)
async def test_cli_failures_map_to_provider_errors(
    fake_cli: Path, monkeypatch: pytest.MonkeyPatch, mode: str, error: type[Exception], fragment: str
) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", mode)
    llm = make_llm(fake_cli)
    with pytest.raises(error, match=fragment):
        await llm.generate(LLMRequest(messages=[ChatMessage("user", "hi")]))
    await llm.close()


async def test_timeout_is_reported_as_unavailable(fake_cli: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "hang")
    llm = make_llm(fake_cli, timeout_s=0.5)
    with pytest.raises(ProviderUnavailableError, match="timed out"):
        await llm.generate(LLMRequest(messages=[ChatMessage("user", "hi")]))
    await llm.close()


async def test_oversized_prompt_is_rejected_before_starting_the_cli(fake_cli: Path) -> None:
    llm = make_llm(fake_cli, claude_lite={"max_arg_bytes": 1_024})
    with pytest.raises(ProviderError, match="max_arg_bytes"):
        await llm.generate(LLMRequest(messages=[ChatMessage("user", "x" * 2_000)]))
    assert calls(fake_cli) == []
    await llm.close()


async def test_concurrency_is_bounded(fake_cli: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    llm = make_llm(fake_cli, claude_lite={"max_concurrency": 1})
    request = LLMRequest(messages=[ChatMessage("user", "hi")])
    await asyncio.gather(llm.generate(request), llm.generate(request))
    first, second = sorted(calls(fake_cli), key=lambda r: r["start"])
    assert second["start"] >= first["end"]
    await llm.close()


async def test_stream_emits_the_answer_then_done(fake_cli: Path) -> None:
    llm = make_llm(fake_cli)
    events = [e async for e in llm.stream(LLMRequest(messages=[ChatMessage("user", "hi")]))]
    assert [e.delta for e in events if not e.done] == ["ok"]
    assert events[-1].done and events[-1].usage is not None and events[-1].model == "claude-opus-5"
    await llm.close()


def test_missing_cli_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cli_path"):
        ClaudeLiteLLM(LLMConfig(provider="claude_lite_llm", claude_lite={"cli_path": str(tmp_path / "nope")}))


def test_conversation_rendering_and_registry_aliases() -> None:
    assert render_prompt([ChatMessage("user", "q")]) == "q"
    turns = [ChatMessage("user", "a"), ChatMessage("assistant", "b"), ChatMessage("user", "c")]
    assert render_prompt(turns) == "[User: a]\n\n[Assistant: b]\n\n[User: c]"
    assert all(name in LLMS for name in ("claude_lite_llm", "claude-lite-llm", "claude_lite"))


async def test_chat_runs_through_the_subscription_cli(fake_cli: Path) -> None:
    settings = make_settings(
        llm={"provider": "claude-lite-llm", "model": "opus", "claude_lite": {"cli_path": str(fake_cli)}}
    )
    container = Container.build(settings)
    await container.start()  # builds the LLM eagerly: a missing CLI would fail here
    try:
        await container.ingestion.ingest(
            "products", [DocumentInput(source_id="d", content="The Delta RAD80 is rated at 1.5 sones.")]
        )
        result = await container.chat.answer("products", [ChatMessage("user", "How loud is the Delta RAD80?")])
        assert result.answer == "The Delta RAD80 is rated at 1.5 sones [1]."
        assert result.citations and result.citations[0].source_id == "d"
        assert result.usage.output_tokens == 7
    finally:
        await container.stop()
