from __future__ import annotations

from pathlib import Path

import pytest

from rag_service.core.config import Settings, interpolate_env, merge_pipeline
from rag_service.core.errors import ConfigurationError


def test_defaults_are_offline_capable() -> None:
    s = Settings.load([])
    assert s.vector_store.provider == "memory"
    assert s.defaults.embedding.dense is not None and s.defaults.embedding.dense.provider == "hash"
    assert s.llm.provider == "extractive"
    assert s.defaults.retrieval.fusion.method == "rrf"
    assert s.defaults.retrieval.fusion.k == 60


def test_env_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_URL", "http://qdrant:6333")
    assert interpolate_env({"a": "${MY_URL}", "b": ["x-${MY_URL}"]}) == {
        "a": "http://qdrant:6333",
        "b": ["x-http://qdrant:6333"],
    }
    assert interpolate_env("${UNSET_VAR_XYZ:-fallback}") == "fallback"
    assert interpolate_env("${UNSET_VAR_XYZ:-}") is None
    with pytest.raises(ConfigurationError, match="UNSET_VAR_XYZ"):
        interpolate_env("${UNSET_VAR_XYZ}")


def test_yaml_layering_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("vector_store:\n  provider: qdrant\ndefaults:\n  retrieval:\n    top_k: 7\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("defaults:\n  retrieval:\n    mode: dense\n")
    monkeypatch.setenv("RAG__VECTOR_STORE__PROVIDER", "pgvector")
    s = Settings.load([base, overlay])
    assert s.defaults.retrieval.top_k == 7  # from base
    assert s.defaults.retrieval.mode.value == "dense"  # from overlay
    assert s.vector_store.provider == "pgvector"  # env wins over YAML


def test_unknown_keys_fail_fast(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("defaults:\n  chunking:\n    chunk_sise: 100\n")
    with pytest.raises(ConfigurationError, match="chunk_sise"):
        Settings.load([bad])


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ConfigurationError):
        Settings.load([], defaults={"chunking": {"chunk_size": 100, "chunk_overlap": 100}})


def test_merge_pipeline_deep_merges() -> None:
    s = Settings.load([])
    merged = merge_pipeline(s.defaults, {"retrieval": {"fusion": {"weights": {"dense": 2.0}}}})
    assert merged.retrieval.fusion.weights == {"dense": 2.0, "sparse": 1.0}  # one weight overridden
    assert merged.retrieval.fusion.k == 60  # untouched sibling kept


def test_auth_requires_keys() -> None:
    with pytest.raises(ConfigurationError, match="api_keys"):
        Settings.load([], security={"auth_enabled": True})


def test_collection_names_validated() -> None:
    with pytest.raises(ConfigurationError, match="invalid collection name"):
        Settings.load([], collections={"Bad Name": {}})


def test_redacted_masks_secrets() -> None:
    s = Settings.load([], llm={"provider": "anthropic", "model": "claude-opus-5", "api_key": "sk-secret"})
    assert "sk-secret" not in str(s.redacted())


def test_embedding_fingerprint_ignores_operational_options() -> None:
    from rag_service.application.pipeline import embedding_fingerprint

    base = Settings.load([]).defaults
    moved_cache = merge_pipeline(
        base, {"embedding": {"sparse": {"options": {"cache_dir": "/elsewhere", "threads": 8}}}}
    )
    assert embedding_fingerprint(moved_cache) == embedding_fingerprint(base)
    retuned_bm25 = merge_pipeline(base, {"embedding": {"sparse": {"options": {"k1": 2.0}}}})
    assert embedding_fingerprint(retuned_bm25) != embedding_fingerprint(base)
    new_model = merge_pipeline(base, {"embedding": {"dense": {"model": "hash-128"}}})
    assert embedding_fingerprint(new_model) != embedding_fingerprint(base)


def test_unknown_top_level_keys_fail_but_x_anchors_are_allowed(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text("x-shared: &shared {top_k: 3}\ndefaults:\n  retrieval: {<<: *shared}\n")
    assert Settings.load([good]).defaults.retrieval.top_k == 3
    typo = tmp_path / "typo.yaml"
    typo.write_text("colections: {}\n")
    with pytest.raises(ConfigurationError, match="colections"):
        Settings.load([typo])
