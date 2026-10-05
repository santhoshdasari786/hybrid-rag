"""Typed, validated configuration for the whole service.

Configuration sources, highest precedence first:

1. Environment variables ``RAG__<SECTION>__<KEY>`` (e.g. ``RAG__VECTOR_STORE__PROVIDER=qdrant``).
2. ``.env`` file (same variable names).
3. YAML file(s) listed in ``RAG_CONFIG_FILE`` (comma separated, deep-merged left to right).
   YAML values may reference environment variables as ``${VAR}`` or ``${VAR:-default}``.
4. Defaults declared in this module.

The ``defaults`` pipeline applies to every collection; ``collections.<name>.pipeline`` holds partial
overrides that are deep-merged on top of it. Retrieval/generation settings can additionally be
overridden per request (see :func:`merge_pipeline`).
"""

from __future__ import annotations

import os
import re
import threading
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from rag_service import __version__
from rag_service.core.errors import ConfigurationError
from rag_service.core.utils import deep_merge

CONFIG_FILE_ENV = "RAG_CONFIG_FILE"
DEFAULT_CONFIG_FILE = "config/rag.yaml"


class StrictModel(BaseModel):
    """Base for config sections: unknown keys are rejected so typos fail fast."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)


# =============================================================================== enums


class LengthUnit(StrEnum):
    CHARS = "chars"
    WORDS = "words"
    TOKENS = "tokens"


class DistanceMetric(StrEnum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class Environment(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    TEST = "test"
    STAGING = "staging"
    PROD = "prod"


# =============================================================================== pipeline


class SemanticChunkingConfig(StrictModel):
    breakpoint_percentile: float = Field(
        90.0, ge=50.0, le=99.9, description="Split where adjacent-sentence distance exceeds this percentile."
    )
    buffer_size: int = Field(1, ge=0, le=5, description="Neighbouring sentences combined when embedding each sentence.")


class ChunkingConfig(StrictModel):
    strategy: str = Field("recursive", description="fixed | recursive | sentence | markdown | semantic | <plugin>")
    chunk_size: int = Field(512, ge=16, le=32_000, description="Maximum chunk length in `length_unit`.")
    chunk_overlap: int = Field(64, ge=0, description="Overlap between consecutive chunks in `length_unit`.")
    length_unit: LengthUnit = LengthUnit.TOKENS
    tokenizer: str = Field(
        "approx", description="approx (chars/4) | tiktoken:<encoding> — used when length_unit=tokens."
    )
    min_chunk_size: int = Field(0, ge=0, description="Chunks shorter than this are merged into their neighbour.")
    separators: list[str] | None = Field(None, description="Recursive splitter separators (highest priority first).")
    markdown_header_levels: list[int] = Field(default_factory=lambda: [1, 2, 3])
    semantic: SemanticChunkingConfig = Field(default_factory=SemanticChunkingConfig)
    options: dict[str, Any] = Field(default_factory=dict, description="Plugin-specific options.")

    @model_validator(mode="after")
    def _check_overlap(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class ContextualHeaderConfig(StrictModel):
    """Prepends document context to the text that is embedded (not to the text returned)."""

    enabled: bool = True
    template: str = Field(
        "{title}\n{section}\n\n{text}",
        description="Placeholders: {title}, {section}, {text}, {source_id}. Empty lines are collapsed.",
    )


class LLMContextConfig(StrictModel):
    """Anthropic-style *Contextual Retrieval*: an LLM writes a short situating context per chunk."""

    enabled: bool = False
    max_document_chars: int = Field(24_000, ge=1_000)
    max_concurrency: int = Field(4, ge=1, le=64)
    max_tokens: int = Field(200, ge=16, le=2_000)


class EnrichmentConfig(StrictModel):
    contextual_header: ContextualHeaderConfig = Field(default_factory=ContextualHeaderConfig)
    llm_context: LLMContextConfig = Field(default_factory=LLMContextConfig)


class DenseEmbeddingConfig(StrictModel):
    provider: str = Field("hash", description="hash | fastembed | openai | <plugin>")
    model: str = "hash-384"
    dimension: int | None = Field(None, ge=2, le=65_536, description="Required when the provider cannot infer it.")
    distance: DistanceMetric = DistanceMetric.COSINE
    batch_size: int = Field(32, ge=1, le=2_048)
    query_prefix: str = Field("", description="Instruction prefix for queries (e.g. BGE / E5 style models).")
    document_prefix: str = ""
    options: dict[str, Any] = Field(
        default_factory=dict, description="Provider options (base_url, api_key, threads, cache_dir…)."
    )


class SparseEmbeddingConfig(StrictModel):
    provider: str = Field("lexical", description="lexical | fastembed | <plugin>")
    model: str = "lexical-bm25"
    batch_size: int = Field(32, ge=1, le=2_048)
    options: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(StrictModel):
    dense: DenseEmbeddingConfig | None = Field(default_factory=DenseEmbeddingConfig)
    sparse: SparseEmbeddingConfig | None = Field(default_factory=SparseEmbeddingConfig)
    query_cache_size: int = Field(2_048, ge=0, le=1_000_000, description="LRU cache of query embeddings (0 disables).")

    @model_validator(mode="after")
    def _at_least_one(self) -> EmbeddingConfig:
        if self.dense is None and self.sparse is None:
            raise ValueError("at least one of embedding.dense / embedding.sparse must be configured")
        return self


class FusionConfig(StrictModel):
    method: str = Field("rrf", description="rrf | dbsf | rsf | <plugin>")
    k: int = Field(60, ge=1, le=1_000, description="RRF smoothing constant.")
    weights: dict[str, float] = Field(
        default_factory=lambda: {"dense": 1.0, "sparse": 1.0},
        description="Per-retriever weight; keys are `dense` and `sparse`.",
    )
    execution: Literal["client", "native"] = Field(
        "client",
        description="client: fuse in the service (store-agnostic). native: let the vector store fuse (Qdrant).",
    )


class QueryTransformConfig(StrictModel):
    strategy: str = Field("none", description="none | multi_query | hyde | rewrite | <plugin>")
    num_queries: int = Field(3, ge=1, le=10)
    include_original: bool = True


class RerankConfig(StrictModel):
    enabled: bool = False
    provider: str = Field("fastembed", description="fastembed | http | <plugin>")
    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    candidates: int = Field(30, ge=1, le=500, description="How many fused candidates are re-ranked.")
    batch_size: int = Field(32, ge=1, le=512)
    min_score: float | None = Field(None, description="Drop results whose rerank score is below this value.")
    options: dict[str, Any] = Field(default_factory=dict)


class MMRConfig(StrictModel):
    enabled: bool = False
    lambda_mult: float = Field(0.5, ge=0.0, le=1.0, description="1.0 = pure relevance, 0.0 = pure diversity.")


class ContextExpansionConfig(StrictModel):
    enabled: bool = False
    window: int = Field(1, ge=1, le=5, description="Neighbouring chunks (each side) merged into `context`.")


class RetrievalConfig(StrictModel):
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(5, ge=1, le=100)
    candidate_k: int = Field(50, ge=1, le=1_000, description="Candidates fetched per retriever before fusion.")
    min_score: float | None = Field(None, description="Drop results whose final score is below this value.")
    dense_score_threshold: float | None = Field(None, description="Similarity cut-off applied inside the dense search.")
    dedupe: bool = Field(True, description="Collapse chunks with identical text.")
    group_by: str | None = Field(
        None,
        description="Collapse results that share this field: `source_id` (one document = one hit) or a metadata "
        "key such as `family` (product variants). Empty string or null disables grouping.",
    )
    group_size: int = Field(1, ge=1, le=20, description="Results kept per group when `group_by` is set.")
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    query_transform: QueryTransformConfig = Field(default_factory=QueryTransformConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    mmr: MMRConfig = Field(default_factory=MMRConfig)
    context_expansion: ContextExpansionConfig = Field(default_factory=ContextExpansionConfig)


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the provided context.\n"
    "- Cite every factual statement with the bracketed source number, e.g. [1] or [2][3].\n"
    "- If the context does not contain the answer, say so plainly; do not invent facts.\n"
    "- Be concise and well structured."
)


class GenerationConfig(StrictModel):
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_context_tokens: int = Field(6_000, ge=256, le=900_000)
    max_tokens: int | None = Field(None, ge=16, description="Overrides llm.max_tokens for this pipeline.")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    condense_history: bool = Field(True, description="Rewrite follow-up questions into standalone queries.")
    history_turns: int = Field(6, ge=0, le=50)
    no_context_answer: str = "I couldn't find relevant information in the knowledge base to answer that."


class PipelineConfig(StrictModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class PayloadIndexConfig(StrictModel):
    field: str
    type: Literal["keyword", "integer", "float", "bool", "datetime", "text"] = "keyword"
    tenant: bool = Field(
        False,
        description="The field partitions the collection by tenant (e.g. `site`). Qdrant builds a tenant-optimised "
        "keyword index; other stores ignore the flag.",
    )

    @model_validator(mode="after")
    def _tenant_is_keyword(self) -> PayloadIndexConfig:
        if self.tenant and self.type != "keyword":
            raise ValueError("tenant indexes must have type keyword")
        return self


class CollectionConfig(StrictModel):
    description: str = ""
    pipeline: dict[str, Any] = Field(
        default_factory=dict, description="Partial PipelineConfig deep-merged over `defaults`."
    )
    payload_indexes: list[PayloadIndexConfig] = Field(default_factory=list)


# =============================================================================== infrastructure


class HnswConfig(StrictModel):
    m: int = Field(16, ge=4, le=128)
    ef_construct: int = Field(100, ge=8, le=2_000)
    ef_search: int | None = Field(None, ge=8, le=4_096)


class QdrantConfig(StrictModel):
    url: str | None = Field(
        "http://localhost:6333", description="Server URL. Ignored when `location` or `path` is set."
    )
    api_key: SecretStr | None = None
    location: str | None = Field(None, description="':memory:' for embedded in-process mode.")
    path: str | None = Field(None, description="Embedded on-disk mode directory.")
    prefer_grpc: bool = False
    timeout_s: int = Field(30, ge=1)
    collection_prefix: str = "rag_"
    on_disk_payload: bool = False
    on_disk_vectors: bool = False
    hnsw: HnswConfig = Field(default_factory=HnswConfig)
    quantization: Literal["none", "scalar", "binary"] = "none"
    shard_number: int | None = None
    replication_factor: int | None = None


class PgVectorConfig(StrictModel):
    dsn: SecretStr = SecretStr("postgresql://rag:rag@localhost:5432/rag")
    db_schema: str = "rag"
    table_prefix: str = "chunks_"
    pool_min_size: int = Field(1, ge=1)
    pool_max_size: int = Field(10, ge=1)
    hnsw_m: int = Field(16, ge=4, le=100)
    hnsw_ef_construction: int = Field(64, ge=8, le=1_000)
    ef_search: int = Field(40, ge=1, le=1_000)
    sparse_dimension: int = Field(
        1_048_576, ge=1_024, description="sparsevec dimension; must exceed the largest sparse index."
    )


class VectorStoreConfig(StrictModel):
    provider: str = Field("memory", description="memory | qdrant | pgvector | <plugin>")
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    pgvector: PgVectorConfig = Field(default_factory=PgVectorConfig)
    options: dict[str, Any] = Field(default_factory=dict, description="Options for plugin stores.")


class RegistryConfig(StrictModel):
    url: str = Field(
        "sqlite+aiosqlite:///./data/rag_registry.db",
        description="SQLAlchemy async URL (sqlite+aiosqlite / postgresql+psycopg).",
    )
    echo: bool = False
    auto_create_schema: bool = Field(
        True, description="Create tables at startup (use Alembic migrations in production)."
    )
    store_content: bool = Field(True, description="Keep normalized source text so collections can be re-indexed.")
    pool_size: int = Field(5, ge=1)


class AnthropicOptions(StrictModel):
    refusal_fallback: Literal["default", "off"] = Field(
        "default", description="Server-side fallback when a request is declined by safety classifiers."
    )
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    cache_system_prompt: bool = True


class ClaudeLiteOptions(StrictModel):
    """``provider: claude_lite_llm`` — Claude through the local Claude Code CLI and the operator's own
    subscription (no API key). Personal / development use only; see doc/adr/0006."""

    oauth_token: SecretStr | None = Field(
        None,
        description="Token from `claude setup-token`, passed to the CLI as CLAUDE_CODE_OAUTH_TOKEN. "
        "Unset: the CLI's own login (`claude login`).",
    )
    cli_path: str | None = Field(None, description="Path to the `claude` executable (default: PATH lookup).")
    workdir: str | None = Field(
        None,
        description="Working directory for the CLI. Default: a private empty temp directory, so no project "
        "CLAUDE.md, hooks or .mcp.json are loaded. Choose a directory outside any project.",
    )
    max_concurrency: int = Field(2, ge=1, le=16, description="CLI processes allowed to run at once.")
    max_arg_bytes: int = Field(
        120_000,
        ge=1_024,
        le=1_000_000,
        description="Largest prompt or system prompt accepted; the CLI receives them as command-line arguments "
        "(Linux caps one argument at 128 KiB).",
    )
    extra_args: list[str] = Field(
        default_factory=list, description="Extra CLI flags, e.g. ['--effort', 'low'] or ['--fallback-model', 'sonnet']."
    )


class LLMConfig(StrictModel):
    provider: str = Field(
        "extractive", description="extractive | anthropic | openai | gemini | claude_lite_llm | <plugin>"
    )
    model: str = "extractive"
    api_key: SecretStr | None = Field(None, description="Falls back to the provider SDK's standard env var.")
    base_url: str | None = Field(None, description="OpenAI-compatible endpoints (vLLM, Ollama, Azure, LM Studio…).")
    timeout_s: float = Field(120.0, gt=0)
    max_retries: int = Field(2, ge=0, le=10)
    max_tokens: int = Field(16_000, ge=16)
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Leave unset for models that reject sampling params."
    )
    anthropic: AnthropicOptions = Field(default_factory=AnthropicOptions)
    claude_lite: ClaudeLiteOptions = Field(default_factory=ClaudeLiteOptions)
    options: dict[str, Any] = Field(default_factory=dict)


class CallbackConfig(StrictModel):
    enabled: bool = False
    allowed_hosts: list[str] = Field(
        default_factory=list, description="Exact host names callbacks may target (SSRF guard)."
    )
    timeout_s: float = Field(10.0, gt=0)


DEFAULT_MIME_TYPES = [
    "text/plain",
    "text/markdown",
    "text/html",
    "text/csv",
    "application/json",
    "application/pdf",
]


class IngestionConfig(StrictModel):
    max_documents_per_request: int = Field(100, ge=1, le=10_000)
    max_document_bytes: int = Field(20 * 1024 * 1024, ge=1_024)
    max_chunks_per_document: int = Field(20_000, ge=1)
    default_mode: Literal["sync", "async"] = "async"
    sync_max_documents: int = Field(20, ge=1, description="Larger batches are forced to async mode.")
    worker_concurrency: int = Field(4, ge=1, le=256, description="Sources processed in parallel by the job runner.")
    embed_concurrency: int = Field(2, ge=1, le=64, description="Concurrent embedding calls (CPU/GPU bound).")
    normalize_unicode: bool = True
    collapse_whitespace: bool = True
    allowed_mime_types: list[str] = Field(default_factory=lambda: list(DEFAULT_MIME_TYPES))
    callbacks: CallbackConfig = Field(default_factory=CallbackConfig)


ApiScope = Literal["query", "ingest", "admin"]


def _default_scopes() -> list[ApiScope]:
    return ["query"]


class ApiKeyConfig(StrictModel):
    name: str
    key: SecretStr | None = Field(None, description="Plain key (inject via ${ENV}); hashed in memory at startup.")
    key_sha256: str | None = Field(None, description="Hex SHA-256 of the key — preferred for config files.")
    scopes: list[ApiScope] = Field(default_factory=_default_scopes)
    collections: list[str] = Field(
        default_factory=lambda: ["*"], description="Collections this key may access ('*' = all)."
    )

    @model_validator(mode="after")
    def _one_secret(self) -> ApiKeyConfig:
        if not self.key and not self.key_sha256:
            raise ValueError(f"api key '{self.name}' needs `key` or `key_sha256`")
        return self


class RateLimitConfig(StrictModel):
    enabled: bool = False
    requests_per_minute: int = Field(600, ge=1)
    burst: int = Field(100, ge=1)


class SecurityConfig(StrictModel):
    auth_enabled: bool = False
    api_key_header: str = "X-API-Key"
    api_keys: list[ApiKeyConfig] = Field(default_factory=list)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    cors_allow_origins: list[str] = Field(default_factory=list)
    trusted_hosts: list[str] = Field(default_factory=lambda: ["*"])
    max_request_body_mb: int = Field(50, ge=1)
    security_headers: bool = True

    @model_validator(mode="after")
    def _keys_when_enabled(self) -> SecurityConfig:
        if self.auth_enabled and not self.api_keys:
            raise ValueError("security.auth_enabled is true but no security.api_keys are configured")
        return self


class AppConfig(StrictModel):
    name: str = "advanced-rag-service"
    environment: Environment = Environment.LOCAL
    version: str = __version__
    api_prefix: str = "/api/v1"
    docs_enabled: bool = True
    openai_compat_enabled: bool = True
    root_path: str = ""
    warmup_models: bool = Field(True, description="Load embedding/rerank models at startup (readiness waits for it).")


class ServerConfig(StrictModel):
    host: str = "0.0.0.0"  # noqa: S104 - container default
    port: int = Field(8080, ge=1, le=65_535)
    workers: int = Field(1, ge=1)
    graceful_shutdown_s: int = Field(30, ge=0)
    gzip_min_bytes: int = Field(1_024, ge=0)


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"
    access_log: bool = True


class ObservabilityConfig(StrictModel):
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    tracing_enabled: bool = False
    otlp_endpoint: str | None = None


# =============================================================================== root settings


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def interpolate_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` / ``${VAR:-default}`` in YAML-loaded data."""
    if isinstance(value, dict):
        return {k: interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v) for v in value]
    if not isinstance(value, str):
        return value

    missing: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        missing.append(name)
        return ""

    result = _ENV_PATTERN.sub(_sub, value)
    if missing:
        raise ConfigurationError(f"Environment variable(s) required by configuration are not set: {', '.join(missing)}")
    # A value that was *only* a placeholder resolving to "" means "unset".
    if result == "" and _ENV_PATTERN.fullmatch(value):
        return None
    return result


def load_yaml_files(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"Configuration file must contain a mapping: {path}")
        merged = deep_merge(merged, data)
    result = interpolate_env(merged)
    assert isinstance(result, dict)
    return result


def config_paths_from_env() -> list[Path]:
    raw = os.environ.get(CONFIG_FILE_ENV)
    if raw is None:
        default = Path(DEFAULT_CONFIG_FILE)
        return [default] if default.exists() else []
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


class _YamlSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], data: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self._data = data

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        # Top-level keys starting with `x-` hold YAML anchors shared within a file (docker-compose style);
        # any other unknown key is a typo and must fail like unknown nested keys do.
        unknown = [k for k in self._data if k not in self.settings_cls.model_fields and not k.startswith("x-")]
        if unknown:
            raise ConfigurationError(f"Unknown top-level configuration key(s): {', '.join(sorted(unknown))}")
        return {k: v for k, v in self._data.items() if k in self.settings_cls.model_fields}


class Settings(BaseSettings):
    """Root configuration object. Build it with :meth:`Settings.load`."""

    model_config = SettingsConfigDict(
        env_prefix="RAG__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        validate_default=True,
    )

    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    utility_llm: LLMConfig | None = Field(
        None, description="Cheaper model for query rewriting / contextual enrichment. Defaults to `llm`."
    )
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    defaults: PipelineConfig = Field(default_factory=PipelineConfig)
    collections: dict[str, CollectionConfig] = Field(default_factory=dict)
    allow_dynamic_collections: bool = Field(True, description="Allow creating collections through the API.")
    plugins: list[str] = Field(
        default_factory=list, description="Extra modules imported at startup to register components."
    )

    @field_validator("collections")
    @classmethod
    def _valid_collection_names(cls, value: dict[str, CollectionConfig]) -> dict[str, CollectionConfig]:
        for name in value:
            validate_collection_name(name)
        return value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = _YamlSource(settings_cls, _PENDING_YAML.get("data", {}))
        return init_settings, env_settings, dotenv_settings, yaml_source, file_secret_settings

    @classmethod
    def load(cls, paths: list[Path] | None = None, **overrides: Any) -> Settings:
        """Load settings from YAML files + environment (+ explicit keyword overrides for tests)."""
        files = paths if paths is not None else config_paths_from_env()
        data = load_yaml_files(files) if files else {}
        with _LOAD_LOCK:
            _PENDING_YAML["data"] = data
            try:
                return cls(**overrides)
            except ConfigurationError:
                raise
            except Exception as exc:  # pydantic ValidationError → uniform config error
                raise ConfigurationError(f"Invalid configuration: {exc}") from exc
            finally:
                _PENDING_YAML.pop("data", None)

    # ------------------------------------------------------------------ helpers

    def effective_utility_llm(self) -> LLMConfig:
        return self.utility_llm or self.llm

    def pipeline_for(self, overrides: dict[str, Any] | None) -> PipelineConfig:
        return merge_pipeline(self.defaults, overrides)

    def redacted(self) -> dict[str, Any]:
        """Config dump safe for logs / admin endpoints (secrets masked)."""
        return self.model_dump(mode="json")


# YAML data is handed to `settings_customise_sources` (a classmethod) through this guarded slot.
_PENDING_YAML: dict[str, dict[str, Any]] = {}
_LOAD_LOCK = threading.Lock()

_COLLECTION_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def validate_collection_name(name: str) -> str:
    if not _COLLECTION_NAME.fullmatch(name):
        raise ValueError(
            f"invalid collection name '{name}': use 1-63 chars of lowercase letters, digits, '_' or '-', "
            "starting with a letter or digit"
        )
    return name


def merge_pipeline(base: PipelineConfig, *overrides: dict[str, Any] | None) -> PipelineConfig:
    """Deep-merge partial override dicts onto a pipeline config and re-validate."""
    data = base.model_dump(mode="python")
    for override in overrides:
        if override:
            data = deep_merge(data, override)
    try:
        return PipelineConfig.model_validate(data)
    except Exception as exc:
        raise ConfigurationError(f"Invalid pipeline override: {exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests should build `Settings` explicitly instead."""
    return Settings.load()
