"""Prometheus metrics. Route labels use the route *template* to keep cardinality bounded."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
HTTP_IN_FLIGHT = Gauge("rag_http_requests_in_flight", "In-flight HTTP requests")
INGESTED_SOURCES = Counter("rag_ingested_sources_total", "Sources processed by ingestion", ["collection", "action"])
INGESTED_CHUNKS = Counter("rag_ingested_chunks_total", "Chunks embedded and written", ["collection"])
RETRIEVALS = Counter("rag_retrievals_total", "Retrieval requests", ["collection", "mode"])
RETRIEVAL_RESULTS = Histogram(
    "rag_retrieval_results",
    "Results returned per retrieval",
    ["collection"],
    buckets=(0, 1, 3, 5, 10, 20, 50, 100),
)
LLM_TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["provider", "kind"])
JOBS_ACTIVE = Gauge("rag_jobs_active", "Background jobs currently running")
