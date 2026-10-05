SHELL := /bin/bash
.DEFAULT_GOAL := help
UV ?= uv
IT_QDRANT_PORT ?= 6399
IT_PG_PORT ?= 5439

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install all extras + dev tools
	$(UV) sync --all-extras --group dev
	$(UV) run pre-commit install

format: ## Format code
	$(UV) run ruff format src tests scripts cookbook
	$(UV) run ruff check --fix src tests scripts cookbook

lint: ## Lint + type-check
	$(UV) run ruff format --check src tests scripts cookbook
	$(UV) run ruff check src tests scripts cookbook
	$(UV) run mypy

test: ## Unit + API + store-contract tests (offline)
	$(UV) run pytest --cov --cov-report=term-missing

it-up: ## Start throwaway Qdrant + pgvector for integration tests
	docker run -d --rm --name rag-it-qdrant -p $(IT_QDRANT_PORT):6333 qdrant/qdrant:latest
	docker run -d --rm --name rag-it-pg -p $(IT_PG_PORT):5432 -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag pgvector/pgvector:pg16
	@for i in $$(seq 1 30); do docker exec rag-it-pg pg_isready -U rag >/dev/null 2>&1 && curl -sf localhost:$(IT_QDRANT_PORT)/readyz >/dev/null && break; sleep 1; done

it-down: ## Stop integration containers
	-docker rm -f rag-it-qdrant rag-it-pg

test-it: ## Store contract tests against real Qdrant + pgvector (run `make it-up` first)
	RAG_IT_QDRANT_URL=http://localhost:$(IT_QDRANT_PORT) RAG_IT_PG_DSN=postgresql://rag:rag@localhost:$(IT_PG_PORT)/rag \
		$(UV) run pytest tests/integration -v

test-models: ## End-to-end with real FastEmbed models (downloads ~700 MB once)
	RAG_MODELS=1 RAG_IT_QDRANT_URL=http://localhost:$(IT_QDRANT_PORT) RAG_IT_PG_DSN=postgresql://rag:rag@localhost:$(IT_PG_PORT)/rag \
		$(UV) run pytest tests/e2e -s

run: ## Run locally with the offline default config
	RAG_CONFIG_FILE=config/rag.yaml $(UV) run rag-service

run-hybrid: ## Run with Qdrant + BGE + SPLADE (needs Qdrant on :6333)
	RAG_CONFIG_FILE=config/rag.yaml,config/profiles/qdrant-hybrid.yaml $(UV) run rag-service

run-claude-lite: ## Run with Claude via your own subscription (claude-lite-llm; personal use, see ADR 0006)
	RAG_CONFIG_FILE=config/rag.yaml,config/profiles/llm-claude-lite.yaml $(UV) run rag-service

cookbook: ## Run every cookbook recipe on the offline stack (STACK=models|qdrant|pgvector, LLM=…)
	$(UV) run python cookbook/run.py --stack $(or $(STACK),offline) --llm $(or $(LLM),extractive)

migrate: ## Apply registry migrations (REGISTRY_URL must point at PostgreSQL)
	$(UV) run alembic upgrade head

smoke: ## Ingest the sample catalog, retrieve, chat and evaluate against a running service
	$(UV) run python scripts/smoke.py --base-url $${RAG_URL:-http://localhost:8080}

docker-build: ## Build the container image
	docker build -t advanced-rag-service:local .

up: ## docker compose up (API + Qdrant + Postgres)
	docker compose up --build -d

down: ## docker compose down
	docker compose down

.PHONY: help install format lint test it-up it-down test-it test-models run run-hybrid run-claude-lite cookbook migrate smoke docker-build up down
