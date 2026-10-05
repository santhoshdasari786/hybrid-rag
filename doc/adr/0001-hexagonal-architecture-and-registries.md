# ADR 0001 — Hexagonal architecture with named component registries

* Status: accepted · Date: 2026-09-27

## Context
The service must let teams choose chunking, embedding, vector store, fusion, reranking and LLM
providers by configuration, and add new ones without forking.

## Decision
* Domain ports as `typing.Protocol`s; adapters implement them; a single composition root
  (`container.py`) wires concrete classes.
* Each strategy family has a `ComponentRegistry`; adapters self-register factories by name;
  configuration selects the name. External packages register through the `rag_service.plugins`
  entry point or the `plugins:` config key.
* Provider-specific settings live in free-form `options` dicts so new components need no schema
  change.

## Consequences
+ Open/closed extension; components are testable in isolation; stores are interchangeable.
− Misspelled names are only caught when a pipeline is built — mitigated by building every
  configured collection's pipeline at startup and on collection creation (fail fast).
