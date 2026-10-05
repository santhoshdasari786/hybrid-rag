---
description: Add a runnable, checked cookbook recipe for a new use case
agent: agent
argument-hint: "use case, e.g. 'legal contracts with clause search'"
---

Add a cookbook recipe for: ${input:usecase:describe the use case}.

Follow `cookbook/README.md` ("Write your own recipe") and use the closest existing recipe in
`cookbook/recipes/` as the template.

1. Name the traps this use case has (as recipe 02 does for near-identical product names) and design
   synthetic data that contains each of them. Use fictional companies and products; never real brands
   with invented facts.
2. Write `config.yaml` with one collection whose settings each answer a trap: chunking, contextual
   header, payload indexes for every filter, retrieval (`group_by`, fusion weights, context
   expansion) and a domain system prompt. Leave embedding models and the reranker to the stack.
3. Write `scenario.yaml` so that every lesson is one step with a precise `expect`. Use
   `when: {stacks: …}` where offline and model stacks legitimately differ, and
   `when: {llm: generative}` for steps that need an LLM that writes. Add an `evaluate` (several
   sources) or `facts` (one long source) step with labelled queries.
4. Run `uv run python cookbook/run.py <recipe> -v`, then `--stack models` when models are available.
   Set thresholds from what you measured, a little below the observed value. Never tune data or
   expectations just to make a check pass: if a result surprises you, explain it in the README.
5. Write `README.md`: problem, configuration line by line (why), what the checks prove with the
   measured numbers per stack, and production notes.
6. Add the recipe to the table in `cookbook/README.md`, and run `make lint test` (`tests/cookbook`
   picks the recipe up automatically).
