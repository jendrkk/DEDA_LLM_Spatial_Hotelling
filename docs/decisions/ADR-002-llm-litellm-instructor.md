# ADR-002: Use LiteLLM + Instructor for provider-agnostic LLM integration

**Status:** Accepted
**Date:** 2024-11

## Context

The project needs to call multiple LLM providers (OpenAI, Anthropic, local
Ollama) with structured JSON output validated by Pydantic.  Using the OpenAI
SDK directly would lock the project to one provider; writing bespoke JSON
parsers per provider is brittle.

## Decision

Use **LiteLLM** as a provider-agnostic HTTP layer and **Instructor** for
automatic Pydantic-validated structured outputs.  Model names are always
pinned to a snapshot (e.g. `gpt-4o-2024-08-06`), never aliases (`gpt-4o`),
to ensure long-term reproducibility of experiments.  Every LLM call is logged
to a JSONL file with the full prompt, response, token count, and wall-clock
time.

## Consequences

* Adding a new LLM provider requires only a LiteLLM model string change.
* Pydantic schemas in `src/hotelling/llm/schemas.py` define the contract.
* Call logs in JSONL format enable post-hoc prompt auditing and cost analysis.
* `litellm` and `instructor` become optional dependencies under `[llm]`.

## References

* https://docs.litellm.ai/
* https://python.useinstructor.com/
* https://platform.openai.com/docs/guides/structured-outputs
