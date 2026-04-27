# ADR 007 — LLM-CEO Calls Are Never Batched Across Chains

**Status:** Accepted  
**Date:** April 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

With ~9 incumbent chains each requiring a CEO call every `T_CEO = 100` periods, a full `T_game = 5,000` run produces approximately 450 CEO calls. Batching all chains into a single API request per epoch (9 chains → 1 call) would reduce this to ~50 calls, saving time and reducing rate-limit pressure.

Batching means sending all chains' states in a single prompt and receiving all chains' envelopes in a single structured response.

---

## Decision

Each chain's LLM-CEO is called in a **separate, independent API request**. Chains are never batched into a single LLM call, regardless of computational cost savings.

---

## Rationale

**Batching destroys the competitive information structure.** The model's defining feature is that chains are strategic competitors with private information. The CEO of Edeka observes Edeka's internal profit history, store performance, and strategy history — information that Rewe's CEO cannot see. Rivals' internal data is deliberately excluded from each CEO's prompt (see §2 of `agent_simulation_technical_report.md`). When chains are batched into a single LLM call, all chains' private data appears in one context window. An LLM does not respect field-level access controls; it reasons over everything it can see. Even with explicit labelling ("CEO 1 may only use fields marked 'private_edeka'"), there is no reliable mechanism to prevent the model from using cross-chain information. Batching therefore creates an implicit super-CEO that can see all competitors' internal state simultaneously — a fundamentally different and economically incorrect agent.

**Batching changes the research question.** A batched call is not measuring how competitive LLM agents behave in a market; it is measuring how a single LLM allocates a market when given perfect information about all players. These are different questions. The first is the research question of this project. The second is closer to a social-planner problem.

**Separation is implementable at acceptable cost.** With batched calls eliminated, the API call volume for `T_game = 5,000` is ~552 calls total (450 CEO + ~100 entrant + 2 entry). At 1,000 RPD (free tier), this fits within a single day. In local mode (Ollama), there are no rate limits at all. The cost of correctness is a factor of ~9 in call count — acceptable.

**Prompt isolation is clean and auditable.** Separate calls mean each CEO's prompt is a self-contained document that can be logged, inspected, and reproduced independently. For any single CEO decision, you can open the JSONL log and see exactly what information that CEO had. This is not possible with batched calls where all information is interleaved.

---

## Consequences

- `agents/chain_ceo.py` receives a single chain's state and returns a single chain's envelope. It has no knowledge of other chains' states.
- The simulation runner calls `chain_ceo.step(chain_state_c)` for each chain `c` in sequence (not in parallel). Parallel API calls are not used — beyond simplicity, sequential calls avoid interleaved logging and make the call order reproducible from the seed.
- Rate limiting is handled by the LiteLLM client with configurable exponential backoff. A `RateLimitError` retries with delay rather than crashing.
- Each call is logged independently to JSONL with `call_type="CEO_EPOCH"`, `chain_id`, `epoch`, token counts, and latency.
- The total call volume scales linearly with `N_chains × (T_game / T_CEO)`. For configurations with more chains or shorter `T_CEO`, budget planning must account for this.

---

## Alternatives Rejected

**Batching with field-level privacy labelling.** Include all chains' data in one prompt but label each field with its owner chain and instruct the model to act as N independent CEOs that respect access controls. Rejected because: (a) LLM instruction-following for access control is not reliable enough for a scientific claim to rest on; (b) there is no way to verify compliance without reading every token of the reasoning trace; (c) even partial cross-chain leakage would invalidate the competitive structure of the model.

**Batching public information only.** Include only publicly observable information (published prices, locations) in a batched call and make CEO calls for private information separately. Reduces savings but does not eliminate the problem: simultaneous reasoning over all chains' public information in one context still allows the model to compute joint-optimal strategies that independent CEOs could not reach. Rejected.

**Parallel async calls.** Call all chain CEOs simultaneously via `asyncio` to reduce wall-clock time while maintaining separate prompts. Preserves information isolation but introduces non-deterministic call ordering and interleaved log entries. Rate limits apply to concurrent requests as well as sequential ones. Not implemented in the baseline; noted as a potential optimisation for production scale.