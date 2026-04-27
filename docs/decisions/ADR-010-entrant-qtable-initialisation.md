# ADR 010 — Entrant Store Q-Table Initialisation Strategies

**Status:** Accepted  
**Date:** April 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The entrant's store-level RL agent is instantiated at Phase 1 (entry), after the incumbent burn-in has already completed. It therefore cannot benefit from Phase 0: while incumbent stores have converged Q-tables encoding thousands of hours of simulated competitive experience, the entrant's store starts with no learned policy.

If the entrant starts with a blank (zero-initialised) Q-table, it will spend the early periods of Phase 2 in purely exploratory behaviour — taking random actions with high probability until its Q-table accumulates enough visits. During this period, the entrant's pricing behaviour is economically meaningless (random), which affects its own profitability and may perturb the incumbents' converged policies.

Separately, the question of how the entrant initialises its Q-table is itself an interesting object of study: does an LLM agent make better decisions about what prior knowledge to inherit than an algorithm?

---

## Decision

Four Q-table initialisation strategies are implemented and selectable via config. The entrant LLM agent may optionally be asked to choose among them (strategies 3 and 4). All four are supported simultaneously without code changes between strategies.

| Strategy | Key | Description |
|---|---|---|
| 1 | `BLANK` | Q-table initialised to zeros. Entrant starts from scratch. |
| 2 | `INHERIT_ALGORITHM` | Algorithm selects the most similar incumbent store by chain type + competitive density group and copies its Q-table. |
| 3 | `INHERIT_LLM_CHOICE` | LLM is shown a list of 3 candidate stores (chain type, group, location zone) and chooses which Q-table to inherit. |
| 4 | `INHERIT_LLM_AUTO` | LLM is asked whether it wants a pre-trained Q-table. If yes, the algorithm selects the most similar store. |

---

## Rationale

**Why not always use BLANK.** A blank Q-table means the entrant is effectively non-strategic for the first several hundred periods of Phase 2. For `T_game = 1,000`, this is a significant fraction of the total game. The entrant's entry decision (Phase 1) is made by an LLM that reasons strategically, but if the store agent subsequently prices randomly, the entry decision cannot be evaluated properly. Inherited Q-tables reduce the cold-start period.

**Why not always use INHERIT_ALGORITHM.** Algorithmic inheritance is correct when the most similar incumbent store is a good proxy for what the entrant will face. But the entrant may be a new chain type (e.g., a bio entrant entering a market dominated by discount and standard stores) with no close analogue. In that case, inheriting from a dissimilar store may be worse than starting blank. The choice between strategies should be configurable.

**Why support LLM-driven initialisation (strategies 3 and 4).** The question "does an LLM make better meta-decisions about its own learning initialisation than a similarity algorithm?" is itself scientifically interesting. Strategy 3 tests whether the LLM can identify a relevant prior when given structured information about candidate stores. Strategy 4 tests whether the LLM has a coherent prior about whether inherited experience is helpful at all. Both are extensions of the broader research question about LLM strategic reasoning.

**Why the algorithm presents 3 candidates (strategy 3), not all stores.** Presenting all ~150 stores would overwhelm the prompt with location and performance data. Three candidates pre-filtered by chain type and group assignment give the LLM a tractable choice while preserving the information relevant to the decision.

**Similarity metric for INHERIT_ALGORITHM and INHERIT_LLM_CHOICE candidates.** Similarity is computed as a weighted combination of: (a) chain type match (binary, highest weight), (b) competitive density group match (binary), (c) LOR neighbourhood status group match (binary), (d) Euclidean distance between store locations (continuous, lowest weight). The exact weights are configurable. If no chain type match exists (new entrant type), only criteria (b–d) are used.

---

## Consequences

- `agents/entrant_llm.py` implements a `QtableInitialiser` that accepts a strategy enum and returns an initialised Q-table array.
- Strategies `INHERIT_LLM_CHOICE` and `INHERIT_LLM_AUTO` require 1 additional LLM call at Phase 1. This call is logged with `call_type="QTABLE_INIT"`.
- The `EntrantEntryDecision` Pydantic schema includes an optional `qtable_init_preference` field. If the strategy is `INHERIT_LLM_CHOICE` or `INHERIT_LLM_AUTO`, this field is populated from the LLM's response; otherwise it is `None`.
- The `QtableInitChoice` schema: `{use_pretrained: bool, chosen_store_id: str | None, rationale: str}`.
- When a Q-table is inherited, it is **copied**, not shared. Subsequent updates to the entrant's Q-table do not affect the source store.
- The initialisation strategy is a configurable parameter in `configs/agents/entrant_llm.yaml`. Default: `INHERIT_ALGORITHM` for production runs; `BLANK` for the Calvano validation benchmark.
- Inherited Q-tables are logged with the source store ID for reproducibility.

---

## Alternatives Rejected

**Always inherit from most similar store (no LLM option).** Simpler implementation, but forecloses the research question about LLM meta-reasoning. Rejected in favour of keeping all four strategies available.

**Transfer learning via Q-value averaging across all same-type stores.** Compute an average Q-table across all stores of the same chain type and use that as the initial table. More statistically stable than a single-store transfer, but: (a) averaging Q-tables that were learned in different competitive environments may produce a table that is not representative of any real competitive situation; (b) the averaging operation has no economic interpretation. Rejected.

**Let the entrant run extra burn-in after entry.** Pause Phase 2, run the entrant through its own burn-in, then resume. Computationally cheap (no LLM calls during the extra burn-in) but economically incorrect: the incumbent CEOs are paused, and the entrant is burning in against a frozen incumbent market that does not reflect Phase 2 dynamics. Rejected.