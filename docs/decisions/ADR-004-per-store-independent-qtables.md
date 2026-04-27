# ADR 004 — Per-Store Independent Q-Tables

**Status:** Accepted  
**Date:** April 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The simulation includes ~100–150 incumbent stores distributed across ~9 chains. Each store requires a pricing and effort policy learned by tabular Q-learning. A design choice must be made about whether stores within the same chain share a single Q-table or maintain independent Q-tables.

A shared Q-table would mean all stores of, say, Edeka consult the same state→action mapping. An independent Q-table means each Edeka store learns its own policy from its own local experience without any direct knowledge transfer to or from sister stores.

---

## Decision

Each store maintains its own independent Q-table. Q-tables are never shared between stores, even if they belong to the same chain and the same group assignment.

---

## Rationale

**Economic motivation.** Store managers in real retail chains act on local information — they observe their immediate competitors, their own store's performance, and the envelope passed down from HQ. They do not observe what the Prenzlauer Berg manager is doing in the Mitte store. Shared Q-tables would imply instantaneous cross-store experience transfer within a chain, which has no economic counterpart. The model's information architecture (§2 of `agent_simulation_technical_report.md`) is designed to mirror real delegation, and per-store independence is a necessary part of that.

**Within-chain price dispersion is a finding, not a bug.** Two stores of the same chain in similar competitive environments may converge to different price policies due to different random initialisations and different local rival histories. This within-chain dispersion is itself an empirically interesting outcome — real chains exhibit it, and the model should be capable of producing it. Shared Q-tables would force all same-chain stores to the same policy, suppressing this variation entirely.

**Backward-compatibility with Calvano.** In Calvano et al. (2020), each firm is an independent Q-learner. The per-store independence is a direct generalisation: each store is a firm with its own learning process. Deviating from this would make the degenerate single-store-per-chain limit no longer equivalent to Calvano's setup, breaking the primary validation benchmark.

**Computational cost is negligible.** At ~10,080 cells per Q-table (lean design with relative action space), 150 stores require ~1.5M float32 values total — approximately 6 MB. There is no memory or speed argument for sharing tables.

---

## Consequences

- Within-chain price dispersion emerges naturally from independent learning and is observable in simulation output.
- Two same-chain stores in similar environments may reach different converged policies; this is expected and should be documented in results.
- Q-table initialisation for the entrant store must be handled explicitly (see ADR-010) since the entrant cannot benefit from the burn-in phase.
- The `store_rl.py` module instantiates one `StoreQLearner` object per store, each with its own Q-table array. There is no shared Q-table registry.

---

## Alternatives Rejected

**Shared Q-table per chain.** Would suppress within-chain price dispersion and misrepresent store manager independence. Rejected on economic grounds.

**Shared Q-table per chain group** (e.g., all contested Edeka stores share one table). More nuanced than full sharing, but still implies cross-store experience transfer that has no real-world counterpart. Also creates a tight coupling between the group assignment system and the RL module that complicates the group extensibility requirement. Rejected.

**Parameter sharing with separate experience replay** (a soft form of transfer learning). Technically interesting but outside the tabular Q-learning framework that Calvano's convergence results apply to. Would require deep RL infrastructure and would break the Calvano validation benchmark. Rejected as out of scope.