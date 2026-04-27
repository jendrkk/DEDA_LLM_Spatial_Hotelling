# ADR 008 — Discount Factor Gamma Is Fixed Globally

**Status:** Accepted  
**Date:** April 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The extended strategy envelope includes an exploration rate `epsilon` as a CEO-settable parameter per store group. A natural extension would be to also allow the CEO to set the discount factor `gamma` per group, giving it control over how much weight store agents place on future rewards relative to current rewards.

In Q-learning, `gamma` appears in the Bellman update:

```
Q(s, a) ← Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

A higher `gamma` makes agents more patient — they value future profits more and are therefore more willing to accept short-term losses to sustain collusive cooperation. A lower `gamma` makes agents myopic — they optimise period-by-period and are less likely to sustain punishment strategies. This makes `gamma` appear superficially similar to `epsilon` as a strategic lever the CEO could use.

---

## Decision

`gamma` is a fixed global parameter set once at simulation initialisation. It is identical for all stores, all chains, and all phases. It is never part of the CEO's output schema and cannot be changed between Phase 0 and Phase 2.

---

## Rationale

**Changing gamma invalidates all existing Q-values.** Every Q-value in a store's table is an estimate of the discounted sum of future rewards under a specific `gamma`. This is not a surface-level parameter — it is embedded in the semantics of every number in the table. If the CEO changes `gamma` from `0.90` to `0.95` at epoch `tau`, every Q-value in every store's table becomes an estimate for the wrong objective. The store would need to relearn its entire policy from scratch, making the burn-in and all accumulated experience worthless. For `T_CEO = 100` and `T_game = 5,000`, the CEO would reset 50 Q-tables per store, 50 times — no store would ever converge.

**Epsilon does not have this property.** `epsilon` controls only the *probability of exploration* — how often the agent takes a random action rather than consulting its Q-table. Changing `epsilon` does not affect the Q-values themselves. The Q-table remains valid; the agent just explores more or less aggressively going forward. This is why `epsilon` is safe as a CEO parameter and `gamma` is not.

**The economic interpretation is problematic.** Allowing the CEO to set `gamma` means the CEO is choosing how much the store values its own future profits — effectively setting the store agent's *preferences*, not its *strategy*. This is economically incoherent: a firm's discount factor reflects the time value of money and its owners' patience, not a tactical choice that HQ makes period by period. Epsilon is a strategy (how much to explore vs. exploit); gamma is a fundamental property of the agent.

**The Calvano benchmark requires a fixed gamma.** Calvano et al. (2020) use a fixed `delta = 0.95` across all agents and all periods. The validation benchmark (reproducing `Δ ≈ 0.85` in the degenerate single-store-per-chain limit) requires identical hyperparameters. A CEO-variable gamma would make the benchmark meaningless.

---

## Consequences

- `gamma` (also called `delta` in Calvano notation) is set in `configs/agents/qlearning.yaml` and applied uniformly to all `StoreQLearner` instances.
- The `GroupEnvelope` Pydantic schema contains `{p_bar, delta_p, e_bar, delta_e, epsilon}` — five parameters. `gamma` is not a field.
- The CEO prompt template does not mention `gamma` and does not ask the LLM to reason about it.
- The sensitivity of results to `gamma` is analysed by varying it across separate simulation runs (as a sweep parameter), not within a single run.
- Default value: `gamma = 0.95` (matching Calvano et al. 2020).

---

## Alternatives Rejected

**CEO sets gamma per group, Q-tables reset on change.** Acknowledge the Q-table invalidation problem and accept it: when the CEO changes `gamma`, simply reset the affected stores' Q-tables. Rejected because: (a) stores spend the majority of Phase 2 in unlearned states; (b) the research question becomes confounded — are we measuring collusion, or measuring how quickly stores recover from a Q-table reset?; (c) results would be highly sensitive to `T_CEO` in a way that is not economically interpretable.

**CEO sets gamma but Q-values are rescaled on change.** When `gamma` changes from `g_old` to `g_new`, rescale existing Q-values by the ratio of geometric discount series. This is mathematically complex, not standard in the Q-learning literature, and introduces approximation error. Rejected as too speculative for a baseline implementation.

**Fix gamma during burn-in, allow CEO to set it in Phase 2 only.** The burn-in uses a fixed gamma; at Phase 2, the CEO can adjust it. This solves the burn-in invalidation problem but still causes Q-table invalidation every `T_CEO` periods in Phase 2. Rejected for the same reasons as the main alternative above.