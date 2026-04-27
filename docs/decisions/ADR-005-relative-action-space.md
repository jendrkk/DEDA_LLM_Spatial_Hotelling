# ADR 005 — Relative Action Space for Store RL Agents

**Status:** Accepted  
**Date:** April 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

Store RL agents must choose a price and effort level each period from a discrete action set. Two designs are possible:

- **Absolute action space:** the agent chooses directly from a set of absolute price levels (e.g., `{1.10, 1.20, 1.30, 1.40, 1.50}`). These levels must be discretised from the CEO's envelope each epoch.
- **Relative action space:** the agent chooses a percentage adjustment from its own last-period price (e.g., `{-15%, -10%, -5%, 0%, +5%, +10%, +15%}`), clipped to the current envelope boundaries at execution time.

The choice affects Q-table size, transferability across CEO epochs, and the economic interpretability of learned policies.

---

## Decision

Store RL agents use a **relative action space**. Price actions are defined as percentage adjustments from the previous period's price. Effort actions are defined as discrete steps (`{-1, 0, +1}`) relative to the previous period's effort level. Both are clipped to the current CEO envelope at execution time.

---

## Rationale

**Eliminates own price from the state space.** With absolute actions, the agent needs to know its current absolute price to determine which absolute actions are available — otherwise it cannot tell whether `1.50` is within its current envelope. This requires including own price as a state variable, adding a factor of 5 to the Q-table size. With relative actions, the agent always has the same action set `{-15%, ..., +15%}` regardless of its current price; no absolute price information is needed in the state. This reduces the Q-table from ~50,000+ cells to ~10,080 cells per store.

**Q-tables survive CEO epoch transitions.** When the LLM-CEO shifts the envelope centre `p_bar` from one epoch to the next — say from €1.40 to €1.50 — the absolute price levels available to the store change entirely. Any Q-values learned under the old absolute action set are no longer valid: the agent has to relearn which absolute levels are available and what their values are. With relative actions, the action set `{-15%, ..., +15%}` is unchanged; the Q-values learned in previous epochs remain valid because they encode *relative* competitive dynamics (cut more when rivals are aggressive, hold when rivals hold), not absolute price levels. The burn-in Q-tables carry over across CEO strategy changes.

**Economic interpretability.** A policy of "when a rival undercuts, reduce price by 10%" is more economically interpretable — and more directly comparable to the punishment/cooperation patterns identified in Calvano et al. (2020) — than a policy defined over absolute price levels that are specific to a particular CEO epoch's envelope.

**Envelope-agnosticism enables group differentiation.** When the CEO differentiates envelopes across store groups (heavy vs. easy competition), stores in different groups have different absolute price ranges. With absolute actions, you would need a different action set per group, complicating the Q-table design significantly. With relative actions, all stores share the same action set regardless of their group's envelope centre.

---

## Consequences

- The `StoreQLearner` state space no longer includes own absolute price. The state includes: own relative position in envelope (5 levels), own relative effort position (3 levels), rival undercut indicator (binary), rival effort indicator (binary), group labels (1 bit per active division).
- At execution time, the `step()` method applies the relative adjustment to the current price and clips the result to `[p_bar - delta_p, p_bar + delta_p]`. The clipping logic must be implemented carefully to avoid systematic bias toward envelope boundaries.
- The action space is fixed at 7 price moves × 3 effort moves = 21 actions. This does not change when the CEO changes the envelope.
- When an action is clipped by the envelope boundary, the agent receives the reward for the clipped price, not the intended price. This is economically correct (the store cannot price outside the envelope) but means the agent learns that boundary actions are less valuable than their intended adjustment would suggest. Over time, this creates a natural aversion to operating at envelope boundaries — a desirable property.
- Q-table size: 480 states × 21 actions = 10,080 cells per store, ~6 MB for 150 stores.

---

## Alternatives Rejected

**Absolute action space with dynamic discretisation per epoch.** The CEO's envelope is re-discretised into m levels at each epoch, and the agent's action set is updated accordingly. This preserves Calvano's original absolute-price framework but requires invalidating Q-tables at each CEO epoch, effectively restarting learning. For `T_CEO = 100` and `T_game = 5,000`, there are 50 Q-table resets per store — the agents never converge. Rejected.

**Absolute action space with own price in state.** Avoids Q-table invalidation but requires own price as a state variable, multiplying Q-table size by ~5 (to ~50,000+ cells per store). At 150 stores this is still computationally tractable (~30 MB), but the coverage problem worsens significantly: each state cell is visited less often, requiring longer convergence times to get reliable Q-values. Rejected in favour of the leaner relative design.

**Continuous action space with function approximation (DQN).** Would eliminate the discretisation problem entirely. However, it breaks the tabular Q-learning framework to which Calvano's convergence results apply, introduces hyperparameter sensitivity (network architecture, replay buffer size), and makes the Calvano validation benchmark harder to reproduce. Noted as a future extension if the tabular approach proves insufficient; rejected for the baseline.