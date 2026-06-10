# ADR 029 — Local-Market Price-Summary Q-Learning State

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The default Q-learning state encodes the joint action indices of the *k* nearest
rivals (Euclidean distance between store locations).  With `k_neighbors=1` and
price-only actions (`m_effort=1`, `m=15`) the state space has 15 entries per
agent — tractable and Calvano-compatible (see ADR-004, ADR-005).

Two limitations motivate an alternative:

1. **Information loss.**  A typical inner-ring store competes with ~28 rivals
   in its demand catchment, but the k-neighbors state observes only *k* of
   them.  Rival prices outside the k-nearest set are invisible to the learner.

2. **Effort blow-up.**  With joint price×effort actions (`m_effort=5`,
   `action_size=75`) and `k_neighbors=3`, the state space is
   `75³ = 421 875` entries per agent.  For ~494 stores this implies a
   ~125 GiB Q-table — intractable on commodity hardware.

---

## Decision

Add a second state representation, selectable via `state_mode` in the agent
config (CLI: `--local-sum`):

| Mode | State signal | State size |
|------|-------------|------------|
| `neighbors` (default) | Mixed-radix encoding of *k* rivals' joint action indices | `action_size^k` |
| `local_summary` | Discretized local competitor price statistics | `B^len(summary_stats)` |

In `local_summary` mode each agent observes a binned summary of competitor
*prices* (not joint actions) over a competitor set defined as either:

- **Demand-overlap** (`local_sum_n` null or 0): stores that share at least one
  catchment cell (CSR built from `city.catch_indptr` / `catch_indices`).
- **N-nearest** (`local_sum_n = N > 0`): the *N* geographically closest stores
  (Euclidean KDTree on store locations; self excluded).

Summary statistics (default: `mean` only; optional `min`) are computed per
store over its competitor set, then discretized into `n_price_bins` bins
spanning `[price_grid.min(), price_grid.max()]`.  The flat state index is a
mixed-radix product over the binned statistics:
`state = bin_mean + bin_min * B` when both stats are active.

`state_size = B^len(summary_stats)` is **independent of competitor count** and
composes with effort (`m_effort>1`) without the `action_size^k` explosion.

The default `neighbors` path is unchanged; `current_state_signal()` delegates
to `get_neighbor_actions_arr()` in that mode so existing runs are
byte-identical.

---

## Rationale

**Economic motivation.**  Store managers observe local market conditions —
typical competitor prices in their trade area — not the full joint action
histories of a handful of geographically nearest rivals.  A price-summary state
better matches the information available to a local manager while keeping the
tabular Q-learning framework.

**Tractability with effort.**  `local_summary` with `B=15` and `summary_stats=
["mean"]` yields `state_size=15` regardless of `m_effort`, enabling joint
price×effort experiments without the k-neighbors memory guard firing.

**Demand-overlap competitor set.**  Using catchment co-occurrence ties the
competitor definition to the demand model: if two stores never appear in the
same consumer's choice set, they are not mutual competitors in the state.

---

## Consequences

- `HotellingMarketEnv` precomputes a competitor CSR and price-bin edges when
  `state_mode="local_summary"`.
- `BatchQLearningAgent` accepts an external `state_size` from the env; in
  `local_summary` mode `_encode_states` passes through the `(N,)` index vector.
- `BatchSimulationEngine` threads a generic `state_signal` (shape `(N,k)` or
  `(N,)`) instead of assuming neighbor actions.
- CLI: `--local-sum` / `--local-sum N` sets `state_mode=local_summary`.
- Metadata records `state_mode`, `local_sum_n`, `n_price_bins`, `summary_stats`.

---

## Approximations

**Euclidean vs transit-time distance for N-nearest mode.**  The demand model
uses transit-time minutes (`catch_tt`); the N-nearest competitor definition
uses Euclidean distance between store coordinates (EPSG:3035), matching the
existing k-neighbors construction.  This is a deliberate approximation: store
locations are a cheap proxy when catchment CSR is unavailable or when the user
requests a fixed neighbor count.

**Price summary vs joint actions.**  Effort levels of competitors are not
encoded in `local_summary` state; only their realized prices enter the
summary.  This is acceptable for the Phase-0 effort-activation experiments
where effort dynamics are secondary to price collusion.

---

## Alternatives Rejected

**Raise k_neighbors to match catchment size.**  State space grows as
`action_size^k`; with effort activated this is computationally infeasible.
Rejected.

**Continuous state (function approximation).**  Outside tabular Q-learning;
breaks Calvano validation protocol.  Rejected as out of scope.

**Shared Q-table per chain.**  Rejected on economic grounds (ADR-004).

---

## Cross-references

- ADR-004 — per-store independent Q-tables  
- ADR-005 — relative action space encoding  
- ADR-021 — effort activation (`m_effort>1`)  
- `HotellingMarketEnv.current_state_signal()` — state signal API  
- `scripts/run_baseline.py --local-sum` — CLI switch  
