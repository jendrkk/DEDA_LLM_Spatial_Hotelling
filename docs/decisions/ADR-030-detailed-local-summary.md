# ADR 030 — Detailed Local-Summary State (Total + Same-Type)

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

ADR-029 introduced `local_summary` Q-learning state: each store observes a
binned summary of competitor *prices* over a local competitor set (demand-overlap
or N-nearest).  The default channel is the mean price over **all** local rivals
(`state_size = B`).

Stores in vertically differentiated chains (discount / standard / bio) face
different subsets of close substitutes.  Within-tier coordination — e.g. discount
stores responding to other discount stores — is economically distinct from
cross-tier price pressure.  A single aggregate local mean conflates these
channels.

---

## Decision

Add **detailed local summary** (`local_summary_detailed=True`, CLI:
`--local-sum-d`): the state combines **two** binned price channels:

| Channel | Competitor set | Statistic |
|---------|----------------|-----------|
| `all` | Demand-overlap or N-nearest (same as plain local summary) | `mean` |
| `same_type` | Subset of `all` with matching `firm.chain_type` | `mean` |

Flat state index: `bin_all + bin_same_type * B` where `B = n_price_bins`.
`state_size = B²` (default `B=15` → 225 states), independent of competitor
count and tractable with effort (`m_effort>1`).

Plain `--local-sum` (single channel) and default k-neighbors are unchanged.

---

## Rationale

**Economic motivation.** Same-chain-type rivals are the close substitutes whose
payoffs interact most directly; conditioning on their local price level is where
within-tier tacit coordination can emerge, while the total-local channel retains
cross-tier market context.

**Tractability.** `B² = 225` states per agent is comparable to k-neighbors with
`k=2` at `action_size=15`, but without the `action_size^k` blow-up when effort
is activated.

**Implementation reuse.** Channels generalize the Prompt-K machinery: each
channel is `(competitor_set, statistic)` with its own CSR; mixed-radix encoding
over bins is unchanged.

---

## Edge case: no same-type local competitor

If a store has no same-type rival in its local competitor set, the CSR row is
empty.  `_local_price_summary` returns the store's **own** price for `cnt==0`,
which bins to a neutral self-reference value.  This is acceptable: the agent
learns that "no same-type signal" corresponds to observing its own price level
in that channel.

---

## Consequences

- `HotellingMarketEnv._ls_channels` lists active `(set_name, stat)` pairs.
- `self._comp_indptr_same` / `_comp_indices_same` built from `all` adjacency
  masked by `chain_type` equality.
- Metadata records `local_summary_detailed` and `local_summary_channels`.
- Mutually exclusive with `--local-sum` on the CLI.

---

## Cross-references

- ADR-029 — local-summary state (plain single-channel mode)  
- ADR-004 — per-store independent Q-tables  
- `scripts/run_baseline.py --local-sum-d` — CLI switch  
