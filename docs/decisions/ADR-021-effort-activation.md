# ADR-021 — Effort Margin Activation and Calibration Guard

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The simulation engine and demand model (`core/market.py`) already accommodate a
joint action space over *price* and *effort* per store.  The effort level `e_j`
enters consumer utility via `+ β · e_j` and generates a quadratic cost
`½ · κ₀ · e_j²` in the profit function.  The discrete effort grid has `m_effort`
levels in `linspace(0, e_max, m_effort)`.

In the **Calvano validation baseline** (`configs/agents/qlearning_baseline.yaml`),
effort is frozen at the absorbing point by setting `m_effort: 1`, which collapses
the grid to a single point `{0}`.  This replicates the price-only game of Calvano
et al. (2020) and provides a clean like-for-like reference for the collusion
index Δ.

Activating effort (`m_effort > 1`) raises two concerns that must be resolved
before any results are scientifically valid:

1. **Action-space explosion.** The joint state space grows as `(m · m_effort)^k`,
   so `m_effort = 5` with `k = 1` triples the state space from 15 to 75 per agent.
   This needs to be accounted for in convergence-time expectations.

2. **Corner solutions.** If `β_effort` is too large relative to `κ₀`, the
   best-response effort is always `e_max` (CORNER-HIGH); if too small, it is
   always `0` (CORNER-LOW).  A corner solution means the effort dimension
   contributes no strategic information and the learned Q-table row for effort
   degenerates.  The resulting Δ would be uninterpretable.

---

## Decision

1. The Calvano validation baseline freezes effort at zero by retaining
   `m_effort: 1` in `configs/agents/qlearning_baseline.yaml`.  This file is
   **not changed** by this ADR.

2. Effort is activated by loading `configs/agents/qlearning_effort.yaml`
   (`m_effort: 5`, same price-grid and hyper-parameters as baseline).  This
   config is selected via `--with-effort` in `scripts/run_baseline.py`, or by
   passing `--m-effort N` for arbitrary grid sizes.

3. Before any effort-activated run is accepted as a result, the diagnostic script
   `scripts/check_effort_calibration.py` must report **INTERIOR** for the
   calibrated `(beta_effort, kappa0, e_max)` triple in
   `configs/env/berlin_inner_ring.yaml`.

---

## Rationale

- Keeping effort frozen in the baseline preserves exact comparability with
  Calvano et al. (2020) and ensures Δ is a valid collusion metric for the
  price-only game.
- Providing a named config (`qlearning_effort.yaml`) with a CLI switch makes
  effort activation opt-in and reproducible without touching the baseline config.
- The calibration guard (`check_effort_calibration.py`) ensures the interior
  condition before any production run, preventing degenerate Q-table learning.

---

## Implementation

| File | Change |
|------|--------|
| `configs/agents/qlearning_effort.yaml` | New config; `m_effort: 5`, joint space 15 × 5 = 75. |
| `scripts/run_baseline.py` | `--with-effort` and `--m-effort INT` CLI flags wired into merged config. |
| `scripts/check_effort_calibration.py` | Diagnostic script: numerical sweep + FOC check; prints INTERIOR / CORNER verdict. |

No changes to `core/market.py`, `env/`, or the simulation engine are required;
effort is already fully wired through the joint action index.

---

## Consequences

- `m_effort: 1` (price-only baseline) is the default for all runs not explicitly
  requesting effort.  CI and existing tests are unaffected.
- The action-space growth `(m · m_effort)^k` should be noted in the methods
  section: `m = 15, m_effort = 5, k = 1 → 75` states per agent, converging in
  roughly 5× more steps than the price-only baseline (Calvano 2020, §III.C).
- `e_max` and `beta_effort` require joint recalibration: the FOC target printed
  by `check_effort_calibration.py` gives the analytic starting point.

---

## Alternatives Rejected

**Continuous effort optimisation (gradient-based inner loop).** Would require
a second, differentiable inner optimiser per step.  Inconsistent with the
tabular Q-learning architecture and the Calvano like-for-like comparison.

**Fixed non-zero effort baseline.** Setting `e_j = e_max / 2` as a constant
would preserve the price-only game structure while still affecting demand.  This
would conflate effort and quality effects and is harder to interpret.  Frozen at
zero is cleaner.

**Chain-specific `κ₀`.** ADR-017 fixed `κ₀` as chain-invariant for the baseline.
Chain-specific cost coefficients are noted as a robustness extension; see
ADR-017 for the rationale.  This ADR does not revisit that decision.

---

## References

- Calvano, E. et al. (2020) *Artificial Intelligence, Algorithmic Pricing, and
  Collusion*, AER §II.A – §III.
- ADR-014: Marginal cost normalised to zero for all chain types.
- ADR-017: `κ₀` (kappa0) is chain-invariant in the baseline.
