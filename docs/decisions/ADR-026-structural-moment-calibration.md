# ADR-026 — Structural Moment Calibration (Method of Moments)

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The spatial Hotelling demand model has structural parameters that are not
directly observed: logit scale $\mu$, outside-option utility $a_0$,
chain-quality shifters $q_S$ and $q_B$ (relative to discount), and the
consumer-type quality sensitivity spread $\alpha_H / \alpha_L$.  Meanwhile,
several empirical moments are available from `configs/calibration/targets.yaml`
and Berlin spatial data:

1. Mean gross margin (retailer accounts / EHI).
2. Outside-option share (online + non-local grocery substitution).
3. Discount and standard inside shares (store-count proxy; bio share implied).
4. Bio patronage income gradient (high- vs low-$\pi_H$ cells).

Transport cost $t$ (ADR-024) and marginal costs $c_\tau$ (ADR-025) are
**data-calibrated** outside the joint solver.  The remaining parameters
require inverting the equilibrium mapping from parameters to simulated moments.

---

## Decision

Calibrate $(\mu, a_0, q_S, q_B, \alpha_H/\alpha_L)$ **jointly by method of
moments** against five empirical targets:

| Moment | Target key | Role |
|--------|------------|------|
| Mean gross margin | derived from prices and ADR-025 costs | Pins $\mu$ (scale) |
| Outside share | `outside_share_target` | Pins $a_0$ |
| Standard inside share | `chain_share_target.standard` | Pins $q_S$ |
| Bio inside share | `chain_share_target.bio` | Pins $q_B$ |
| Bio income gradient | `bio_share_income_gradient_target` | Pins $\alpha_H/\alpha_L$ |

The objective minimises squared relative error between simulated and target
moments.  Each simulation call runs the **equilibrium solver** (Bertrand-Nash
prices at fixed $q$, $\alpha$, $t$, $c$) on the inner-ring Berlin grid, then
aggregates choice masses to compute moments.

**Implementation** lands in later prompts (`calibration/structural.py` joint
solver, CLI script, config wiring).  This ADR records the design only.

---

## Rationale

- Method of moments is standard for discrete-choice spatial models when
  closed-form inversion is unavailable.
- Fixing $t$ and $c_\tau$ from external data (ADR-024, ADR-025) reduces
  identification burden and keeps cost side auditable.
- Five moments for five parameters is exactly identified in the baseline;
  over-identification tests can be added if additional moments become available.

---

## Implementation

| Component | Status |
|-----------|--------|
| `configs/calibration/targets.yaml` | Done — empirical targets |
| `compute_transport_cost`, `compute_marginal_costs` | Done — data-only calibrators |
| Joint moment solver + equilibrium inversion | **Later prompt** |
| Injection into simulation `City` / YAML configs | **Later prompt** |

---

## Consequences

- Main-run simulations use empirically grounded $(\mu, a_0, q, \alpha, t, c)$
  rather than hand-tuned Calvano defaults.
- Calibration quality must be reported (moment fit table, sensitivity to
  target ranges documented in `targets.yaml` comments).
- Poor fit may indicate misspecified moments (e.g. store-count shares as
  revenue proxy) — document in robustness section.

---

## Alternatives Rejected

**Calibrate all parameters including $t$ and $c_\tau$ in the joint solver.**
Over-parameterised; transport and margins have direct external anchors.

**Sequential one-at-a-time calibration.** Ignores cross-moment correlations
(e.g. $\mu$ affects all shares simultaneously).

**Bayesian hierarchical calibration.** Heavier infrastructure; MoM sufficient
for thesis scope unless fit is poor.

**Keep hand-tuned $\mu = 0.25$, $a_0 = 0$.** Inconsistent with euro-scale
structural model and documented market shares.
