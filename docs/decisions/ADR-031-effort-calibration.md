# ADR-031 — Effort Parameter Calibration at the Euro Scale

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Jedrzej Slowinski  
**Related:** ADR-028 (structural MoM calibration), ADR-021 (effort activation)

---

## Context

Store effort `e_j` enters consumer utility as `+ β · e_j` and generates a
quadratic cost `½ · κ₀ · e_j²` in the profit function.  Before this ADR the
placeholder values `beta_effort = 0.001` and `kappa0 = 1.0` made effort
**inert**: at `e_max = 10` the maximum willingness-to-pay for full service was
`β · e_max = 0.01` EUR, i.e. **0.025 %** of a 40 EUR basket — far below any
plausible retail service premium.

The price-side structural calibration (ADR-028) pins `(μ, a₀)` to gross-margin
and outside-share targets with effort frozen at zero.  Effort parameters were
never scaled to the euro basket anchor introduced in ADR-023.

As with transport cost `t`, quality `q`, and the outside option `a₀`, the model
is **scale-covariant** in `β · e`: doubling both `β` and all equilibrium
efforts leaves demand unchanged.  We therefore need an explicit structural
normalization for effort, analogous to the price-ladder anchor for `q_S, q_B`.

No clean Berlin-specific dataset exists for retail **service-quality**
elasticity (unlike gross margins or the VTT-based transport cost).  A single
structural knob `X` — the fraction of the basket a representative consumer
would pay for a store at maximum service — is therefore adopted from the
store-choice literature (German grocery is price-dominant; service premiums of
5–15 % of the basket are plausible).

---

## Decision

### 1. Structural knob `X` (effort importance)

Set in `configs/calibration/targets.yaml` as `effort_importance_X` (default
0.10, documented range 0.05–0.15):

$$\beta = \frac{X \cdot p_{\text{basket}}}{e_{\max}}$$

so that willingness-to-pay for full effort equals `X` percent of the standard
basket: `β · e_max = X · basket_price_standard_eur`.

### 2. Interior target `ρ` (equilibrium effort level)

Set `effort_interior_target_rho` (default 0.40, range 0.25–0.50) so that the
**mean** analytic Nash effort at the price-only equilibrium demand satisfies
`mean(e*) = ρ · e_max`.  This keeps effort a live strategic lever for
Q-learners rather than a corner solution.

At the price-only Nash (`β = 0` for the demand solve), compute store demands
`D_j` and `D̄ = mean(D)`.  Then:

$$\kappa_0 = \frac{\beta \cdot \bar D}{\rho \cdot e_{\max}}$$

Under the joint FOC `e*_j = β · D_j / κ₀`, this implies
`e*_j = ρ · e_max · (D_j / D̄)` and `mean(e*) = ρ · e_max`.

### 3. Normalization `e_max = 1.0`

`effort_e_max` in `targets.yaml` and `e_max` in
`configs/agents/qlearning_effort.yaml` are both set to **1.0**, treating effort
as a unitless service intensity in `[0, 1]`.  `β` and `κ₀` absorb all euro
scaling; `e_max` is not written into the env YAML.

### 4. Sequential calibration after the price solve

Effort parameters are computed **after** the `(μ, a₀)` MoM solve
(`calibrate_structural`), using the calibrated city with real `μ, a₀, q, c`.
The price-only Nash (`β = 0`) provides `D̄`; no re-solve of `μ, a₀` is
required for the baseline.

Turning effort on in the joint Nash causes a small **outside-share drift**
(typically < 1 pp) because service utility draws some mass from the outside
option.  Gross margin is approximately unchanged.  If drift is large, re-solve
`μ, a₀` with effort active or lower `X` / `ρ`.

### 5. Verification

`scripts/check_effort_calibration.py` solves the **exact** joint price-effort
Bertrand-Nash with calibrated `(β, κ₀)` and checks:

- `interior_fraction ≥ 0.80` (fraction of stores with `0 < e* < e_max`)
- `|margin_drift| < 0.02` relative to ADR-028 targets

### 6. Sensitivity

`X` and `ρ` are varied in sensitivity runs; they are not identified from data.
Document the chosen values and ranges in the calibration report.

---

## Implementation

| Artifact | Role |
|----------|------|
| `configs/calibration/targets.yaml` | `effort_importance_X`, `effort_interior_target_rho`, `effort_e_max` |
| `compute_effort_params()` in `structural.py` | Analytic `(β, κ₀)` from price-only Nash |
| `scripts/calibrate_structural.py` | Report + write `beta_effort`, `kappa0` to env YAML |
| `scripts/check_effort_calibration.py` | Exact joint-Nash verification |
| `tests/unit/test_effort_calibration.py` | Unit tests for formulas and monotonicity |

Price-only runs (`m_effort = 1` in `qlearning_baseline.yaml`) are unaffected.

---

## Consequences

- Effort is economically meaningful: at defaults, `β ≈ 4` EUR per unit effort
  and WTP for full service is 10 % of the basket.
- Q-learning with `m_effort > 1` operates on a non-degenerate effort grid.
- The calibration is transparent about what is structural (`X`, `ρ`) vs
  data-pinned (`μ`, `a₀`, `t`, `c`, `q`).
