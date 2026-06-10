# ADR-028 — Quality from Price Ladder; Exogenous Alpha; Two-Parameter MoM Solve

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Jedrzej Slowinski  
**Supersedes:** ADR-026 five-parameter joint calibration design (for the solve step only).

---

## Context

ADR-026 specified a just-identified five-parameter method-of-moments (MoM)
solve for $(\mu, a_0, q_S, q_B, \alpha_H/\alpha_L)$ against five empirical
moments, with $t$ and $c_\tau$ fixed from external data (ADR-024/025).

The first implementation rebuilt the full Berlin City on every residual
evaluation and used store-count chain shares plus the bio income gradient as
calibration targets.  In practice the solver **diverged**:

- $q_S$ and $q_B$ ran to $\mathcal{O}(10^4)$ because the quality parameters
  were unbounded in log-space and weakly identified against count-share proxies.
- $\alpha_H/\alpha_L$ ran to $\mathcal{O}(10^2)$ when the bio gradient target
  conflicted with the compressed $\pi_H$ distribution in inner Berlin.
- Outside-option and discount inside shares collapsed toward zero under extreme
  parameter draws.

Meanwhile `Firm.chain_type` is now stored explicitly by the loader, so
calibration code should not re-infer chain type from the quality value.

---

## Decision

### 1. Data-only qualities from the price ladder

Under the population-weighted normalization $\bar\alpha = 1$ (ADR-023), set:

$$q_S = p_S - p_D, \qquad q_B = p_B - p_D$$

where $p_\tau = \text{basket\_price\_standard\_eur} \times \text{price\_index}_\tau$.

Default ladder ($40 \times \{0.85, 1.00, 1.30\}$): $q_S = 6$ EUR, $q_B = 18$ EUR.

Implemented in `compute_qualities()`.

### 2. Exogenous $\alpha_H / \alpha_L$

Set `alpha_ratio` directly in `configs/calibration/targets.yaml` (default
2.5) from the German organic-consumption income-gradient literature (NVS II,
EsKiMo II, BMEL Ökobarometer).  Derive $(\alpha_L, \alpha_H)$ via
`_alphas_from_ratio()` and the mass-weighted $\bar\pi_H$ from the loaded City.

The model's realised bio income gradient is a **validation output**, not a
solve target.

### 3. Solve only $(\mu, a_0)$

Two MoM targets:

| Moment | Target | Role |
|--------|--------|------|
| Mean gross margin | `gross_margin_common` (or equal-weight chain margins) | Pins $\mu$ |
| Outside share | `outside_share_target` | Pins $a_0$ |

`scipy.optimize.least_squares(method='trf')` on $y = [\log\mu, a_0]$ with
bounds $\mu \in [0.5, 25]$, $a_0 \in [-50, 0]$.

### 4. Build City once; mutate $\mu$ and $a_0$

Load the dense Berlin City **once** per calibration run.  Between residual
evaluations, assign `city.mu` and `city.a0` in place.  Do **not** reload
parquet or rebuild the travel-time matrix inside the optimizer loop.

### 5. Chain classification via `Firm.chain_type`

`_firm_chain_types(city)` reads `firm.chain_type` directly.  Raises
`ValueError` if any firm lacks a valid type — no fallback to quality matching.

### 6. Validation outputs (not in objective)

Report model chain shares against the store-count proxy reference
(0.397 / 0.419 / 0.184) and bio income gradient against
`bio_share_income_gradient_target`.

---

## Rationale

- Price-ladder qualities tie vertical differentiation to observed market price
  premia — the economically interpretable mapping under $\bar\alpha = 1$.
- Store-count shares are a poor revenue proxy and over-identify $q$ when combined
  with the bio gradient target.
- Two well-scaled moments for two parameters ($\mu$, $a_0$) are robust; the
  City-built-once design reduces runtime from $\mathcal{O}(30\,\text{min})$ to
  $\mathcal{O}(1\text{–}2\,\text{min})$.

---

## Implementation

| File | Change |
|------|--------|
| `configs/calibration/targets.yaml` | Remove `chain_share_target`; add `alpha_ratio` |
| `structural.py` | `compute_qualities()`; 2-D `calibrate_structural()` |
| `moments.py` | `_firm_chain_types(city)` from `Firm.chain_type` |
| `scripts/calibrate_structural.py` | Two-moment + validation report |

---

## Consequences

- ADR-026's five-parameter design is **superseded for production calibration**.
  ADR-026 remains historical context for the moment-mapping design.
- Calibrated `q_S`/`q_B` in the written env YAML now reflect the price ladder,
  not optimizer output.
- If chain shares or bio gradient validation fail badly, revise the price
  ladder or `alpha_ratio` assumption — do not re-introduce them as solve
  targets without a better revenue proxy.

---

## Alternatives Rejected

**Retain five-parameter joint solve with tighter bounds.** Bounds mask
identification failure; divergent probes still broke moment computation.

**Solve $\alpha_ratio$ only; fix $q$ from ladder.** Partial fix; $q$ ridge
was the dominant failure mode.

**Re-infer chain type from quality.** Fragile when $q_S \approx q_B$ or
qualities are large; superseded by `Firm.chain_type`.

**Rebuild City per eval with caching.** Correct but $\mathcal{O}(30\,\text{min})$
per run; in-place mutation is safe because kernels read `city.mu`/`city.a0`
at call time.
