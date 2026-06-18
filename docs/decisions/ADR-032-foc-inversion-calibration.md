# ADR-032 — FOC-Inversion Calibration (Closed-Form Alternative to ADR-026)

**Status:** Accepted
**Date:** June 2026
**Deciders:** Jedrzej Slowinski

---

## Context

The structural method-of-moments solver designed in ADR-026 was intended to
identify the five parameters $(\mu, a_0, q_S, q_B, \alpha_H/\alpha_L)$ jointly
against five empirical moments. In practice the 5-parameter solve failed:
$q$ ran to ~40,500 and $\alpha_H/\alpha_L$ to ~88, driven by conflicting
count-share targets and a weak income-gradient moment on the compressed
$\pi_H$ distribution.

The fallback adopted in the live pipeline was a 2-parameter least-squares
solve for $(\mu, a_0)$ with $(q_S, q_B, \alpha_H/\alpha_L)$ fixed as priors:
$q_S = 6$, $q_B = 18$ taken from the basket price ladder, $\alpha$-ratio
= 2.5 from BMEL Ökobarometer organic-by-income literature.

An identification audit flagged two structural problems with this fallback:

**Assumption 1 — q-from-price-ladder.** Reading $q_S$, $q_B$ off observed
price ratios reverses the causal direction of identification. The Bertrand-
Nash price decomposes as
$p_\tau^N = c_\tau + \mu / (1 - s_\tau)$,
so the price premium of bio over discount reflects three things at once —
quality, cost, and market-power — and cannot be inverted to a quality
intercept without assuming the other two. Fixing $q$ from price implies
that $\mu$ absorbs all misspecification in the share and cost inputs.

**Assumption 2 — 2-parameter MoM conditional on three priors.** The
gross-margin moment used to identify $\mu$ depends on shares
$s_\tau$ that themselves depend on the fixed priors $(q, \alpha, t)$. The
moment is not invariant to misspecified priors; $\mu$ is therefore a
residual that clears one accounting identity given five assumptions, not a
structural estimate.

Both problems stem from the same root: choice probabilities only identify
the *scaled* indirect utility $(V - V_0)/\mu$, so separating $\mu$ from
$q$ requires an exclusion restriction. The 5-parameter joint solve provided
none beyond store counts.

---

## Decision

Add an alternative closed-form calibration method, **FOC-inversion**, that
identifies $(\mu, q_S, q_B)$ from one piece of external demand-share data —
the empirical $s_B/s_D$ inside-market-share ratio — combined with:

* per-chain-type Nash-FOC inversion for $\mu$,
* aggregate logit log-share-ratio inversion for $q_S, q_B$,
* the existing data-side calibration of $t$ (ADR-024) and chain-specific
  $c_\tau$ (ADR-025, with `use_common_margin: false`).

The method is implemented in
`src/hotelling/calibration/foc_inversion.py::calibrate_foc_inversion()`
and is selectable from the CLI:

    python scripts/calibrate_structural.py --method foc_inversion

The legacy MoM solver remains available as `--method mom_2param` (default,
preserving prior runs).

### Step-by-step

**Inputs (from `configs/calibration/targets.yaml`).**

| Symbol | YAML key | Default | Status |
|---|---|---|---|
| $s_{\text{outside}}$ | `outside_share_target` | 0.04 | shared with ADR-026 |
| $s_B/s_D$ | `foc_inversion.s_B_over_s_D` | 0.464 (store-count proxy) | **NEW** — replace with empirical estimate |
| $s_S/s_D$ | `foc_inversion.s_S_over_s_D` | `null` → 207/196 = 1.0561 | defensible because D/S spatial densities are similar |
| $\alpha_H/\alpha_L$ | `foc_inversion.alpha_ratio` (or fallback) | 2.5 | shared with ADR-026 |
| $t$ | computed | ADR-024 | unchanged |
| $c_\tau$ | computed | ADR-025 (chain-specific) | `force_chain_specific_costs: true` |
| $p_\tau$ | basket × index | ADR-023 | unchanged |

**Step A — absolute shares.**

$$s_D = \frac{1 - s_{\text{outside}}}{1 + s_S/s_D + s_B/s_D}, \quad s_S = (s_S/s_D)\, s_D, \quad s_B = (s_B/s_D)\, s_D$$

**Step B — μ from Nash-FOC inversion (per chain type).**

$$\hat\mu_\tau = (p_\tau - c_\tau)(1 - s_\tau), \quad \tau \in \{D, S, B\}$$

Three implied values. In a true single-product-per-type Bertrand-Nash they
would coincide; with multiple firms per type they diverge. The
share-weighted aggregate $\hat\mu = \sum_\tau s_\tau \hat\mu_\tau / \sum_\tau s_\tau$
is the calibrated estimate. The spread $\max - \min$ is the
over-identification diagnostic: a small spread (< 30% relative) indicates
the share inputs are internally consistent with the FOC.

**Step C — population-weighted spatial accessibility per type.**

$$A_{\tau,i} = \sum_{j: \theta_j = \tau} \exp(-t \cdot d_{ij} / \hat\mu), \quad A_\tau = \frac{\sum_i \omega_i A_{\tau,i}}{\sum_i \omega_i}$$

with $\omega_i = $ `cell_pop_i + lambda_phi_i`. Computed in numerically
stable form via per-cell log-sum-exp.

**Step D — closed-form $q$ from the aggregate log-share-ratio identity.**

Under the existing α-normalisation ($\bar\alpha = 1$):

$$q_\tau = \hat\mu \cdot \Big[\ln \frac{s_\tau}{s_D} - \ln \frac{A_\tau}{A_D}\Big] + (p_\tau - p_D), \quad \tau \in \{S, B\}; \quad q_D = 0$$

This is an exact algebraic inversion of the aggregate (type-level) logit
choice probability under the simplifying assumption that intra-type stores
share the same price and quality (which is enforced in the model anyway via
chain-type fixed effects).

**Step E (optional) — $a_0$ refinement.**

When `refine_a0: true`, $a_0$ is adjusted by 1-D root-find
(`scipy.optimize.brentq`) so the model's outside share matches
$s_{\text{outside}}$. This re-pins $a_0$ given the newly identified
$(\mu, q_S, q_B)$ — necessary because the previous $a_0$ in
`berlin_inner_ring.yaml` was conditional on the (incorrect) q-from-price-
ladder priors.

---

## Rationale

* **Mathematical content.** The key identification idea — that the
  Nash-FOC $p_\tau - c_\tau = \mu / (1 - s_\tau)$ inverts $\mu$ directly
  from prices, costs, and shares without any equilibrium iteration — is
  standard in the IO literature (Berry 1994; Berry, Levinsohn & Pakes 1995;
  Nevo 2001). The closed-form $q$ recovery from the aggregate log-share-ratio
  is the BLP "mean-utility inversion" specialised to the chain-type-as-product
  case.
* **Why only $s_B/s_D$.** Three free inside shares $(s_D, s_S, s_B)$
  satisfy $s_D + s_S + s_B = 1 - s_{\text{outside}}$, so two of the three
  are needed in addition to $s_{\text{outside}}$. The $s_S/s_D$ ratio is
  proxied well by the store-count ratio (207/196 = 1.056) because
  standard and discount stores are spatially interleaved across the inner
  Ring with similar densities. Bio, by contrast, is heavily concentrated
  in high-population cells (Mitte, Prenzlauer Berg, Friedrichshain) and
  the store-count proxy (91/196 = 0.46) is systematically downward-biased.
  Replacing $s_B/s_D$ with an empirical estimate is the single highest-
  leverage data improvement and the only one for which a store-count
  proxy is materially misleading.
* **Why chain-specific costs are forced.** A common gross margin (0.22)
  collapses the cost ordering $c_D < c_S < c_B$ across chain types and
  pollutes the FOC-spread diagnostic by injecting a constant bias into
  $(p_\tau - c_\tau)$. The chain-specific margins in
  `gross_margin_by_chain` (0.18 / 0.24 / 0.30) from BKartA/EHI sources
  are used regardless of the global `use_common_margin` flag when
  FOC-inversion is active.
* **No equilibrium solver in steps A–D.** The closed-form pipeline runs
  in milliseconds. The only equilibrium solve is the optional Step E,
  which is a 1-D root-find evaluating the equilibrium at trial $a_0$
  values (~5–10 evaluations).

---

## Implementation

| Component | File | Status |
|-----------|------|--------|
| Pure-function pipeline | `src/hotelling/calibration/foc_inversion.py` | Step 1 |
| YAML schema additions | `configs/calibration/targets.yaml :: foc_inversion` | Step 1 |
| Unit tests | `tests/unit/test_foc_inversion.py` | Step 1 |
| CLI `--method` flag | `scripts/calibrate_structural.py` | Step 2 (this) |
| Diagnostic report block | `scripts/calibrate_structural.py :: _print_foc_diagnostics` | Step 2 (this) |
| Side-by-side diagnostic | `scripts/diagnose_calibration_methods.py` | Step 3 |

---

## Consequences

* The legacy MoM solver (ADR-026) is preserved as the default. Existing
  calibrated YAMLs and downstream runs are unaffected.
* When `--method foc_inversion` is used, the written YAML carries a
  `calibration_method: foc_inversion` stamp plus
  `calibration_s_B_over_s_D`, `calibration_s_S_over_s_D`, and
  `calibration_mu_foc_spread_relative` for auditability.
* The default $s_B/s_D = 0.464$ in `targets.yaml` is the store-count
  proxy and **must be replaced** with an empirical estimate before the
  results carry research weight. Three sources, in increasing quality:
  BKartA Beschluss B2-62/14 (Edeka/Kaiser's Berlin 2014–2015) local-market
  format shares; Google Maps Popular Times catchment-weighted visit
  volumes; GfK / NielsenIQ household-panel format shares (academic
  licence).
* The FOC-spread diagnostic provides a *testable* over-identification
  check: if the three implied $\mu_\tau$ values differ by more than ~30%,
  the input shares (or the chain-specific costs) are inconsistent with
  Bertrand-Nash play, and the calibration should be reported as
  unreliable.

---

## Alternatives Rejected

**Continue the 5-parameter joint MoM with better moments.** Tried; failed
to converge meaningfully (Assumption 2 in the audit). The 2-param fallback
inherited the same identification problem. Requires structural changes
(e.g. instrumental variable for price) that are out of scope.

**Estimate the conditional-logit demand model on household scanner data.**
The principled IO approach (Thomassen et al. 2017; Cleeren et al. 2010).
Requires GfK Haushaltspanel or NielsenIQ Germany under academic licence
(€10k–€50k commercial; possibly available via HU Berlin / DIW affiliation).
Out of scope for this iteration but the right long-run path.

**Use BLP estimation with cost-side instruments.** The classic Berry-
Levinsohn-Pakes (1995) and Nevo (2001) approach. Requires variation in
prices and instruments (e.g. wholesale cost shifters, distribution-
distance instruments). No suitable instrument set is currently available
for Berlin grocery; deferred.

**Replace store-count $s_S/s_D$ with another proxy.** Considered. Standard
and discount stores have similar inner-Ringbahn spatial densities (within
~5%) and similar catchment overlaps. The store-count proxy is materially
accurate for $s_S/s_D$; the bias is concentrated in $s_B/s_D$ which is
explicitly the one input we require externally.
