# ADR-023 — Real-Euro Basket Normalisation and Scale Covariance

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The spatial Hotelling demand model expresses consumer utility in euros per
shopping trip:

$$u_{ij} = \alpha_h \cdot q_j + \beta \cdot e_j - p_j - t \cdot d_{ij}$$

where $p_j$ is the basket price, $c_j$ the marginal cost, $t$ the transport
coefficient (EUR per one-way minute), and $d_{ij}$ one-way transit minutes.
Prior baselines (ADR-014) normalised marginal costs to zero and used ad-hoc
price levels (~0.3–0.8) inherited from the Calvano replication.  Structural
calibration against German grocery-market moments requires a coherent euro
scale for prices, costs, and transport disutility.

The multinomial logit choice probabilities depend only on utility *differences*
across alternatives.  A uniform rescaling of all monetary terms leaves shares
unchanged.

---

## Decision

Adopt **euros per shopping trip** as the canonical unit for all monetary
parameters in the structural calibration pipeline.  The representative
**standard-chain basket price** (`basket_price_standard_eur` in
`configs/calibration/targets.yaml`, default 40 EUR) is the **free scale
anchor** — a modelling convention, not an empirically pinned absolute price
level.

Document **scale covariance**: multiplying $p$, $c$, $\alpha \cdot q$, $t
\cdot d$, $a_0$, and $\mu$ by a common positive factor $\lambda$ leaves all
logit market shares and the Calvano collusion index $\Delta$ invariant.

---

## Rationale

- A 40 EUR standard basket is interpretable for Berlin grocery shopping and
  aligns with published basket-price studies (Statista, Thünen-Institut).
- Scale covariance means the anchor is chosen for readability, not
  identification; empirical moments pin ratios and relative levels.
- Separating the euro anchor (this ADR) from transport-cost calibration
  (ADR-024) and marginal-cost calibration (ADR-025) keeps each data source
  auditable in `targets.yaml`.

---

## Implementation

| File | Role |
|------|------|
| `configs/calibration/targets.yaml` | `basket_price_standard_eur: 40.0` scale anchor; all empirical inputs. |
| `src/hotelling/calibration/structural.py` | `compute_marginal_costs` uses basket price × price index. |
| `docs/decisions/ADR-026-structural-moment-calibration.md` | Joint moment solver consumes euro-scaled parameters. |

No changes to `core/`, `env/`, or `simulation/` in this step.

---

## Consequences

- Reported prices and costs in thesis tables are in EUR per basket, not
  dimensionless Calvano units.
- Sensitivity to the anchor (e.g. 35 vs 45 EUR) should be verified; shares
  and $\Delta$ are theoretically invariant, but numerical tolerances in the
  equilibrium solver may introduce tiny drift.
- ADR-014's $c = 0$ normalisation remains the **Calvano validation anchor**
  only; the main-run cost model uses ADR-025 gross-margin costs.

---

## Alternatives Rejected

**Retain Calvano dimensionless prices (c = 0, p ~ 0.5).** Incompatible with
empirical gross-margin and wage-based transport calibration without opaque
rescaling factors.

**Pin basket price from a single observed transaction.** Too noisy; basket
composition varies.  A representative index (standard = 1.0, discount/bio
relative) is more stable.

**Non-euro numeraire (e.g. index points).** Loses direct comparability with
VTT literature and retailer margin reports.
