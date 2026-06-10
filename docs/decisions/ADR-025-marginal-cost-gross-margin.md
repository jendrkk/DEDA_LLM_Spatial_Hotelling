# ADR-025 — Marginal Cost from Price Ladder and Gross Margin

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

ADR-014 set $c_D = c_S = c_B = 0$ as a Calvano-replication normalisation.
For the structural main run, chain-type price differences and profitability
must reflect observable retail economics.  German sector data (EHI Handel
aktuell; REWE Group / Ahold Delhaize annual reports) report gross margins
by format: discounters leaner (~18%), standard full-range (~24%), organic
premium (~30%).

Prior ad-hoc cost premiums (e.g. 0.10 / 0.13 / 0.17 relative to a discount
numeraire) are not tied to published margin data and are superseded here.

---

## Decision

For each chain type $\tau \in \{\text{discount}, \text{standard}, \text{bio}\}$:

$$p_\tau = \text{basket\_price\_standard\_eur} \times \text{price\_index}_\tau$$

$$\text{margin}_\tau = \begin{cases}
\text{gross\_margin\_common} & \text{if use\_common\_margin} \\
\text{gross\_margin\_by\_chain}[\tau] & \text{otherwise}
\end{cases}$$

$$c_\tau = p_\tau \times (1 - \text{margin}_\tau)$$

**Baseline:** `use_common_margin: true` with `gross_margin_common: 0.22`
(parsimonious single margin).  **Sensitivity:** `use_common_margin: false`
uses chain-specific margins from `gross_margin_by_chain`.

`compute_marginal_costs()` asserts $c_{\text{discount}} < c_{\text{standard}}
< c_{\text{bio}}$ and raises `ValueError` if violated.

**Main-run cost model:** ADR-025 costs feed structural calibration and
production simulations.  **Calvano anchor:** $c = 0$ (ADR-014) remains for
the like-for-like $\Delta$ validation run only.

---

## Rationale

- Gross margin is directly reported by retailers; converting to $c_\tau =
  p_\tau(1 - \text{margin})$ avoids circular dependence on equilibrium prices
  when $p_\tau$ is set from the price ladder anchor.
- A common margin in the baseline reduces free parameters; chain-specific
  margins test robustness.
- With default indices (0.85 / 1.00 / 1.30) and common margin 0.22,
  $c_{\text{discount}} \approx 26.5$, $c_{\text{standard}} = 31.2$,
  $c_{\text{bio}} \approx 40.6$ EUR per basket — preserving the expected
  cost ordering.

---

## Implementation

| File | Role |
|------|------|
| `configs/calibration/targets.yaml` | `price_index`, margins, `use_common_margin`. |
| `src/hotelling/calibration/structural.py` | `compute_marginal_costs()`. |
| `tests/unit/test_calibration_dataonly.py` | Ordering, hand-check, and violation tests. |

---

## Consequences

- Nash prices and $\Delta$ in the main run are no longer invariant to a
  uniform cost shift (unlike ADR-014); this is intentional for empirical
  realism.
- Thesis methods should cite margin sources and state baseline vs
  sensitivity margin specification.
- Fixed costs (ADR-022) remain separate from marginal costs $c_\tau$.

---

## Alternatives Rejected

**Ad-hoc relative premiums (0.10 / 0.13 / 0.17).** Not traceable to published
data; superseded.

**BKartA margin → cost without price ladder.** Requires assuming equilibrium
prices to back out costs; circular for calibration.

**Chain-specific margins in baseline.** More parameters than moments justify;
reserved for sensitivity (`use_common_margin: false`).
