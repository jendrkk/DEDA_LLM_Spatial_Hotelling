# ADR-024 — Transport Cost from Value of Travel Time

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

ADR-020 fixed the functional form: transport disutility is linear in one-way
transit minutes, $-t \cdot d_{ij}$, with $t$ in EUR per minute.  The prior
baseline used $t = 0.5$ EUR/min as a hand-tuned anchor.  The r5py
`TravelTimeMatrix` stores **one-way** origin-cell → destination-store minutes
(median, 60-min cap; ADR-016).  A grocery shopping occasion is a **round
trip**, so the effective disutility must account for travel in both directions.

The value of travel time (VTT) literature provides a principled link between
wages and the monetary cost of time spent travelling.

---

## Decision

Set the transport coefficient as:

$$t = \text{round\_trip\_factor} \times \theta \times \frac{w_{\text{monthly}}}{H_{\text{month}}} \div 60$$

where:

| Symbol | Meaning | Default |
|--------|---------|---------|
| $w_{\text{monthly}}$ | Berlin median gross monthly wage (EUR) | 3955 |
| $H_{\text{month}}$ | Full-time hours per month | 167 |
| $\theta$ (`vtt_wage_ratio`) | Fraction of gross wage as VTT for personal/shopping trips | 0.5 |
| `round_trip_factor` | Multiplier for one-way travel times | 2.0 |

Implemented in `compute_transport_cost()` (`src/hotelling/calibration/structural.py`).
Empirical inputs live in `configs/calibration/targets.yaml`.

**Primary specification:** a single global $t$ derived from the Berlin-wide
median wage.  Spatially heterogeneous $t_i$ (income-varying VTT) is deferred
to ADR-027.

---

## Rationale

- Wardman et al. (2016) meta-analysis and German BVWP 2030 methodology support
  $\theta \in [0.35, 0.50]$ for personal trips; 0.5 is the default.
- `round_trip_factor = 2` correctly maps one-way matrix entries to round-trip
  shopping disutility without re-computing the travel-time matrix.
- With defaults, $t \approx 0.39$ EUR/min — within the prior sanity band
  (0.3–0.5) and consistent with a 10-min one-way trip costing ~7.8 EUR of
  round-trip disutility.

---

## Implementation

| File | Change |
|------|--------|
| `configs/calibration/targets.yaml` | `wage_monthly_gross_eur`, `work_hours_per_month`, `vtt_wage_ratio`, `round_trip_factor`. |
| `src/hotelling/calibration/structural.py` | `compute_transport_cost()` pure function. |
| `tests/unit/test_calibration_dataonly.py` | Hand-calculation and sanity-band tests. |

Equilibrium injection into `City` / simulation configs is handled in a later
prompt.

---

## Consequences

- $t$ is auditable from published wage and VTT sources; no opaque tuning.
- If travel times are ever stored as round-trip minutes, set
  `round_trip_factor: 1.0`.
- Global $t$ ignores within-city income heterogeneity; ADR-027 will address
  spatial VTT if needed.

---

## Alternatives Rejected

**Fixed t = 0.5 EUR/min (ADR-020 anchor).** Not linked to empirical VTT;
superseded for the structural main run.

**Quadratic-in-minutes VTT.** ADR-020 rejected for equilibrium existence
reasons in the deterministic setting; logit smoothness makes linear VTT
sufficient.

**Per-cell t_i in this ADR.** Adds complexity before baseline global
calibration is validated; deferred to ADR-027.
