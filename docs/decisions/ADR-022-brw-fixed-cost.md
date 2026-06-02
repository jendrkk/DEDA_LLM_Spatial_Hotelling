# ADR 022 — BRW-Derived Per-Store Fixed Cost: Mapping, Normalisation, and Pricing Inertness

**Status:** Accepted  
**Date:** June 2026  
**Deciders:** Jedrzej Slowinski  
**Supersedes (for accounting only):** ADR-015 rent=0 baseline — see §Relationship to ADR-015 below.

---

## Context

ADR-015 set per-store rent to zero for the baseline and noted that BRW-derived rent enters
profit only as an additive constant and therefore does **not** affect Nash equilibrium prices.
ADR-016 completed the transit-distance matrix; the model is now stable enough to include
correct profit accounting without changing the pricing dynamics.

The Bodenrichtwerte (BRW) dataset (`data/raw/brw_2023_vector.gpkg`,
Senatsverwaltung für Stadtentwicklung, FIS-Broker `s_wfs_brw`, licence dl-de/by-2.0)
provides standard land values in €/m² at the parcel level across all of Berlin.
Rather than converting BRW into a per-m² rent (which requires unobserved store floor
areas and commercial yield rates, see ADR-015 §Rationale), this ADR adopts a simpler
dimensionless mapping: BRW feeds a **size-independent lump-sum fixed cost per period**,
`fixed_cost_j`, stored as a new field on `Firm`.

### Why size-independent?

A size-independent lump sum avoids the unobserved-floor-area problem of the
rent×size channel (ADR-015).  The relative ordering of land costs across stores is
preserved by the normalisation; only the absolute scale is set by the operator via
`rent_scale`.

---

## Decision

### 1. New `Firm.fixed_cost` field

Add `fixed_cost: float = 0.0` to the frozen `Firm` dataclass, placed after `rent` and
before `chain` (both retain their defaults).  Default 0.0 preserves all existing
behaviour when the field is absent.

### 2. Updated profit formula

$$\pi_{jt} = (p_{jt} - c_j)\,D_{jt} - \tfrac{1}{2}\kappa_0\,e_{jt}^2 - R_j \cdot s_j - F_j$$

where $F_j = \texttt{fixed\_cost}_j$.  The implementation in `profit()` (market.py) adds a
`fixed_cost` parameter (array or scalar, default 0.0).  `market_clearing` reads
`fixed_cost` from `city.firms` into a pre-computed per-firm array alongside the existing
`sizes`, `rents`, and `kappa0` arrays.

### 3. BRW → fixed_cost mapping in `load_berlin_city`

Two new keyword arguments:

- `rent_scale: float = 0.0` — master on/off switch and overall scale.
  `0.0` (default) disables fixed costs entirely (ADR-015 baseline preserved).
- `rent_normalization: str = "mean_ratio"` — normalisation method.

When `rent_scale > 0` and the `"brw"` column is present in `supermarkets.parquet`:

1. Coerce `brw` to `float`; fill NaN and non-positive values with the **median**
   of valid BRW values (logged with count).
2. Apply normalisation (all methods produce a dimensionless ratio × rent_scale):

   | Method | Formula | Property |
   |---|---|---|
   | `"mean_ratio"` (default) | $F_j = r_s \cdot \text{brw}_j / \overline{\text{brw}}$ | mean $F = r_s$ exactly |
   | `"median_ratio"` | $F_j = r_s \cdot \text{brw}_j / \operatorname{median}(\text{brw})$ | robust to outliers |
   | `"minmax"` | $F_j = r_s \cdot (\text{brw}_j - \min) / (\max - \min)$ | range $[0, r_s]$ |

3. Store as `Firm.fixed_cost`; keep `rent=0.0` unchanged (the rent×size channel
   remains inert, consistent with ADR-015).

If `rent_scale == 0` or the `"brw"` column is absent, `fixed_cost_j = 0.0` for all
firms and a single INFO log line is emitted.

### 4. Calibration guidance

`rent_scale` should be chosen as a target fraction of mean per-period **GROSS margin**
at Bertrand-Nash prices: $r_s \approx f \cdot \overline{(p_{\text{Nash}} - c_j)\,D_{jt}}$
where $f \in [0.05, 0.15]$ is a plausible commercial rent burden.
Run `scripts/run_baseline.py` to print mean gross margin; set `rent_scale`
accordingly.  **Do not auto-calibrate**: the choice of $f$ is a modelling assumption
that should be stated explicitly in the methods section.

### 5. Configuration

`configs/env/berlin_inner_ring.yaml` adds:
```yaml
rent_scale: 0.0
rent_normalization: mean_ratio
```
`runner.py::run_single_session` forwards both to `load_berlin_city` via
`env_cfg.get(...)`.

---

## Critical inertness caveat

**`fixed_cost` does NOT affect equilibrium prices or the learned pricing policy.**

A per-firm fixed cost $F_j$ is an additive constant in firm $j$'s per-period profit:

$$\pi_{jt} = \underbrace{(p_{jt} - c_j)\,D_{jt} - \tfrac{1}{2}\kappa_0\,e_{jt}^2}_{\text{variable part}} - \underbrace{R_j s_j + F_j}_{\text{fixed part}}$$

The price FOC $\partial \pi_{jt}/\partial p_{jt} = 0$ involves only the variable part.
Therefore:

- (a) **Bertrand-Nash prices, joint-monopoly prices, the Q-table price grid, and
  Calvano Δ are all invariant to `fixed_cost`.**  Phase-0 Q-learning converges to
  the same price grid and the same Δ regardless of `rent_scale`.
- (b) The fixed cost **shifts every `Q(s, a)` value for a given firm by a constant**.
  Because the argmax over actions is unchanged, the learned Phase-0 pricing policy
  is also unaffected.

`fixed_cost` **does** matter at the **entry/exit margin** (Phase 1+): the entrant's
breakeven condition and incumbents' exit threshold depend on absolute profit levels.
This is the primary reason for including the field now rather than later.

---

## Relationship to ADR-015 and ADR-016

This ADR supersedes ADR-015 **only for profit accounting**: the rent=0 *pricing*
baseline is preserved (rent×size channel stays inert).  The BRW→fixed_cost channel
is the new instrument; the exponential sensitivity-run formula from ADR-015 is
**not** implemented (it is superseded by the simpler normalised lump-sum here).
ADR-016 (transit-adjusted distance matrix) is orthogonal and unchanged.

---

## Alternatives Rejected

**BRW → per-m² rent (ADR-015 sensitivity run formula).** Requires unobserved store
floor areas and commercial yield rates.  The size-independent lump sum adopted here
avoids these assumptions while preserving relative land-cost ordering.

**Auto-calibrate rent_scale to a fixed percentage of mean Nash margin.**
Auto-calibration obscures a modelling assumption.  The chosen fraction $f$ should be
stated in the methods section; the operator sets it explicitly.

**Use `rent * size` channel instead of a new `fixed_cost` field.**  Mixing the
two channels would require non-zero `rent` values, which conflicts with the ADR-015
"rent = 0 baseline" convention and would make it harder to distinguish the two cost
channels in future sensitivity analyses.

---

## Consequences

- `Firm.fixed_cost` defaults to 0.0: all existing test fixtures and production code
  are backward-compatible without modification.
- `profit()` signature gains an optional `fixed_cost` parameter (default 0.0):
  all existing callers remain valid.
- `market_clearing` reads `fixed_cost` from `city.firms`; no caller changes needed.
- `load_berlin_city` gains `rent_scale` and `rent_normalization` kwargs (both
  default to the neutral values): existing callers are backward-compatible.
- `run_single_session` forwards the new kwargs from the env config section.
- The thesis methods section should state: "Per-period fixed costs are set to zero in
  the baseline (ADR-015).  A sensitivity run uses Bodenrichtwert-normalised fixed
  costs (rent_scale = X, mean_ratio normalisation) to verify that land-cost
  heterogeneity does not qualitatively alter collusion dynamics or entrant location
  choice.  Fixed costs do not affect equilibrium prices (see ADR-022)."
