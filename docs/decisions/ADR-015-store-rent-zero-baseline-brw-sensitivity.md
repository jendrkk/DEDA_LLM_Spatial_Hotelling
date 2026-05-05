# ADR 015 — Store Rent R_ℓ = 0 in Baseline; BRW-Derived Sensitivity Run Only

**Status:** Accepted  
**Date:** May 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The full cost structure for store $j$ (§D.1 of `economic_model_specification.md`) includes a periodic rent term:

$$\mathrm{TC}_{jt} = c_{\theta_c} \cdot q_{jt} + \kappa(e_{jt}) + R_{\ell_j} + \frac{F_c}{|\mathcal{J}_c|}$$

where $R_{\ell_j}$ is the per-period rent at store location $\ell_j$. The intent was to ground this in real Berlin land values using the **Bodenrichtwerte (BRW)** dataset (`data/raw/brw_2023_vector.gpkg`), which provides standard land values per m² at the parcel level across all of Berlin, published annually by the Senatsverwaltung für Stadtentwicklung (FIS-Broker `s_wfs_brw`, licence dl-de/by-2.0).

Three questions must be answered to convert BRW into a usable $R_{\ell_j}$:

1. **Store size assumption.** BRW gives €/m² land value; store rent is €/month. The conversion requires an assumed store floor area and a land-value-to-commercial-rent multiplier (typically expressed as an annual rent yield). Neither store floor areas (not in OSM for most locations) nor yield rates (vary by micro-location and lease vintage) are publicly available at the store level.

2. **Rent is a fixed cost in the per-period profit function.** $R_{\ell_j}$ subtracts a constant from profit at each period regardless of price or quantity decisions. It therefore does **not** affect the Nash equilibrium price — the first-order condition $\partial \pi_{jt}/\partial p_{jt} = 0$ is unchanged by an additive constant in $\pi$. Rent affects absolute profit levels and the entry/exit breakeven threshold, but not the equilibrium pricing strategies that are the primary object of analysis.

3. **Entrant entry decision.** The entrant's LLM reasons about the expected profit at candidate sites, which is affected by $R_{\ell}$. However, the LLM evaluates sites holistically (demand density, chain type fit, competitive pressure), and the exact rent level is one of many factors. The directional effect of BRW-derived rent (central sites cost more) is already partially captured by the BRW component of $\phi_i$, which reduces demand from a central location's competitive advantage.

---

## Decision

**Set $R_{\ell_j} = 0$ for all stores in the baseline model.**

One **sensitivity run** includes BRW-derived rent, using the exponential transformation:
$$R_\ell = r_0 \cdot \exp\!\left(\gamma \cdot \frac{\mathrm{BRW}_\ell - \bar{\mathrm{BRW}}}{\sigma_{\mathrm{BRW}}}\right)$$
where $\bar{\mathrm{BRW}}$ and $\sigma_{\mathrm{BRW}}$ are the mean and standard deviation of BRW values across inner-Ring store locations, and $(r_0, \gamma)$ are chosen to reproduce a plausible central-to-peripheral rent ratio of approximately 5:1 (€/month, Mitte vs. outer Ring). The sensitivity run is intended to verify that the $R = 0$ simplification does not qualitatively change collusion dynamics or entrant location choice distributions.

---

## Rationale

**Rent does not affect Nash equilibrium prices.** As a fixed per-period cost, $R_{\ell_j}$ drops out of the pricing first-order condition. The Nash price vector is unchanged whether $R = 0$ or $R = R_{\text{BRW}}$. Since the primary outputs of the simulation — Calvano Δ, IRF, price time series, market shares — are all derived from pricing behaviour, including rent in the baseline adds calibration complexity without changing the scientific question.

**Conversion from BRW to store rent requires unobserved inputs.** The BRW gives €/m² land values, not commercial rents. A defensible conversion would require: (a) store floor area (not available in OSM for most locations; rough estimates from Google Maps satellite measurement would be required), (b) an assumed commercial rent-to-land-value yield (typically 4–8% annually for Berlin commercial property, but varies substantially by micro-location). Introducing these assumed values into the model without empirical grounding would add noise without adding informational content.

**The $R = 0$ normalisation is consistent with ADR-014.** Given that marginal costs are also normalised to zero (ADR-014), profit in the simulation is a measure of mark-up relative to the zero cost baseline. Adding a non-zero, location-specific rent would mix two different economic constructs (operational pricing profit vs. rent-adjusted net profit) in a way that complicates interpretation without adding scientific value in the baseline.

**The entrant sensitivity to rent is best explored in a controlled sensitivity run.** The BRW-derived rent sensitivity run allows direct comparison of entrant location distributions with and without rent pressure, which is a cleaner test of rent's effect than including it in all baseline runs.

---

## Consequences

- `configs/model/costs.yaml` sets `rent_mode: zero` for the baseline. A `rent_mode: brw_exponential` variant is supported.
- `src/hotelling/spatial/brw.py` — the `fetch()` method (currently `NotImplementedError`) is implemented for the sensitivity run. It reads `data/raw/brw_2023_vector.gpkg`, spatial-joins to store locations, applies the exponential transformation with configurable $(r_0, \gamma)$, and exports `data/processed/brw_rents.parquet`.
- In baseline runs: `R_ell` is a length-$n_{\text{stores}}$ zero vector loaded from config.
- In the BRW sensitivity run: `R_ell` is loaded from `brw_rents.parquet`.
- The thesis methods section states: "Periodic store rent is set to zero in the baseline. A robustness run uses Bodenrichtwerte-derived rent (central-to-peripheral ratio ≈ 5:1) to verify that rent pressure does not qualitatively alter collusion dynamics or entrant location choice."
- The `data-pipeline` documentation is updated to note that `brw_rents.parquet` is produced only when `rent_mode = brw_exponential`.

---

## Alternatives Rejected

**Use BRW-derived rent in all runs.** Adding $R_{\ell_j}$ to the baseline makes calibration more complex (two additional free parameters $r_0$ and $\gamma$), does not change any equilibrium price decisions, and introduces unobserved store-size assumptions. Rejected for the baseline; retained as a sensitivity.

**Use a flat per-store rent (same for all locations).** A flat rent is equivalent to $c_\theta + \text{constant}$ in the cost function — observationally equivalent to raising marginal cost by a constant. Since marginal costs are normalised to zero (ADR-014), adding a flat rent would simply shift all profits down by a constant, leaving Δ unchanged. No information is gained; the approach is rejected as redundant.

**Scrape store floor areas from satellite imagery to improve the BRW conversion.** Technically feasible but a large manual data-cleaning effort with diminishing scientific returns. The primary contribution of the project is the agent architecture and pricing dynamics. Noted as a possible extension for a revised version with a Berlin market focus.
