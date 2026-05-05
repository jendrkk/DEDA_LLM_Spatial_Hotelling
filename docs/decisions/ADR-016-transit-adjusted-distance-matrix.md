# ADR 016 — Distance Metric: Transit-Adjusted Travel Time (Option A)

**Status:** Accepted  
**Date:** May 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The transport cost in consumer utility is $t \cdot d_{ij}^2$, where $d_{ij}$ is the distance between consumer cell $i$ and store $j$. The choice of distance metric has a substantive effect on the model: it determines which stores are realistically in competition for which consumers, and whether consumers near transit infrastructure are modelled as effectively "closer" to distant stores.

Three distance metrics were evaluated:

**Option A — Transit-adjusted travel time (GTFS-based effective distance $\tilde{d}_{ij}$).** Replace Euclidean $d_{ij}$ with a travel-time-derived effective distance computed from GTFS routing (walking + transit, using the VBB GTFS feed already in the pipeline). The utility becomes $V_{hij} = \alpha_h q_\theta + \beta e_{jt} - p_{jt} - t\,\tilde{d}_{ij}^2$. This collapses physical distance and transit accessibility into one metric: a consumer near Eberswalder Straße U2 is effectively "closer" to a store two U-Bahn stops away than Euclidean distance would suggest.

**Option B — Consumer-cell transport cost multiplier.** Keep Euclidean distance but make the transport cost coefficient cell-specific:
$$V_{hij} = \alpha_h q_\theta + \beta e_{jt} - p_{jt} - \frac{t}{1 + \psi\, \phi^{\text{hub}}_i}\, d_{ij}^2$$
where $\phi^{\text{hub}}_i \in [0,1]$ is the transit-hub index at the consumer's cell (already constructed from VBB GTFS). High transit access → lower effective $t_i$ → consumer $i$ is willing to shop further away. Introduces one new parameter $\psi \geq 0$; $\psi = 0$ recovers the baseline.

**Option C — Euclidean distance, uniform transport cost.** The simplest option. Ignores transit entirely. Consumer mobility is assumed to be proportional to crow-flies distance everywhere in the inner Ring.

The key insight motivating Options A and B: the prime-location index $\phi_i^{\text{hub}}$ (built from VBB GTFS) already enters the **demand weight** $(\omega_i + \lambda \phi_i)$ at the store's cell, capturing "more people flow through transit-rich store locations." But this does not capture the consumer-side mobility effect: a resident near a U-Bahn station is more willing to travel further to reach a preferred store, even if that store is not near a transit hub. These are two distinct mechanisms; the demand-weight component captures supply-side accessibility, while the distance metric captures demand-side mobility.

---

## Decision

**Option A — Transit-adjusted travel time as the effective distance $\tilde{d}_{ij}$.** This is the theoretically correct operationalisation of transport cost in an urban environment with heterogeneous transit access.

---

## Rationale

**Option A is the most correct economic model.** The quadratic transport cost $t \cdot d^2$ is a reduced-form representation of the full cost of a grocery trip: time cost, cognitive effort, and physical distance. In an urban environment with a dense transit network, the effective trip cost between two points is much better approximated by travel time (walking + transit) than by Euclidean distance. A consumer near Frankfurter Allee U5 faces a lower effective cost to reach a store in Friedrichshain than to reach a store 600m away in the opposite direction with no direct transit connection.

**The mechanism is genuinely distinct from the $\phi_i$ demand weight.** The demand weight $(\omega_i + \lambda \phi_i^{\text{hub}})$ captures that a store near a transit hub attracts passing commuters who were not planning a grocery trip. The distance metric $\tilde{d}_{ij}$ captures that a consumer near a transit hub is more mobile — willing to shop at stores they would not reach on foot. These two effects operate through different channels in the demand equation and should not be collapsed into a single $\phi_i$ term.

**Option B introduces an unidentified parameter $\psi$.** While Option B is computationally lighter (no GTFS routing required), the $\psi$ parameter controlling the transit multiplier on $t$ cannot be calibrated without a natural moment condition. Option A avoids this problem by directly computing travel times from GTFS data, which is observable.

**The computation is feasible as a one-time offline precompute.** The full travel-time matrix between ~8,500 demand cells and ~130 stores (approximately 1.1 million cell-store pairs) can be computed offline using GTFS + a walking model and cached to `data/processed/distance_matrix_transit.parquet`. This is a one-time cost of several hours of computation, not a per-period cost. At simulation time, loading the cached matrix takes less than one second.

**Option C ignores a substantive feature of the Berlin market.** Berlin's inner Ring has one of the densest urban transit networks in Germany. U-Bahn, S-Bahn, tram, and bus lines create a heterogeneous accessibility landscape that Euclidean distance completely ignores. In areas like Wedding (relatively transit-rich) vs. parts of Moabit (less frequent service), the difference in effective consumer reach is substantial. A model that ignores this will mispredict which stores serve which consumers.

---

## Implementation

The transit-adjusted distance matrix is computed offline, once, and cached. It is not recomputed at simulation time.

**Tool chain:**
- VBB GTFS feed (already in `data/raw/`) provides the transit schedule.
- Walking model: 4 km/h, maximum walking leg 500m to/from stop.
- Routing: OpenTripPlanner (OTP) or r5py (Python wrapper for R5 routing engine, specifically designed for GTFS-based isochrone and travel-time matrix computation).
- Reference time: Tuesday 10:00 AM (midday, mid-week, off-peak — representative of a typical grocery trip; sensitivity run at Friday 17:30 peak hour).

**Output:** `data/processed/distance_matrix_transit.parquet` — shape (n_cells, n_stores), values are travel time in seconds (stored as int16 for space efficiency). At query time, convert to effective distance via $\tilde{d}_{ij} = v_{\text{walk}} \cdot (\text{travel\_time}_{ij} / 3600)$ where $v_{\text{walk}} = 4$ km/h, yielding $\tilde{d}_{ij}$ in km. This preserves the dimensional consistency of the $t$ parameter (calibrated in €/km²).

**Fallback:** If the transit matrix computation fails or is not yet available, the simulation loads the Euclidean matrix (`distance_matrix_euclidean.parquet`) instead. This is controlled by `configs/env/berlin_inner_ring.yaml` parameter `distance_metric: transit | euclidean`. The Euclidean fallback enables development iteration before the full transit matrix is computed.

**Euclidean robustness run:** One sensitivity run uses `distance_metric: euclidean` to assess how much the transit adjustment changes qualitative results (entrant location distribution, concentration map, collusion level by neighbourhood).

---

## Consequences

- `src/hotelling/spatial/distance.py` is updated to support `mode ∈ {"euclidean", "transit"}`. The `transit` mode reads `distance_matrix_transit.parquet`; the `euclidean` mode computes L2 distances from cell and store centroids.
- A new script `scripts/compute_transit_matrix.py` (or notebook `notebooks/GEO_03_transit_distance.ipynb`) orchestrates the r5py / OTP batch query and saves the output to Parquet. This script is run once before the main simulation.
- `configs/env/berlin_inner_ring.yaml` adds `distance_metric: transit` as the default; `euclidean` as the fallback.
- The `distance_matrix_transit.parquet` file is **not committed to the repo** (size: ~4 MB as int16, acceptable but generated from raw GTFS data). It is listed in `.gitignore`; the `Makefile` target `make data` triggers its computation.
- The thesis methods section states: "Travel time between each consumer cell and store location is computed using the VBB GTFS feed and the r5py routing engine, at a representative mid-week, off-peak travel time. Effective distance is derived from travel time assuming a 4 km/h walking speed. A robustness run with Euclidean distance confirms qualitative results are unchanged."
- The existing `data-pipeline-osm-zensus-lor-brw` documentation (both vault note and any repo docs) is updated to include the transit distance matrix as Step 5 of the pipeline.

---

## Alternatives Rejected

**Option B — Consumer-cell transport cost multiplier.** Theoretically appealing and computationally lighter. Rejected because the $\psi$ parameter cannot be identified from observable moments without additional data, and because Option A provides the same mechanism at the cost of one offline computation.

**Option C — Euclidean distance, uniform transport cost.** The minimum viable baseline. Retained as a robustness sensitivity run but rejected as the default because it systematically misrepresents consumer reach in transit-rich areas and would bias the spatial HHI map and entrant location choice distribution.

**Network road distance (OSMnx + OSRM).** An intermediate option: walking/cycling distance along the road network, without modelling transit. Captures the non-linearity of the street grid (no crow-flies shortcuts through blocks) but ignores the large speed advantage of transit for distances > 500m. In inner-Ring Berlin, grocery trips beyond 700m are frequently made by transit (U-Bahn, S-Bahn, tram) rather than on foot or by bike. Road distance would understate effective reach for transit-using consumers. Rejected in favour of full transit routing.
