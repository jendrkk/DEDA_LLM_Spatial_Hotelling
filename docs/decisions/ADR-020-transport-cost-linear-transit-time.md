# ADR-020: Transport cost is linear in transit travel time

## Status
Accepted. Supersedes the transport-cost specification in
economic_model_specification.md (§F.2, §L.2) for the implemented baseline.

## Context
The economic model spec specifies a quadratic Euclidean transport term
`- t * d_ij^2` with `t` in EUR/km^2, motivated by equilibrium existence
(d'Aspremont 1979). ADR-016 replaced Euclidean distance with VBB/DB transit
travel time. The implementation stores travel-time minutes in `City.dist2_km2`
and applies the disutility linearly: `transport_cost * travel_time_minutes`.

## Decision
The canonical baseline disutility is LINEAR in transit travel-time minutes:
`U includes  - transport_cost * t_ij`, with `transport_cost` in EUR/min and
`t_ij` the cell->store transit minutes.

## Rationale
- Under multinomial logit, choice probabilities are smooth and a Bertrand-Nash
  equilibrium exists for any monotone transport disutility; the quadratic
  requirement from the deterministic Hotelling/d'Aspremont setting does not bind.
- Transit travel time is the economically relevant access cost in inner-Ringbahn
  Berlin and uses the richer GTFS/InfraGo data (ADR-016) rather than crow-flies km.
- Linearity keeps `transport_cost` interpretable as EUR per minute of travel.

## Consequences
- `City.dist2_km2` is a misnomer (holds minutes, applied linearly). A field rename
  is deferred to a separate change to avoid cross-module churn.
- Benchmarks (Bertrand-Nash, joint-monopoly) are computed against this same linear
  form, so Delta is internally consistent.
- A `transport_exponent` parameter (default 1.0) is available for sensitivity runs;
  set 2.0 to test a quadratic-in-minutes variant. This is NOT quadratic-in-km.

## Calibration anchor
`transport_cost = 0.5 EUR/min` => a 10-min trip costs 5 EUR of disutility,
of the same order as the logit scale mu=0.25 and prices ~0.3-0.8.
