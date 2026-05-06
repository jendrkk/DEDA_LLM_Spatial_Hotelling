# ADR 017 — ALKIS Building Floor-Space Efficiency Factors and Employee Hard Cap

**Status:** Accepted  
**Date:** May 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The demand-side layer of the simulation requires a spatially explicit measure of commercial employment density per 100 m grid cell.  The raw data source is the IHK Berlin Gewerbedaten (see [[ADR-013-ihk-deduplication-ordinal-signal]]).  After the deduplication pipeline established in ADR-013, two structural problems remain:

**Problem 1 — Headquarters reporting.**  IHK registrations attribute a firm's full legal-entity headcount to a single registered address, typically the headquarters building.  A building with 10 m² of office space might nominally carry 500 employees because the legal entity operating there is a shell holding for a larger group.  The reported headcount is dimensionally inconsistent with the building's physical capacity.

**Problem 2 — Multi-registration inflation.**  Even after deduplication at the same coordinate, multiple registrations can share a building address (different floors, different tenants) and individually report plausible headcounts that in aggregate exceed what the building can physically hold.

Both problems distort the employment density surface used to calibrate consumer demand weights and the WTP proxy.  A **physical hard cap** derived from building geometry and function corrects for both: the cap is geometry-grounded and function-specific, independent of how many IHK entries are attached to the building.

The cap requires two components:
1. A **floor-space efficiency factor (EF)** mapping ALKIS Gebäudefunktion (GFK) code → NUF/BGF ratio, converting gross floor area (footprint × storeys) to net usable area.
2. A **m²/employee norm** mapping GFK code → typical net usable area per employee position, converting net usable area to a maximum headcount.

---

## Decision

Implement a GFK-indexed lookup table for both EF and m²/employee in `src/hotelling/spatial/gebaeude_capacity.py`.  Compute the hard cap per building as:

```
H = (footprint_m2 × num_floors × EF(gfk, hochhaus)) / m2_per_employee(gfk)
```

Enforce the cap against IHK headcounts using:
- **Single-company case:** `X_capped = min(X, H)`
- **Multi-company case:** proportional scaling — `X_i_capped = X_i × min(1, H / Σ Xⱼ)`

The proportional rule preserves relative firm sizes while enforcing the physical aggregate constraint.  It is applied per building, grouping all IHK entries joined to the same `gebaeude.gpkg` polygon.

---

## Rationale

### Why a hard cap rather than a soft prior?

A hard cap is the right choice here because the constraint is **physical**, not statistical.  A building with a 400 m² footprint, 3 floors, and 0.75 EF has exactly 900 m² of net usable area.  No matter what any IHK record says, 900 m² cannot physically hold 2,000 office workers.  A soft prior (e.g. a Bayesian shrinkage toward expected density) would require calibration data we don't have; the hard cap requires only building geometry and two lookup tables.

### Why proportional scaling for the multi-company case?

Proportional scaling `X_i_capped = X_i × min(1, H / Σ Xⱼ)` has three desirable properties:

1. **Preserves relative size:** if firm A reports twice as many employees as firm B, it retains twice as many after capping.  The ordinal signal from ADR-013 is maintained.
2. **Aggregate constraint is exactly binding:** Σ X_i_capped = min(Σ Xⱼ, H) exactly.
3. **Degenerates cleanly to the single-firm case:** when N = 1, the rule becomes min(X, H).

The alternative — capping each firm independently at H/N — would ignore the size distribution within the building and could leave the aggregate well below H for buildings with one large and several tiny tenants.

### EF calibration

EF values are calibrated primarily against **DIN 277-1:2016** (Grundflächen und Rauminhalte im Bauwesen), which defines NUF (Nutzungsfläche), NGF (Nettogrundfläche), and BGF (Bruttogrundfläche) and provides empirical NUF/BGF ranges by building function.  For commercial types (GFK 2010–2060), the **gif MF-G 2017** (Mietflächenrichtlinie Gewerbeflächen, Gesellschaft für Immobilienwirtschaftliche Forschung) provides a calibration cross-check specifically for German leasable office and retail space.  The **RICS Code of Measuring Practice (6th ed.)** serves as a further validation reference for offices and high-rises.

Key design choices:

- **GFK 2010 (Handel + Dienstleistungen)** is the most important code in Berlin: it is the ALKIS catch-all for most commercial office buildings.  EF = 0.75 at ground level; drops to 0.68 for `hochhaus = True`, reflecting the larger structural and service core that scales with height.
- **GFK 3051 (Krankenhaus)** receives the lowest EF (0.55): hospital floor plates are consumed by double-loaded ward corridors, medical gas risers, clean-room zones, and isolation bays.
- **GFK 2143 (Lagerhalle)** receives the highest EF (0.91): a warehouse is essentially a slab with a roof; the wall and ancillary share is minimal.
- **Hochhaus penalty (0.07):** high-rises sacrifice 6–8 percentage points of floor plate to structural cores, additional elevator banks, mechanical plant floors, and fire-escape shafts.  The penalty is exempt for building types where the concept is physically meaningless (kiosks, sheds, churches, utility buildings).  Minimum EF after penalty = 0.40.

### m²/employee calibration

m²/employee norms are sourced from authoritative German construction and industry benchmarks:

- **ArbStättV (Arbeitsstättenverordnung, Anhang 1.2):** statutory minimum of 8 m² net area per person in German workplaces.  Used as a binding lower floor and as the primary source for building types not covered by sector-specific studies.
- **gif Büroflächenreport 2022/2023 (Berlin):** Berlin office market average of approximately 13 m² net internal area per workstation.  Used for GFK 2010, 2020, 3010, 3012, and related administrative office types.  12 m² adopted as a tight-packing cap (maximum headcount scenario).
- **RICS Global Occupancy Costs Survey 2023:** European office average of ~10 m² NIA per workstation, validating the lower range of the gif figure.
- **Statistisches Bundesamt, Strukturerhebung Einzelhandel:** employee density in German retail floor space, used for GFK 2050–2055 (retail types).
- **IHA Hotelmarkt Deutschland 2023:** hotel staffing ratios (FTE per room), used for GFK 2071 (Hotel/Motel/Pension).
- **Deutsche Krankenhausgesellschaft (DKG):** hospital staffing and area norms, used for GFK 3051–3052.
- **BVL Logistikimmobilien report (Bundesvereinigung Logistik):** warehouse and logistics employee density, used for GFK 2140–2150.
- **DEHOGA Branchenbericht Gastronomie:** restaurant kitchen and front-of-house staffing, used for GFK 2081, 2083.
- **KMK Lehrerarbeitsstätten guidelines:** school staffing norms, used for GFK 3021–3023.

Building types with no meaningful commercial employee capacity (residential, unmanned infrastructure, parking) are assigned `m²_per_employee = 999`, making the hard cap effectively infinite and ensuring no cap is enforced against incidental IHK registrations at those addresses.

### Relationship to ADR-013

This ADR operates **downstream** of ADR-013.  The deduplication pipeline from ADR-013 must be applied first; the hard cap is then applied to the deduplicated dataset.  The two mechanisms address different pathologies:

- ADR-013 removes **inter-registration duplicates** (same physical employer registered multiple times at the same coordinate).
- ADR-017 corrects **intra-building over-reporting** (one or more registrations whose aggregate headcount exceeds the building's physical capacity).

---

## Implementation

**Module:** `src/hotelling/spatial/gebaeude_capacity.py`

**Public API:**

| Function | Purpose |
|---|---|
| `get_efficiency_factor(gfk, hochhaus)` | NUF/BGF ratio |
| `get_m2_per_employee(gfk)` | m² net area per employee |
| `compute_employee_hard_cap(footprint, floors, gfk, hochhaus)` | Hard cap H |
| `apply_hard_cap_single(reported, H)` | Enforce cap, single company |
| `apply_hard_cap_multi(series, H)` | Enforce cap, N companies (proportional) |
| `enrich_gebaeude(gdf)` | Add EF, usable area, cap columns to building GDF |

**Pipeline integration:** the hard cap is applied in `notebooks/GEO_02_city_data.ipynb` after the IHK deduplication step (ADR-013) and before aggregating employment to grid cells.

**Data dependency:** requires `data/raw/gebaeude.gpkg` (ALKIS Stadtstruktur download via `download_stadtstruktur()` in `city_data.py`) to supply `gfk`, `hochhaus`, `anzahl_der_oberirdischen_geschosse`, and polygon geometry.

**CRS note:** footprint area is computed from geometry in EPSG:3035 (metres), consistent with the rest of the pipeline.

---

## Consequences

- IHK employment totals per grid cell will be materially lower in cells dominated by large-employer headquarters entries and in cells with single tall office buildings.  This is the intended effect: the cap removes physically implausible density spikes.
- The LOR-level ordinal ranking of commercial activity (the primary use of IHK data per ADR-013) is preserved because the cap is applied proportionally and the scale of the correction is largest for the most implausible outliers.
- Grid cells without a matched building polygon (e.g. IHK registrations whose coordinates do not fall inside any `gebaeude.gpkg` polygon) receive no cap.  This is acceptable: the deduplication in ADR-013 already removes the worst outliers, and unmatched points are a small fraction of registrations.
- The thesis methods section states: "IHK-reported headcounts are subject to a physical capacity cap derived from ALKIS building footprints, floor counts, and GFK-specific efficiency factors (DIN 277-1:2016; gif MF-G 2017).  For buildings with multiple IHK registrations, aggregate headcounts are scaled proportionally to the building's physical capacity."

---

## Alternatives Rejected

**No cap (raw IHK after deduplication only):** Rejected.  Deduplication alone does not remove headquarters-attribution inflation.  The Bahntower entry (7,500–9,999 employees attributed to one building) survives deduplication because it is a single registration — it is simply wrong relative to the building's actual capacity.

**Uniform cap by LOR:** Rejected.  A single LOR-level density threshold ignores the heterogeneity of building functions and sizes within each LOR.  A Hochhaus office block and a Gartenhaus in the same LOR should not share the same cap.

**Bayesian shrinkage toward expected density:** Rejected.  Would require a calibrated prior for each GFK type that we do not have independent data to estimate.  The hard cap from DIN 277 / gif norms is empirically grounded and does not require estimation.
