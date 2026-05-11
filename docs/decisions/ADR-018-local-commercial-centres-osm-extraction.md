# ADR 018 — Local Commercial Centres (LCC) via OSM Extraction

**Status:** Accepted  
**Date:** May 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The φ_i index (prime-location footfall weight in the demand model) is composed of three distinct mechanisms:

1. **Transit hubs** — stations with high passenger throughput, generating incidental grocery demand from commuters.
2. **Prime locations** — employment-dense clusters generating worker-driven daytime demand (lunch, after-work top-up). Previously labelled "CBD" in early project documents.
3. **High-amenity commercial areas** — concentrations of non-grocery retail and services that attract shoppers and pedestrians for non-grocery purposes, creating incidental grocery demand spillover.

Component (2) has been resolved: GEO_02 uses the AABPL (Automatic Algorithm for the Boundary-identification of Prime Locations) algorithm applied to IHK deduplicated employment density to identify employment clusters. These clusters are the operative definition of "prime locations" going forward. The output is saved to `data/processed/prime_location_clusters.parquet`.

Component (3) was previously called "prime locations" in the φ_i construction notes and was loosely defined as "shops and high amenities". This definition was imprecise and conflated with component (2). It is renamed **Local Commercial Centres (LCC)** and operationalised via OSM data extraction.

The change also resolves a terminology conflict: the label "CBD" (Central Business District) was used in early documents to denote employment-dense areas. "CBD" is geographically imprecise in Berlin's polycentric urban structure — there is no single CBD. The employment-cluster concept is better named "prime locations", consistent with the AABPL algorithm's own terminology. "Local commercial centres" then cleanly names the retail-amenity concept, following Berlin's Zentrenkonzept vocabulary (Hauptzentrum, Stadtteilzentrum, Ortsteilzentrum).

---

## Decision

1. **Retire the label "CBD" entirely.** Employment-dense clusters = "prime locations" (φ_i^prime), identified by AABPL on IHK data (done in GEO_02).
2. **Rename the shop/amenity component from "prime locations" to "Local Commercial Centres" (LCC)** (φ_i^lcl), extracted from OSM.
3. **LCC are extracted from OSM** via `fetch_pois()` with the tag sets defined below, in GEO_03.
4. **Transit hubs** logic and label are unchanged (φ_i^hub).

The composite φ_i becomes:
$$\phi_i = w_h\,\phi_i^{\text{hub}} + w_p\,\phi_i^{\text{prime}} + w_l\,\phi_i^{\text{lcl}}$$

Default weights: $w_h = 0.4$, $w_p = 0.3$, $w_l = 0.3$.

---

## Rationale

**Why LCC from OSM, not from administrative data (StEP Zentren)?**

StEP Zentren 2030/2040 (FIS-Broker WFS) defines Berlin's official retail centre hierarchy. It remains useful as a validation layer and provides a binary Hauptzentrum membership signal. However:

- StEP Zentren reflects planning classifications, not observed retail density. Areas between centres may have dense retail strips not captured by the hierarchy.
- OSM retail data is continuously updated and captures actual present-day commercial presence.
- OSM polygon area features (`landuse=retail`, `shop=mall`) provide footprint geometry suitable for constructing a continuous density surface, not just point signals.

The LCC index uses OSM as the primary source; StEP Zentren is retained as a validation cross-check.

**Why not use pedestrian counters (Hystreet) for LCC?**

Hystreet covers only 9 anchor locations and requires academic access registration (not yet obtained). OSM provides full coverage of inner-Ring commercial areas immediately. Hystreet data, when available, will feed the φ_i^prime component (prime locations + pedestrian flow signal), not LCC.

**Why "local commercial centres" and not "retail centres" or "shopping areas"?**

"Local commercial centres" is the direct English translation of Berlin's Zentrenkonzept term "Nahversorgungszentrum / Ortsteilzentrum" — the functional neighbourhood-level retail cluster. This ensures conceptual alignment with the urban planning literature and Berlin's own spatial planning documentation.

---

## OSM Tag Strategy for LCC Extraction

LCC identification uses three distinct OSM feature types, extracted in separate `fetch_pois()` calls and merged:

### Call 1: Malls and department stores — definitive LCC anchors
```python
fetch_pois(
    tags={"shop": ["mall", "department_store"]},
    name="lcl_anchors",
)
```
These are unambiguous large-format retail. Every `shop=mall` and `shop=department_store` polygon is a local commercial centre by definition.

### Call 2: Large-category non-grocery retail stores
```python
fetch_pois(
    tags={"shop": ["chemist", "electronics", "furniture", "hardware",
                   "clothes", "sports", "toy", "shoes", "books",
                   "stationery", "optician", "jewellery"]},
    name="lcl_retail",
)
```
Captures the major non-grocery chain categories: dm/Rossmann (`chemist`), MediaMarkt/Saturn (`electronics`), IKEA (`furniture`), Baumarkt (`hardware`), fashion chains (`clothes`), etc. These are the co-tenants that define a functioning commercial centre.

### Call 3: Retail landuse zones
```python
fetch_pois(
    tags={"landuse": "retail"},
    name="lcl_landuse",
)
```
Area-level features marking commercial strips and Geschäftsstraßen. Provides polygon geometry rather than point geometry — centroids of `landuse=retail` polygons define LCC area centres.

### Call 4: Retail buildings
```python
fetch_pois(
    tags={"building": "retail"},
    name="lcl_building",
)
```
Building-level retail classification. Overlaps substantially with Call 3 but captures standalone retail buildings not covered by landuse tagging.

**Merge and dedup:**
```python
gdf_lcl = (
    pd.concat([gdf_anchors, gdf_retail, gdf_landuse, gdf_building], ignore_index=True)
    .drop_duplicates(subset="osm_id")
    .pipe(lambda df: gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326"))
)
```

**Explicit exclusions** (not handled in query — filtered post-hoc if needed):
- `shop=supermarket`, `shop=convenience`, `shop=greengrocer`, `shop=bakery`, `shop=butcher`, `shop=deli`, `shop=alcohol`, `shop=beverages` — all grocery/food-primary. These are the simulation's target stores; they must not appear in the LCC index, which measures the commercial environment surrounding grocery demand, not the stores themselves.

---

## φ_i^lcl Construction from LCC Features

1. Spatial join: assign each inner-Ring 100m cell to LCC features within a 300m radius.
2. Weighted count: $\phi_i^{\text{lcl,raw}} = \sum_{k \in \text{LCC within 300m}} w_k \cdot \exp(-d_{ik} / r_0)$ where $w_k = \log(1 + \text{area}_k / 500\text{m}^2)$ (area bonus for polygon features; $w_k = 1$ for point features) and $r_0 = 200\text{m}$ (decay distance).
3. Normalise to $[0,1]$ across inner-Ring cells → $\phi_i^{\text{lcl}}$.

---

## Consequences

- The term "CBD" is removed from all project documentation, code comments, config files, and vault notes. Grep for "CBD" and replace with "prime location" (employment cluster) or "local commercial centre" (retail amenity) as context requires.
- `berlin-special-locations-phi-index-construction.md` (vault) is updated to reflect the three-component φ_i with the new terminology.
- `data-pipeline-osm-zensus-lor-brw.md` (vault) Step 5 is updated to distinguish the three φ_i components.
- GEO_03 notebook covers: (a) LCC OSM extraction via `fetch_pois()` multi-call pattern, (b) φ_i^lcl construction, (c) gridded φ_i assembly combining hub + prime + lcl components.
- `fetch_pois()` is extended (see Cursor prompt in project notes) to accept `List[Dict]` as `tags`, enabling a single-query multi-block Overpass call. Until that enhancement lands, use the multi-call + concat pattern above.
- The `CHAIN_QID_MAP` in `osm.py` is grocery-chain specific and requires no changes.

---

## Alternatives Rejected

**Use StEP Zentren as the primary LCC source.** Administrative classification, not observed retail density. Useful as validation; rejected as primary source because it misses informal commercial strips and does not update with real-world changes.

**Use OSM amenity density (cafes, restaurants, bars) as LCC proxy.** Amenity density captures a different signal: social/cultural attractiveness, not commercial retail density. Cafes and restaurants attract leisure dwell time but do not generate the same shopping spillover as retail chains. Retained as a minor weight in φ_i^lcl construction but not the dominant input.

**Merge all OSM commercial features into a single `fetch_pois()` call.** Requires the `List[Dict]` tags enhancement to `fetch_pois()`. Valid once the enhancement lands; the current multi-call pattern is equivalent and is the interim approach.

---
