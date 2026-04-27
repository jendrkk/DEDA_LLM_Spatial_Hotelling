# ADR 012 — Geographic Scope Changed from Pankow to Inner-Ringbahn Berlin

**Status:** Accepted  
**Date:** April 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The original project plan (`Agents_in_the_City__LLM-Driven_Spatial_Market_Entry_and_Algorithmic_Pricing_in_Berlin.md`) specified **Pankow** as the geographic scope for all OSM, Zensus, and LOR data pipelines. The Hydra config was named `berlin_pankow.yaml`. Pankow was chosen for tractability: it is a single district with a manageable number of grocery stores and a compact geographic extent.

However, the economic model requires meaningful **competition between chain types** (Discount vs. Standard vs. Bio) and **consumer heterogeneity across LOR social-status levels**. Pankow is one of Berlin's highest-status districts, with limited Discount chain presence and relatively homogeneous socioeconomic composition. This creates two modelling problems:

1. The vertical differentiation between chain types ($\alpha_H > \alpha_L$, high-WTP vs. low-WTP consumers) produces little variation in demand if all LOR social-status indices are concentrated near the high end.
2. The entrant's location choice problem is less interesting in a relatively homogeneous district — there are fewer strategically distinct sites.

---

## Decision

The geographic scope is the **inner-Ringbahn Berlin** (the area enclosed by the S41/S42 S-Bahn ring), not Pankow. All data pipelines, config files, and geographic references are updated to reflect this scope.

---

## Rationale

**Consumer heterogeneity.** The inner Ringbahn contains Berlin's most socioeconomically diverse neighbourhoods: Mitte, Prenzlauer Berg, Friedrichshain (high-status), Neukölln, Wedding, Moabit (lower-status), and the transition zones between them. The full range of LOR social-status indices is represented, making the type-mix mapping `π_Hi = S_{r(i)}` economically meaningful. In Pankow, the majority of LOR indices are concentrated above 0.6, compressing the consumer heterogeneity that motivates the vertical differentiation model.

**Chain type diversity.** The inner Ring contains all three chain types ($D$, $S$, $B$) in significant numbers: Aldi Nord, Lidl, Netto, and Penny are well-represented in lower-status areas; Edeka, Rewe, and Kaufland in mixed areas; Bio Company and Alnatura in higher-status areas. Pankow has few Discount chain locations, making the cost-asymmetry hypothesis (H5 in the project plan) hard to test.

**Competitive density.** The inner Ring contains approximately 100–150 grocery stores across the target chains — sufficient to populate the store RL agent pool meaningfully and to generate spatial variation in competitive density (the `DIVISION_COMPETITION` group variable). Pankow has ~30–50 stores, producing limited spatial variation.

**The entrant's location choice is richer.** The inner Ring contains clear spatial gaps between high-status and low-status areas, and between Discount-dense and Bio-dense zones. An LLM entrant reasoning about where to locate has genuinely distinct strategic alternatives. In Pankow, the strategic landscape is more uniform.

---

## Consequences

- `configs/env/berlin_pankow.yaml` is renamed to `configs/env/berlin_inner_ring.yaml`. All references to `berlin_pankow` in configs, CLI commands, and notebook titles are updated.
- The OSM data fetch scope changes: `district=pankow` → bounding box of the S41/S42 ring (approximately `52.46°N–52.55°N, 13.32°E–13.49°E`).
- Expected demand grid size increases from ~3,000 cells (Pankow) to ~8,000–9,000 cells (inner Ring) at 100 m resolution.
- Expected store count increases from ~30–50 to ~100–150. This increases the Q-table pool size proportionally and slightly increases the distance matrix (from ~450,000 to ~1,350,000 float32 entries — still ~5.4 MB).
- The distance matrix precompute step takes longer (roughly 3× the Pankow computation), but is still a one-time cost cached to Parquet.
- CLI command `hotelling data fetch-osm` must accept a bounding box parameter rather than a district name.
- Notebook `01_data_overview.ipynb` must be updated with the inner Ring bounding box and a new choropleth.
- The `berlin_inner_ring.yaml` config must include the S-Bahn ring polygon as a spatial filter for both demand cells and candidate entrant sites.
- LOR Planungsräume count increases from ~30 (Pankow) to ~150 (inner Ring), which is still tractable for the compressed zone summary in CEO prompts (8–10 aggregated zones).

---

## Data Pipeline Impact

| Dataset | Pankow scope | Inner Ring scope | Change |
|---|---|---|---|
| Zensus 2022 100m cells | ~3,000 cells | ~8,000–9,000 cells | 3× more cells |
| OSM grocery POIs | ~30–50 stores | ~100–150 stores | 3–4× more stores |
| LOR Planungsräume | ~30 | ~150 | 5× more LORs |
| Distance matrix | ~150,000 entries | ~1,350,000 entries | 9× larger, ~5.4 MB |
| Commercial candidate sites | ~200 | ~1,000–1,500 | ~6× more sites |

All changes are increases in data volume but not in pipeline complexity. The same loaders, the same distance computation, and the same LOR mapping logic apply — only the spatial filter changes.

---

## Alternatives Rejected

**Retain Pankow scope.** Computationally cheaper and faster for development iteration. Rejected because the consumer heterogeneity and chain type diversity that motivate the model's vertical differentiation and group division features are not present in sufficient degree. A model calibrated to Pankow would produce artificially homogeneous results, weakening the research contribution.

**Expand to all of Berlin.** Would include the outer districts (Spandau, Reinickendorf, Marzahn-Hellersdorf, etc.), increasing the cell count to ~50,000+ and the store count to ~400+. Computationally manageable but adds outer-district geography that is less relevant to the inner-city grocery competition dynamics at the core of the research question. The S-Bahn ring is a natural and defensible boundary.

**Use a synthetic city grid.** Replace real Berlin data with a synthetic 2D grid for tractability. Completely abandons the calibrated spatial IO contribution — the primary distinguishing feature of the project relative to prior work (Calvano, Porto). Rejected.