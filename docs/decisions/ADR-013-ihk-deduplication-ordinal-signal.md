# ADR 013 — IHK Gewerbedaten Used as Ordinal Signal Only; Deduplication Required Before Any Use

**Status:** Accepted  
**Date:** May 2026  
**Deciders:** Jedrzej Slowinski  

---

## Context

The IHK Gewerbedaten (`data/raw/2023_12_IHK_Berlin_Gewerbedaten.csv`) is the only publicly available fine-grained commercial activity dataset for inner-Ring Berlin. It was considered as a data source for two purposes: (a) constructing the CBD component of the prime-location index $\phi_i^{\text{CBD}}$, and (b) computing daytime employment density as a demand-side complement to Zensus 2022 residential population.

During exploratory analysis in `notebooks/GEO_02_city_data.ipynb`, a severe data quality issue was discovered: a single 100m × 100m cell containing a Flink depot showed 12,000+ employees, despite containing a single physical employer. Investigation revealed that Flink is registered under multiple legal entities (operating GmbH, holding, logistics subsidiary) in the IHK, each with the same coordinate and each carrying the full firm employment class (`1,000–2,500 Beschäftigte`). The same pattern was observed for DB AG (Bahntower entry: 7,500–9,999 employees — the firm's total Berlin workforce, not the people in that building), and several other large employers.

The root cause is structural: **IHK membership is based on the legal entity, not the physical workplace.** The `Beschäftigtengrößenklasse` field records total headcount of the legal entity, attributed to the entity's registered address. This means:

1. **Headquarters effect:** All employees of a multi-location firm may be counted at the HQ address; branch offices may report 0.
2. **Multi-registration inflation:** The same address appears multiple times for holding structures, generating systematic summing errors at grid-cell resolution.

The IHK data is designed for LOR- or Bezirk-level aggregation; it is not appropriate for 100m grid-cell analysis without deduplication.

---

## Decision

The IHK Gewerbedaten is used **as an ordinal signal of commercial activity at the LOR level only**, and only after applying the following deduplication pipeline:

1. **Midpoint conversion:** Map `Beschäftigtengrößenklasse` categories to numeric midpoints (e.g., `1,000–2,500` → 1,750).

2. **Same-coordinate large-employer collapse:** For any (lat, lon) pair where the maximum midpoint across all registrations is ≥ 500 employees, keep only the single entry with the highest employee count. Rationale: ≥ 500 at the same exact coordinate is almost certainly a multi-registration of one physical employer; retaining the max avoids both summing duplicates and discarding the signal entirely.

3. **Retain all small-employer entries:** For coordinates where the maximum midpoint is < 500, retain all entries. Multiple distinct small businesses at the same address are plausible and should be summed.

4. **Physical plausibility ceiling:** After deduplication, flag any 100m cell with total deduplicated employment > 1,000 for manual inspection. The theoretical maximum for a 100m × 100m cell at dense office usage (15 m²/person, 60% commercial footprint) is approximately 400 workers in a typical block; 1,000 is an extreme-case ceiling for CBD tower scenarios.

The deduplicated output is then aggregated to the Planungsraum (LOR) level. At LOR level, the remaining noise from the headquarters effect (branch-office zero-reporting) averages out sufficiently to produce a usable ordinal signal.

The IHK data is **not used** as the primary or sole source for any model parameter. It is one input among several for the $\phi_i^{\text{CBD}}$ component of the prime-location index (alongside Bodenrichtwerte). It is not used as the `ω_i` residential population input — that remains the Zensus 2022 100m raster.

---

## Rationale

**Correctness.** Using the raw IHK sums at cell level would produce employment densities that are 6–10× the true value for cells containing large employers. This would cause the $\phi_i^{\text{CBD}}$ component to spike in a small number of cells around major HQ addresses (Flink depot, DB Bahntower, Amazon logistics) rather than reflecting the actual distribution of commercial activity. The downstream effect would be implausible demand weights in those cells.

**Defensibility.** The deduplication rule is conservative: it makes no assumption about how many physical employees a firm actually has at a location — it simply prevents the same firm from being counted multiple times at the same point. The max-retention strategy (keep the largest, not the sum) is biased downward relative to truth, which is safer for a demand model (understating footfall is less harmful than overstating it).

**Ordinal-only use.** Given the headquarters effect cannot be fully corrected without auxiliary data (Bundesagentur für Arbeit small-area employment, available on request), the IHK signal is treated as an ordinal rank of commercial intensity at LOR level, not as a cardinally accurate employment count. This is explicitly documented in the model methods section.

---

## Consequences

- A new preprocessing step is added to the data pipeline: `notebooks/GEO_02_city_data.ipynb` implements `deduplicate_ihk()` and exports `data/processed/ihk_deduplicated.parquet`.
- `src/hotelling/spatial/ihk.py` implements the deduplication function for production use.
- `src/hotelling/spatial/phi_index.py` uses `ihk_deduplicated.parquet` as one of two inputs for the CBD component (the other being BRW from `data/raw/brw_2023_vector.gpkg`).
- The IHK deduplication step is documented as a known limitation in the thesis methods section: "IHK employment data has been deduplicated to remove multi-registration artefacts but may still understate employment at branch-office locations due to the headquarters attribution convention."
- The Bundesagentur für Arbeit small-area employment data (formal data request, 4–6 week lead time) is noted as the preferred long-run replacement for the IHK CBD signal.

---

## Alternatives Rejected

**Use raw IHK sums at cell level.** Produces extreme outlier cells (12,000+ employees) from multi-registration artefacts. Rejected immediately on data quality grounds.

**Drop IHK entirely, use BRW only for φ_i^CBD.** BRW captures land values, which are a good proxy for commercial rent but not directly for employment density. High BRW residential areas (e.g., Charlottenburg villa zones) would be wrongly flagged as CBD-like. IHK, even after deduplication, adds a genuine employment-density signal that BRW cannot provide.

**Use Zensus 2022 Erwerbspersonen grid as CBD proxy.** Measures residential employment status (people who are employed), not workplace employment density. A cell in Prenzlauer Berg may have many employed residents who commute to Mitte — this is not a CBD signal for the Prenzlauer Berg cell.

**File a BA data request and wait.** The Bundesagentur für Arbeit small-area employment data would provide social-insurance-based workplace counts by PLZ, substantially better than IHK. It cannot be obtained immediately; IHK is the only available dataset for the current project timeline. The BA request should be filed now as a long-run improvement.
