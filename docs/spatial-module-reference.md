# `hotelling.spatial` — Module Reference

> **Status:** authoritative reference — updated 2026-05-06  
> **Package:** `hotelling[spatial]`  
> **Install:** `pip install hotelling[spatial]`

---

## Overview

`hotelling.spatial` provides all geographic building blocks needed to run
LLM-driven spatial competition simulations on real city data.  The module
handles the full data pipeline:

1. **Boundary data** (`boundaries.py`) — download and load administrative
   polygon boundaries from Overpass.
2. **Sub-city units** (`admin.py`) — Berlin LOR shapes (Lebensweltlich
   Orientierte Räume) and similar planning-area geometries.
3. **Population data** (`census.py`) — Zensus 2022 100 m grid download,
   loading, filtering, and full INSPIRE grid construction.
4. **POI data** (`osm.py`) — fetch points-of-interest (nodes, ways, relations)
   from OpenStreetMap via Overpass, with full tag preservation.
5. **Grid** (`grid.py`) — regular square cell lattice for consumer sampling
   and discrete spatial games.
6. **Distances** (`distance.py`) — Euclidean and network (OSRM) distance
   matrix computation.
7. **Pipeline orchestrator** (`exe.py`) — single entry point that chains all
   seven phases from raw download to simulation-ready grid assembly.
8. **Grid assembly** (`assembly.py`) — merges population, LOR, POI, LCC mall
   layers (`add_lcc_layer`), and socio-economic layers into the final simulation
   grid.
9. **Building capacity** (`gebaeude_capacity.py`) — ALKIS GFK → floor-space
   efficiency factors (NUF/BGF) and employee hard caps for ALKIS `Gebäude`
   footprints; enforces physical capacity constraints against IHK headcounts.
10. **Chain classification** (`osm.py`) — `CHAIN_TYPE_MAP` and
    `process_supermarkets` for normalising and classifying supermarket POIs
    into three vertical-differentiation tiers (discount / standard / bio).

All modules are imported lazily from `hotelling.spatial` so that only the
`[spatial]` optional dependencies are required, and only when the relevant
function is actually called.

---

## Quick-start

```python
from hotelling.spatial import (
    download_city_boundary,
    load_boundary,
    fetch_pois,
    download_zensus_2022,
    load_zensus_2022,
    filter_zensus_2022,
    build_full_grid,
    download_lor_shapes,
)
from pathlib import Path
from hotelling.spatial import run_default_data_pipeline

# 1. Administrative boundary
download_city_boundary("Berlin")                 # saves data/raw/city_boundary_Berlin.geojson
boundary = load_boundary(Path("data/raw/city_boundary_Berlin.geojson"))  # EPSG:3035

# 2. Population grid
download_zensus_2022()                           # saves data/raw/zensus2022_grid.parquet
filter_zensus_2022(Path("data/raw/city_boundary_Berlin.geojson"))
zensus = load_zensus_2022()                      # EPSG:3035 GeoDataFrame

# 3. Full demand grid (100 m INSPIRE cells with population weights)
grid_gdf = build_full_grid(boundary, zensus)

# 4. LOR shapes (Berlin)
download_lor_shapes(if_old=False)                # saves data/raw/lor_shapes_2021.parquet

# 5. OSM points-of-interest (supermarkets, convenience stores)
pois = fetch_pois("Berlin")                      # EPSG:4326 GeoDataFrame
pois_3035 = pois.to_crs("EPSG:3035")            # reproject for spatial joins

# 6. Run full pipeline
run_default_data_pipeline()
```

---

---

## Module: `gebaeude_capacity.py`

> **Decision:** [[ADR-017-gebaeude-floor-space-efficiency-employee-cap]]

Derives two quantities from the ALKIS `Gebäudefunktion` (GFK) code and the
`hochhaus` flag for every building polygon in `gebaeude.gpkg`:

1. **Efficiency factor (EF = NUF/BGF)** — the share of gross floor area
   (footprint × floors) that constitutes net usable space after deducting
   walls, staircases, elevator shafts, technical rooms, and corridors.
2. **Employee hard cap H** — the physical maximum number of employees the
   building can accommodate: `H = (footprint × floors × EF) / m²_per_employee`.

### `get_efficiency_factor`

```
hotelling.spatial.gebaeude_capacity.get_efficiency_factor(
    gfk: int | None,
    hochhaus: bool = False,
) -> float
```

Returns the NUF/BGF ratio for the given GFK code.  If `hochhaus=True` and
the code is not in the exempt set, applies `HOCHHAUS_PENALTY = 0.07`.
Falls back to group-level defaults (rounded GFK ÷ 1000) then to
`EF_DEFAULT_UNKNOWN = 0.75` for unrecognised codes.

### `get_m2_per_employee`

```
hotelling.spatial.gebaeude_capacity.get_m2_per_employee(
    gfk: int | None,
) -> float
```

Returns net usable floor area (m²) per employee position for the GFK type.
Returns `999.0` for building types with no meaningful employee capacity
(residential, unmanned infrastructure, parking), making the hard cap
effectively infinite.

### `compute_employee_hard_cap`

```
hotelling.spatial.gebaeude_capacity.compute_employee_hard_cap(
    footprint_m2: float,
    num_floors: int | None,
    gfk: int | None,
    hochhaus: bool = False,
) -> float
```

Returns the physical hard cap H.  Returns `np.inf` for GFK types where
`m²_per_employee ≥ 999`.  `num_floors = None / 0 / NaN` is treated as 1.

### `apply_hard_cap_single`

```
hotelling.spatial.gebaeude_capacity.apply_hard_cap_single(
    reported: float,
    hard_cap: float,
) -> float
```

Returns `min(reported, hard_cap)`.  NaN is passed through unchanged.

### `apply_hard_cap_multi`

```
hotelling.spatial.gebaeude_capacity.apply_hard_cap_multi(
    reported: pd.Series,
    hard_cap: float,
) -> pd.Series
```

Enforces the hard cap across multiple IHK registrations sharing the same
building using **proportional scaling**:

```
X_i_capped = X_i × min(1.0,  H / Σ X_j)
```

Preserves relative firm sizes.  NaN entries are passed through.  Returns
`reported` unchanged when `hard_cap = np.inf`.

**Groupby pattern (GeoDataFrame workflow):**

```python
# 1. Spatial-join IHK points to building polygons
ihk_joined = gpd.sjoin(ihk_gdf, gebaeude[["gfk", "hochhaus", "floors", "geometry"]],
                       how="left", predicate="within")

# 2. Compute hard cap per building
ihk_joined["hard_cap"] = ihk_joined.apply(
    lambda r: compute_employee_hard_cap(
        gebaeude.loc[r["index_right"]].geometry.area,
        r["floors"], r["gfk"], bool(r["hochhaus"])
    ), axis=1
)

# 3. Apply multi-company cap
ihk_joined["empl_capped"] = (
    ihk_joined.groupby("index_right", group_keys=False)
    .apply(lambda g: apply_hard_cap_multi(g["empl"], g["hard_cap"].iloc[0]))
)
```

### `enrich_gebaeude`

```
hotelling.spatial.gebaeude_capacity.enrich_gebaeude(
    gdf: gpd.GeoDataFrame,
    gfk_col: str = "gfk",
    hochhaus_col: str = "hochhaus",
    floors_col: str = "anzahl_der_oberirdischen_geschosse",
) -> gpd.GeoDataFrame
```

Convenience wrapper.  Adds `efficiency`, `usable_area_m2`, and
`employee_hard_cap` columns to a building GeoDataFrame and returns the
enriched copy.

### Module constants

| Constant | Value | Description |
|---|---|---|
| `HOCHHAUS_PENALTY` | 0.07 | EF deduction for high-rise buildings |
| `EF_MIN` | 0.40 | Minimum EF after penalty |
| `GFK_BASE_EFFICIENCY` | dict | Full GFK → EF lookup |
| `GFK_M2_PER_EMPLOYEE` | dict | Full GFK → m²/employee lookup |
| `GFK_LABEL` | dict | Full GFK → German label |

---

## Module: `osm.py`

### `fetch_pois`

```
hotelling.spatial.fetch_pois(
    type: str = "supermarket",
    city: str = "Berlin",
    tags: dict | list[dict] | None = None,
    name: str | None = None,
    cache_dir: Path | None = None,
    timeout: int = 180,
) -> geopandas.GeoDataFrame
```

Fetch points-of-interest from OpenStreetMap for any named city.

> **Built-in type profiles:**
> - `"supermarket"` — fetches `shop=supermarket` elements; adds `chain` column.
>   Cache: `OSM_POIs_{city}_supermarket.parquet`
> - `"LCC"` — fetches Large Commercial Centre anchor stores (malls, department stores,
>   chemist chains, variety stores, electronics, DIY, furniture, sports chains).
>   Cache: `OSM_POIs_{city}_LCC.parquet`. No `chain` column.
> - `"stations"` — fetches `railway=station` elements (S-Bahn, U-Bahn, regional,
>   long-distance). Cache: `OSM_POIs_{city}_stations.parquet`. No `chain` column.

**Data flow:**

1. Nominatim (`nominatim.openstreetmap.org`) resolves the city name to an
   Overpass area ID.
2. Overpass (`overpass-api.de`) returns all matching nodes, ways, and
   relations with full geometry and all tags.
3. Geometries are parsed:
   - nodes → `Point(lon, lat)`
   - closed ways (first == last coord, ≥ 4 pts) → `Polygon`
   - open ways → discarded
   - relations → `Polygon` / `MultiPolygon` (polygonized from member ways,
     outer/inner ring handling)
4. All OSM tags across all elements are merged as GeoDataFrame columns.
   Rows missing a tag carry `NaN`.
5. A `point` column is added: equals `geometry` for nodes, centroid for
   polygon elements.
6. A `chain` column is derived from `brand:wikidata` via `CHAIN_QID_MAP`.
7. The result (without `point`) is cached as
   `{cache_dir}/OSM_POIs_{city}.parquet`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `str` | `"supermarket"` | Query profile: `"supermarket"`, `"LCC"`, `"stations"`, or any custom string |
| `city` | `str` | `"Berlin"` | Nominatim place name |
| `tags` | `dict \| list[dict] \| None` | see type | OSM tag filter(s). Ignored for built-in types. Pass a list of dicts for OR-union of multiple tag blocks. |
| `name` | `str \| None` | `None` | Override cache file stem. File: `OSM_POIs_{city}_{name}.parquet` |
| `cache_dir` | `Path \| None` | `Path("data/raw")` | Parquet cache directory |
| `timeout` | `int` | `180` | Overpass query timeout (s) |

**Tag filter format:**

```python
# exact match
{"shop": "supermarket"}

# OR match (Overpass regex)
{"shop": ["supermarket", "convenience", "grocery"]}

# key-exists check
{"healthcare": True}

# multiple keys (all must match)
{"shop": ["supermarket"], "opening_hours": True}
```

**Returns:** `geopandas.GeoDataFrame` with CRS EPSG:4326.

Guaranteed columns: `osm_id`, `osm_type`, `geometry`, `point`, `chain`.
Additional columns: every OSM tag key found in the data.

**Geometry column:**

| Column | Type | Description |
|---|---|---|
| `geometry` | `Point \| Polygon \| MultiPolygon` | Raw OSM geometry (active GeoPandas column) |
| `point` | `shapely.Point` (object dtype) | Representative point for distance calculations |

> **Note:** `point` is a plain object column, not a `GeoSeries`.  It is
> recomputed from `geometry` on every load and is not stored in the Parquet
> cache.  After reprojecting to EPSG:3035, recompute with:
> ```python
> pois_3035["point"] = pois_3035.geometry.apply(
>     lambda g: g if g.geom_type == "Point" else g.centroid
> )
> ```

**Raises:**

- `requests.HTTPError` — non-transient HTTP failure from Overpass or Nominatim
- `ValueError` — Nominatim returned no results for `city`
- `RuntimeError` — all 3 Overpass retry attempts returned transient errors

---

### `normalize_chain_name`

```
hotelling.spatial.normalize_chain_name(
    wikidata_qid: str | None,
    brand: str | None = None,
    *,
    name: str | None = None,
) -> str | None
```

Map OSM supermarket tags to a canonical chain name.  Resolution priority
(first match wins):

1. `brand:wikidata` QID → `CHAIN_QID_MAP`
2. `brand` field → `_BRAND_NAME_MAP` (case-insensitive)
3. `name` field → `_BRAND_NAME_MAP` (case-insensitive)
4. Raw `brand` as-is for unrecognised chains (independents keep their identity)

Returns `None` when all four sources are unavailable.

---

### `CHAIN_QID_MAP`

`dict[str, str]` — Wikidata QID → canonical chain name mapping for Berlin
grocery and drugstore chains.

| QID | Chain |
|---|---|
| Q151954 | Rewe |
| Q11462860 | Penny |
| Q700965 | Lidl |
| Q41187 | Aldi Süd |
| Q125054261 | Aldi Nord |
| Q685967 | Edeka |
| Q459358 | Netto |
| Q2662792 | Kaufland |
| Q1145968 | dm |
| Q183538 | Rossmann |

---

### `CHAIN_TYPE_MAP`

`dict[str, str]` — Canonical chain name → vertical-differentiation tier.

Three tiers used by the two-type demand model:

| Tier | Chains |
|---|---|
| `"discount"` | Aldi Nord, Lidl, Netto Marken-Discount, Netto, Penny, Norma |
| `"standard"` | Edeka, Rewe, Kaufland, HIT, CAP, Nah & Frisch, Mix Markt |
| `"bio"` | Denns BioMarkt, Bio Company, Alnatura, LPG BioMarkt |

### `process_supermarkets`

```
hotelling.spatial.process_supermarkets(
    pois_raw: geopandas.GeoDataFrame,
    grid: geopandas.GeoDataFrame,
) -> geopandas.GeoDataFrame
```

Clean, normalise, and clip raw supermarket POIs to the simulation grid.

Reprojects to grid CRS, replaces geometry with centroids, clips to grid
extent, adds `chain_type` from `CHAIN_TYPE_MAP`, and retains only
recognised chains. Returns columns: `geometry`, `name`, `chain`, `chain_type`.

Writes to `data/processed/supermarkets.parquet` when called from the pipeline.

---

## Module: `boundaries.py`

### `download_city_boundary`

```
hotelling.spatial.download_city_boundary(city_name: str) -> None
```

Download the administrative boundary of a German city from Overpass and save
as GeoJSON in **EPSG:3035** at `data/raw/city_boundary_{city_name}.geojson`.

> **Warning:** The GeoJSON file is stored in EPSG:3035 (metres), which
> violates RFC 7946.  Always use `load_boundary()` to load it — never
> `gpd.read_file()` directly.  See `docs/crs-handling-and-known-issues.md`,
> Issue 2.

---

### `download_relation_boundary`

```
hotelling.spatial.download_relation_boundary(relation_id: int) -> None
```

Download the boundary of an OSM relation by numeric ID from Overpass and save
as GeoJSON in EPSG:3035 at `data/raw/relation_boundary_{relation_id}.geojson`.

---

### `load_boundary`

```
hotelling.spatial.load_boundary(path: Path) -> geopandas.GeoDataFrame
```

Load a boundary GeoJSON file produced by `download_city_boundary` or
`download_relation_boundary`.  Reads the `properties.crs` field to set the
correct CRS (EPSG:3035 for files produced by this package).

---

## Module: `admin.py`

### `download_lor_shapes`

```
hotelling.spatial.download_lor_shapes() -> None
```

Download Berlin LOR (Lebensweltlich Orientierte Räume) planning-area shapefiles
from the Berlin Senate Department, extract, reproject to EPSG:3035, and save
as `data/raw/lor_shapes_2019.parquet` or `data/raw/lor_shapes_2021.parquet`.

Requires the optional `py7zr` dependency for `.7z` extraction.

### `equip_lor_with_population`

```
hotelling.spatial.admin.equip_lor_with_population(
    lor: geopandas.GeoDataFrame,
    population_grid: geopandas.GeoDataFrame,
) -> geopandas.GeoDataFrame
```

Spatial-join population grid centroids to LOR polygons and sum `Einwohner`
per LOR unit.  Both inputs must share the same CRS.

### `shapes_around_boundary`

```
hotelling.spatial.admin.shapes_around_boundary(
    shapes: geopandas.GeoDataFrame,
    boundary: geopandas.GeoSeries,
    buffer_distance: float = 1000.0,
) -> geopandas.GeoDataFrame
```

Return shapes that intersect the boundary or lie within `buffer_distance`
metres of it.

### `refine_shapes_selection`

```
hotelling.spatial.admin.refine_shapes_selection(
    shapes: geopandas.GeoDataFrame,
    boundary: geopandas.GeoSeries,
    population_grid: geopandas.GeoDataFrame,
    buffer_distance: float = 1000.0,
    extend_selection_by: int = 10,
) -> geopandas.GeoDataFrame
```

Extend the initial buffer-based shape selection by including the
`extend_selection_by` most densely populated shapes closest to the boundary.
Returns the full `shapes` GeoDataFrame augmented with selection and scoring
columns.

---

## Module: `census.py`

### `download_zensus_2022`

```
hotelling.spatial.download_zensus_2022() -> None
```

Download the Zensus 2022 100 m population grid from Destatis, extract,
convert to a GeoDataFrame with EPSG:3035 geometry, and save as
`data/raw/zensus2022_grid.parquet`.

### `load_zensus_2022`

```
hotelling.spatial.load_zensus_2022() -> geopandas.GeoDataFrame
```

Load the Zensus 2022 grid from `data/raw/zensus2022_grid.parquet` in
EPSG:3035.  Returns point geometries (midpoints of 100 m cells).

### `filter_zensus_2022`

```
hotelling.spatial.filter_zensus_2022(boundary_path: Path) -> None
```

Filter the Zensus grid to points within the boundary geometry and write
`data/raw/zensus2022_grid_filtered.parquet`.

### `build_full_grid`

```
hotelling.spatial.build_full_grid(
    boundary: geopandas.GeoDataFrame,
    zensus: geopandas.GeoDataFrame,
    cell_size: float = 100.0,
) -> geopandas.GeoDataFrame
```

Construct the complete INSPIRE 100 m grid inside `boundary`, then left-join
Zensus population counts.  Cells with no population record receive
`Einwohner = 0`.  Returns a GeoDataFrame in EPSG:3035 with columns
`x_mp_100m`, `y_mp_100m`, `geometry`, and `Einwohner`.

### `run_default_data_pipeline`

```
hotelling.spatial.run_default_data_pipeline() -> None
```

Convenience function that runs the complete Berlin data pipeline:
`download_zensus_2022` → `download_city_boundary("Berlin")` →
`download_relation_boundary(14983)` → `download_lor_shapes` →
`filter_zensus_2022`.

---

## Module: `grid.py`

### `SquareGrid`

```python
@dataclass
class SquareGrid:
    width: int = 50
    height: int = 50
    cell_size: float = 100.0           # metres
    population: np.ndarray | None = None  # shape (height, width)
    crs: str | None = None
```

Regular square cell lattice with optional population density weights.

| Method | Description |
|---|---|
| `total_population()` | Sum of all cell weights |
| `sample_locations(n, rng)` | Sample `n` cells proportional to weights *(not yet implemented)* |
| `cell_to_metres(row, col)` | Map cell index to (x, y) metres from origin *(not yet implemented)* |

---

## Module: `distance.py`

### `euclidean_distance_matrix`

```
hotelling.spatial.euclidean_distance_matrix(
    locations_a: np.ndarray,  # shape (M, 2) — (x, y) in metres
    locations_b: np.ndarray,  # shape (N, 2) — (x, y) in metres
) -> np.ndarray               # shape (M, N) — metres
```

Pairwise Euclidean distance matrix using `scipy.spatial.KDTree`.
*(Not yet implemented.)*

### `network_distance_matrix`

```
hotelling.spatial.network_distance_matrix(
    locations_a: np.ndarray,  # shape (M, 2) — (lon, lat) WGS84
    locations_b: np.ndarray,  # shape (N, 2) — (lon, lat) WGS84
    osrm_base_url: str = "http://router.project-osrm.org",
    cache_path: Path | None = None,
) -> np.ndarray               # shape (M, N) — metres
```

Pairwise network routing distance matrix via the OSRM table API.
*(Not yet implemented.)*

---

## Module: `exe.py` — Pipeline Orchestrator

### `run_default_data_pipeline`
hotelling.spatial.run_default_data_pipeline(
lor_year: int = 2021,
ringbahn_relation_id: int = 14983,
buffer_distance: float = 500.0,
extend_selection_by: int = 6,
ihk_path: Path | None = None,
) -> None

Run the complete Berlin inner-Ringbahn spatial data pipeline in seven phases:

| Phase | Description | Key output |
|---|---|---|
| 1 — Download | Census, boundaries, LOR, ESIx/MSS, GTFS, OSM POIs | `data/raw/` |
| 2 — Filter | Clip Zensus to Berlin boundary | `zensus2022_grid_filtered.parquet` |
| 3 — LOR selection | Select Ringbahn-area LOR districts | `lor_ringbahn.parquet` |
| 4 — Grid construction | `find_optimal_rectangle` → INSPIRE 100 m lattice | `pop_grid.parquet` |
| 5 — Enrichment | LOR attrs, ESIx/MSS, IHK (cell + building), AABPL clustering, transport hubs | `gebaeude_stadtstruktur.parquet`, `prime_location_clusters.parquet`, `grid_with_stations.parquet` |
| 6 — POI layers | Supermarket normalisation, per-cell chain flags, LCC mall scores | `supermarkets.parquet`, `grid_malls.parquet` |
| 7 — Assembly | Validate schema, fill defaults | `simulation_grid.parquet` |

> **Grid boundary note (Phase 4):** The grid is built inside an optimal
> *rectangular* bounding box (`find_optimal_rectangle`) rather than the
> irregular LOR polygon union. This guarantees a regular INSPIRE 100 m
> lattice with no partial edge cells — required by the simulation engine.
> The rectangle is centred on the Ringbahn boundary centroid and fitted to
> maximise population density. Parameters: `rect_buffer_distance=350.0 m`,
> `rect_augment_layers=(2, 0, 4, 2)`. See ADR-019.

**IHK note:** IHK business microdata cannot be downloaded automatically.
Place `2023_12_IHK_Berlin_Gewerbedaten.csv` in `data/raw/` before running,
or pass its path via the `ihk_path` parameter.  The pipeline proceeds without
it (employment columns will be absent) if the file is missing.

---
## Module: `assembly.py` — Grid Assembly

### `add_lor_attributes`
hotelling.spatial.add_lor_attributes(
grid: geopandas.GeoDataFrame,
lor: geopandas.GeoDataFrame,
) -> geopandas.GeoDataFrame

Spatial-join LOR planning-area identifiers (`PLR_ID`, `PLR_NAME`) to grid
cells by matching each cell's centroid to the containing LOR polygon.

---

### `add_poi_layer`
hotelling.spatial.add_poi_layer(
grid: geopandas.GeoDataFrame,
pois: geopandas.GeoDataFrame,
chain_col: str = "chain",
) -> geopandas.GeoDataFrame

Count and classify OSM POIs per grid cell.  Adds `poi_count`, `poi_chains`,
and per-chain boolean flags (e.g. `has_Rewe`, `has_Lidl`).

---

### `assemble_simulation_grid`
hotelling.spatial.assemble_simulation_grid(
pop_grid: geopandas.GeoDataFrame,
lor: geopandas.GeoDataFrame,
pois: geopandas.GeoDataFrame,
) -> geopandas.GeoDataFrame

Assemble the final simulation-ready grid from all spatial layers.
Guarantees the output columns: `x_mp_100m`, `y_mp_100m`, `geometry`
(100 m² Polygon, EPSG:3035), `Einwohner`, `PLR_ID`, `PLR_NAME`, `poi_count`.

---

### `add_lcc_layer`

```
hotelling.spatial.add_lcc_layer(
    grid: geopandas.GeoDataFrame,
    lcc_gdf: geopandas.GeoDataFrame,
) -> geopandas.GeoDataFrame
```

Add Local Commercial Centre mall-intersection data to the simulation grid.
Processes `shop=mall` features from `fetch_pois(type="LCC")`. For each grid
cell, computes what fraction of each overlapping mall's area falls within
that cell. Adds columns: `mall_area` (m²), `mall_intersection_fraction`,
`has_mall` (bool). Saves intermediate to `data/processed/grid_malls.parquet`.

---
## Module: `city_data.py` — City Data Downloads and Processing

### Download functions

| Function | Output file(s) | Source |
|---|---|---|
| `download_index_data()` | `esix.gpkg`, `mss.gpkg` | Berlin GDI WFS |
| `download_stadtstruktur()` | `stadtstruktur.gpkg`, `gebaeude.gpkg`, `zentren.gpkg` | Berlin GDI WFS |
| `download_station_data()` | `db_station_data.csv`, `gtfs-2024/` | DB InfraGo / VBB |

### Processing functions

**`process_ihk_data(grid, ihk_path)`**
Load IHK Berlin business microdata CSV, parse employee-count ranges
(e.g. `"1 - 3 Beschäftigte"` → midpoint 2), spatial-join to grid cells,
and add an `empl` column (total employment per cell).
Requires manually-placed file; skipped gracefully if absent.

**`process_esix_mss_data(grid)`**
Load `esix.gpkg` / `mss.gpkg`, reproject to EPSG:3035, spatial-join to
grid, add `esix_score` and `mss_score` columns.

**`identify_transport_hubs(grid, gtfs_dir=None)`**
Load OSM station POIs, spatial-join to grid cells, match DB station classes,
add `station_count`, `station_names`, `station_class`, `matched_db_station`
columns. Saves `data/processed/grid_with_stations.parquet`.

**`identify_cbd(grid, zentren_path=None)`**
*(Deprecated — ADR-018; returns grid unchanged with a DeprecationWarning.)*

### `process_gebaeude_stadtstruktur`

```
hotelling.spatial.process_gebaeude_stadtstruktur(
    gebaeude_path: Path | None = None,
    stadtstruktur_path: Path | None = None,
    ihk_path: Path | None = None,
) -> geopandas.GeoDataFrame
```

Build the enriched building-level GeoDataFrame used by the AABPL pipeline.

Steps:
1. Load ALKIS building footprints (`gebaeude.gpkg`) and urban-structure
   classification (`stadtstruktur.gpkg`).
2. Intersects sjoin with tie-breaking by maximum intersection area.
3. Impute `aog` (above-ground floors) for unmatched buildings via
   `bezgfk` → AOG lookup.
4. Convert `hoh` to bool.
5. Apply `gebaeude_capacity` to compute `efficiency`, `usable_area_m2`,
   `employee_hard_cap`.
6. Match IHK geocoordinates to nearest building via STRtree (≤500 m threshold).
7. Apply employee hard cap: `approx_empl = min(empl, employee_hard_cap)`.

Saved output: `data/processed/gebaeude_stadtstruktur.parquet`.

### `run_prime_location_clustering`

```
hotelling.spatial.run_prime_location_clustering(
    gebaeude_stadtstruktur: geopandas.GeoDataFrame,
    k_percentile: float = 99.5,
    min_empl: float = 10.0,
    radius_m: int = 500,
) -> geopandas.GeoDataFrame
```

Detect employment-dense prime-location clusters using the AABPL algorithm
(Automatic Algorithm for Boundary-Identification of Prime Locations).
Wraps `scripts/aabpl_wrapper.detect_employment_clusters`.

Defines the φ_i^prime component (employment-density footfall weight) for the
demand model, as per ADR-018. Requires the `aabpl` package.

Saved output: `data/processed/prime_location_clusters.parquet`.

---

## CRS Reference

| Layer | CRS | Notes |
|---|---|---|
| OSM POIs (`fetch_pois`) | **EPSG:4326** | Raw WGS84 degrees from Overpass |
| City / relation boundaries | **EPSG:3035** (non-std GeoJSON) | Use `load_boundary()` only |
| Zensus grid | **EPSG:3035** | INSPIRE coordinate system |
| LOR shapes | **EPSG:3035** | Reprojected from original Shapefile |
| Folium/Leaflet | EPSG:3857 internally | Accepts WGS84 for bounds |

Always reproject POI data before spatial joins with census or LOR data:

```python
pois_3035 = pois.to_crs("EPSG:3035")
```

See `docs/crs-handling-and-known-issues.md` for full CRS inventory and known issues.

---

## File I/O Reference

| File | Format | Written by | Loaded by | CRS |
|---|---|---|---|---|
| `data/raw/city_boundary_{city}.geojson` | GeoJSON (EPSG:3035, non-std) | `download_city_boundary` | `load_boundary` | EPSG:3035 |
| `data/raw/relation_boundary_{id}.geojson` | GeoJSON (EPSG:3035, non-std) | `download_relation_boundary` | `load_boundary` | EPSG:3035 |
| `data/raw/lor_shapes_2019.parquet` | Parquet | `download_lor_shapes(if_old=True)` | `gpd.read_parquet` | EPSG:3035 |
| `data/raw/lor_shapes_2021.parquet` | Parquet | `download_lor_shapes(if_old=False)` | `gpd.read_parquet` | EPSG:3035 |
| `data/raw/zensus2022_grid.parquet` | Parquet | `download_zensus_2022` | `load_zensus_2022` | EPSG:3035 |
| `data/raw/zensus2022_grid_filtered.parquet` | Parquet | `filter_zensus_2022` | `gpd.read_parquet` | EPSG:3035 |
| `data/raw/OSM_POIs_{city}.parquet` | Parquet | `fetch_pois` | `gpd.read_parquet` + `_add_point_column` | EPSG:4326 |
| `data/raw/OSM_POIs_Berlin_LCC.parquet` | Parquet | `fetch_pois(type="LCC")` | `gpd.read_parquet` + `_add_point_column` | EPSG:4326 |
| `data/raw/OSM_POIs_Berlin_stations.parquet` | Parquet | `fetch_pois(type="stations")` | `gpd.read_parquet` + `_add_point_column` | EPSG:4326 |
| `data/processed/lor_2019.parquet` | Parquet | `join_lor_names(if_old=True)` | `gpd.read_parquet` | EPSG:3035 |
| `data/processed/lor_2021.parquet` | Parquet | `join_lor_names(if_old=False)` | `gpd.read_parquet` | EPSG:3035 |
| `data/processed/lor.parquet` | Parquet | `load_lor(year)` | `gpd.read_parquet` | EPSG:3035 |
| `data/processed/lor_ringbahn.parquet` | Parquet | `select_ringbahn_lor` | `gpd.read_parquet` | EPSG:3035 |
| `data/processed/pop_grid.parquet` | Parquet | `build_full_grid` + `build_grid_polygons` | `gpd.read_parquet` | EPSG:3035 |
| `data/processed/gebaeude_stadtstruktur.parquet` | Parquet | `process_gebaeude_stadtstruktur` | `gpd.read_parquet` | source (EPSG:25833) |
| `data/processed/prime_location_clusters.parquet` | Parquet | `run_prime_location_clustering` | `gpd.read_parquet` | EPSG:3035 |
| `data/processed/grid_with_stations.parquet` | Parquet | `identify_transport_hubs` | `gpd.read_parquet` | EPSG:3035 |
| `data/processed/grid_malls.parquet` | Parquet | `add_lcc_layer` | `gpd.read_parquet` | EPSG:3035 |
| `data/processed/supermarkets.parquet` | Parquet | `process_supermarkets` (pipeline) | `gpd.read_parquet` | EPSG:3035 |
| `data/processed/simulation_grid.parquet` | Parquet | `assemble_simulation_grid` | `gpd.read_parquet` | EPSG:3035 |

---

## Retry and Error Handling

All Overpass and Nominatim HTTP calls in `osm.py` and `boundaries.py` use:

- **3 attempts** with exponential back-off
- Transient codes (429, 502, 503, 504) trigger a retry after `10 × 2^attempt` seconds
- Non-transient errors are raised immediately via `raise_for_status()`

---

## Dependencies

| Package | Min version | Required for |
|---|---|---|
| `geopandas` | 0.14 | All spatial operations |
| `shapely` | 2.0 | Geometry construction and polygonization |
| `pyproj` | 3.6 | CRS reprojection |
| `requests` | 2.31 | Overpass / Nominatim HTTP |
| `pyarrow` | 14.0 | Parquet read/write |
| `rasterio` | 1.3 | Raster reprojection (notebooks) |
| `py7zr` | 0.20 | LOR `.7z` archive extraction |
| `pdfplumber` | 0.10 | `download_station_data` (DB PDF parsing) |
| `openpyxl` | 3.1 | `join_lor_names` (LOR Excel files) |

Install all at once:

```bash
pip install "hotelling[spatial]"
```
