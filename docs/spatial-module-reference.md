# `hotelling.spatial` — Module Reference

> **Status:** authoritative reference — updated 2026-04-30  
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
download_lor_shapes()                            # saves data/raw/lor_shapes.parquet

# 5. OSM points-of-interest (supermarkets, convenience stores)
pois = fetch_pois("Berlin")                      # EPSG:4326 GeoDataFrame
pois_3035 = pois.to_crs("EPSG:3035")            # reproject for spatial joins
```

---

## Module: `osm.py`

### `fetch_pois`

```
hotelling.spatial.fetch_pois(
    city: str = "Berlin",
    tags: dict | None = None,
    cache_dir: Path | None = None,
    timeout: int = 180,
) -> geopandas.GeoDataFrame
```

Fetch points-of-interest from OpenStreetMap for any named city.

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
| `city` | `str` | `"Berlin"` | Nominatim place name |
| `tags` | `dict \| None` | `{"shop": ["supermarket", "convenience"]}` | OSM tag filter (see below) |
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
    fallback_name: str | None = None,
) -> str | None
```

Map a `brand:wikidata` QID to a canonical chain name using `CHAIN_QID_MAP`.
Returns `fallback_name` (or `None`) when the QID is absent from the map.

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
as `data/raw/lor_shapes.parquet`.

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
| `data/raw/lor_shapes.parquet` | Parquet | `download_lor_shapes` | `gpd.read_parquet` | EPSG:3035 |
| `data/raw/zensus2022_grid.parquet` | Parquet | `download_zensus_2022` | `load_zensus_2022` | EPSG:3035 |
| `data/raw/zensus2022_grid_filtered.parquet` | Parquet | `filter_zensus_2022` | `gpd.read_parquet` | EPSG:3035 |
| `data/raw/OSM_POIs_{city}.parquet` | Parquet | `fetch_pois` | `gpd.read_parquet` + `_add_point_column` | EPSG:4326 |

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

Install all at once:

```bash
pip install "hotelling[spatial]"
```
