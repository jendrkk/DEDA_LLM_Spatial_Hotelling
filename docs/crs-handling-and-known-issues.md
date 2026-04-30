# CRS Handling and Known Projection Issues

> **Status:** active reference — updated 2026-04-30  
> **Scope:** all spatial data in this repository (`src/hotelling/spatial/`, `data/raw/`, `notebooks/test.ipynb`)

---

## CRS inventory

| File / layer | Stored CRS | Notes |
|---|---|---|
| `zensus2022_grid.parquet` | EPSG:3035 | Downloaded as EPSG:3035; geometry = Points (midpoints) |
| `zensus2022_grid_filtered.parquet` | EPSG:3035 | Filtered by midpoint `.within()` Berlin boundary |
| `city_boundary_Berlin.geojson` | **EPSG:3035** (non-standard) | GeoJSON contains EPSG:3035 metre coordinates, NOT WGS84 degrees. CRS stored in `properties.crs`. Must be loaded via `load_boundary()` — never with `gpd.read_file()` directly. |
| `relation_boundary_14983.geojson` | **EPSG:3035** (non-standard) | Same as above. |
| `lor_shapes.parquet` | EPSG:3035 | Loaded and verified in pipeline. |
| `OSM_POIs_{city}.parquet` | **EPSG:4326** | Parquet cache written by `fetch_pois()`. Mixed geometry types (Point / Polygon / MultiPolygon). Loaded via `gpd.read_parquet()`. The `point` column is plain Shapely objects (not a GeoSeries) and is recomputed on load — not stored in the file. Reproject to EPSG:3035 before spatial joins with census / LOR data. |
| Folium / Leaflet map tiles | EPSG:3857 (Web Mercator) | Leaflet renders everything in EPSG:3857. |
| `ImageOverlay` bounds | WGS84 (EPSG:4326) | Folium expects `[[lat_min, lon_min], [lat_max, lon_max]]` but internally converts to EPSG:3857. |

---

## Issue 1 — Raster projection mismatch (FIXED)

### What was wrong

The Zensus population raster was built in **EPSG:3035 pixel space** (one pixel = one 100×100 m LAEA cell), then handed to Folium's `ImageOverlay` via WGS84 corner coordinates. Leaflet converts those corners to EPSG:3857 and bilinearly stretches the image across the resulting rectangle. Because EPSG:3035 is a non-conformal equal-area projection (Lambert Azimuthal), the pixel grid is not aligned with the EPSG:3857 grid. The result: every cell appeared shifted, squished, and slightly rotated relative to the OSM tile background.

### Fix applied (`notebooks/test.ipynb`, Cell 5)

After rasterizing in EPSG:3035, reproject the RGBA raster array to **EPSG:3857** using `rasterio.warp.reproject` with `Resampling.nearest`. Derive the `ImageOverlay` bounds from the EPSG:3857 extent converted to WGS84, not from the EPSG:3035 extent. A raster in EPSG:3857 pixel space stretched over EPSG:3857-derived lat/lon bounds is an identity stretch in Leaflet — pixel-perfect alignment.

```python
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pyproj import CRS, Transformer

src_crs = CRS.from_epsg(3035)
dst_crs = CRS.from_epsg(3857)

dst_transform, dst_width, dst_height = calculate_default_transform(
    src_crs, dst_crs, src_width, src_height,
    left=minx_proj, bottom=miny_proj, right=maxx_proj, top=maxy_proj,
)
# Reproject each RGBA band independently with Resampling.nearest
# Then derive WGS84 bounds from dst_transform + dst_width/dst_height
```

---

## Issue 2 — Non-standard GeoJSON CRS (known fragility, not yet migrated)

### What is wrong

`download_city_boundary()` and `download_relation_boundary()` in `boundaries.py` save geometry in **EPSG:3035** inside `.geojson` files. This violates RFC 7946 (GeoJSON standard mandates WGS84 / EPSG:4326). The files are written with EPSG:3035 metre coordinates (e.g. `[4534589.19, 3277957.14]`), not degree coordinates.

The pipeline works because `load_boundary()` reads the custom `properties.crs` field and forces the correct CRS. However:

- Any external tool (`gpd.read_file()`, QGIS auto-detect, any standard GeoJSON reader) will silently interpret those values as degrees, producing wildly wrong geometry with zero error.
- The notebook has a `# CRITICAL FIX` comment documenting exactly this risk in Cell 1.

### Recommended migration (not yet done)

Store boundary files in WGS84 (EPSG:4326) as proper GeoJSON, and convert to EPSG:3035 on load inside `load_boundary()`. This would make the files interoperable with any GIS tool and eliminate the fragility. The change affects `download_city_boundary()`, `download_relation_boundary()`, and `load_boundary()`. A separate ADR should record this decision before implementation.

**Until migration:** always use `load_boundary(path)` — never `gpd.read_file(path)` — for any boundary GeoJSON in this repo.

---

## Issue 3 — Edge-cell boundary overflow in visualization (FIX PENDING)

### What is wrong

`filter_zensus_2022()` in `census.py` filters cells using a **midpoint `.within()` boundary** predicate. This is semantically correct for the simulation (each cell's representative location is inside scope). But in visualization, each cell is rendered as a 100×100 m square (±50 m from the midpoint). Any cell whose midpoint falls within 0–50 m of the boundary line will pass the filter, but its rendered square protrudes beyond the boundary.

**Quantification (Berlin boundary, 2026-04-30):**
- Affected cells: **408 out of 40,663** (~1.0%)
- Midpoint distance to boundary line: 0.1 m – 69.5 m

This produces the visual artifact of isolated cells appearing outside the red boundary line in the Folium map.

### Fix to apply (`notebooks/test.ipynb`, Cell 5)

After building `zensus_3035_polygons` (the GeoDataFrame with 100×100 m squares), clip to the Berlin boundary before rasterizing:

```python
berlin_union = berlin_3035.geometry.unary_union
zensus_3035_polygons['geometry'] = zensus_3035_polygons.geometry.intersection(berlin_union)
zensus_3035_polygons = zensus_3035_polygons[
    ~zensus_3035_polygons.geometry.is_empty &
    (zensus_3035_polygons.geometry.geom_type.isin(['Polygon', 'MultiPolygon']))
].copy()
```

**Do not change** `filter_zensus_2022()` in `census.py` — the midpoint filter is correct for the data pipeline. Clipping is a visualization concern only.

### Extended recommendation for simulation scope

When filtering to the inner-Ringbahn boundary (`relation_boundary_14983.geojson`) for the actual simulation demand grid, use `.intersects()` (not `.within()`) on the 100 m squares, then clip each square to the boundary. This ensures no demand-cell area is silently excluded at the ring boundary and the demand grid is geometrically exact.

---

## Issue 4 — OSM POI dual-geometry column (design note)

### What it means

`fetch_pois()` returns a GeoDataFrame with two geometry-like columns:

* `geometry` — the active GeoPandas geometry column containing the raw OSM shape
  (`Point` for nodes, `Polygon` / `MultiPolygon` for closed ways and relations).
* `point` — a plain Python object column of Shapely `Point` objects.  For nodes
  this equals `geometry`; for polygon elements it is the centroid.

The `point` column is stored as a plain object column (not a `GeoSeries`) so
that it serialises transparently to Parquet without requiring multi-geometry
Arrow encoding.  It is **not written to the cache file** and is recomputed from
`geometry` on every load via `_add_point_column()`.

### Implications

1. **CRS mismatch with census / LOR data.** OSM POIs are returned in EPSG:4326
   (WGS84 degrees).  Census and LOR layers are in EPSG:3035 (metres).  Always
   reproject before spatial joins:

   ```python
   pois_3035 = gdf.to_crs("EPSG:3035")
   # also update the point column after reprojection:
   pois_3035["point"] = pois_3035.geometry.apply(
       lambda g: g if g.geom_type == "Point" else g.centroid
   )
   ```

2. **Mixed geometry types.** The `geometry` column contains a mix of `Point`,
   `Polygon`, and `MultiPolygon` objects.  GeoPandas handles this correctly
   (stores as `"geometry"` dtype), but some serialisers or visualisation tools
   may require filtering to a single geometry type.

3. **Centroid vs. entrance location.** For large supermarket polygons the
   centroid may fall slightly off the actual entrance.  The `point` column is
   intended for distance calculations only, not for precise entrance mapping.

---

## Summary table

| Issue | Severity | Status | Fix location |
|---|---|---|---|
| Raster in EPSG:3035 overlaid on EPSG:3857 Leaflet map | High (visible misalignment) | ✅ Fixed | `test.ipynb` Cell 5 |
| GeoJSON files storing EPSG:3035 coords (non-standard) | Medium (fragility, no current breakage) | ⚠️ Known, not migrated | `boundaries.py` |
| Edge-cell squares protruding beyond boundary in viz | Low (1% of cells, cosmetic) | 🔧 Fix pending | `test.ipynb` Cell 5 |
| Ringbahn-scope filter should use `.intersects()` on squares | Low (potential for slightly wrong demand at edge) | 🔧 Recommended | `census.py` |
| OSM POIs in EPSG:4326 — must reproject for spatial joins with census/LOR | Low (easy to handle, documented) | ✅ By design | `osm.py` / Issue 4 |
| OSM `point` column not persisted in Parquet cache (recomputed on load) | Informational | ✅ By design | `osm.py` |
