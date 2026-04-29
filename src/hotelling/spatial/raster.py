"""Zensus 2022 100m raster loader.

Responsibility: load, clip, and reproject census population rasters for use
as consumer density inputs.

Public API: load_zensus_2022, load_ghs_pop_fallback

Key dependencies: rioxarray, xarray, rasterio, shapely

References:
    Statistisches Bundesamt Zensus 2022;
    Global Human Settlement Layer (GHS-POP).
"""
from __future__ import annotations

from pathlib import Path
from shapely.geometry.polygon import Polygon
from typing import Optional

import urllib.request
import zipfile
import os
import time
import logging
import shutil
import pandas as pd
import shapely.geometry
import shapely.ops
from pyproj import Transformer
import json
import requests
import geopandas as gpd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first existing column name from a prioritized candidate list."""
    normalized_map = {str(col).strip().casefold(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().casefold()
        if key in normalized_map:
            return normalized_map[key]
    raise KeyError(f"None of the candidate columns found: {candidates}")

def _boundary_is_closed(boundary: gpd.GeoDataFrame) -> bool:
    """Check if the boundary is closed."""
    return boundary.geometry.iloc[0].is_closed

def download_zensus_2022() -> None:
    """Download the Zensus 2022 100m population raster from the Destatis portal."""
    logger.info("Starting Zensus 2022 download and conversion.")
    link = "https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Bevoelkerungszahl.zip"
    save_path = "data/raw/zensus2022_grid.zip"
    extract_dir = Path("data/raw/zensus2022_grid")
    urllib.request.urlretrieve(link, save_path)
    logger.info("Downloaded Zensus archive to %s.", save_path)
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(str(extract_dir))
    logger.info("Extracted Zensus archive to %s.", extract_dir)
    os.remove(save_path)

    csv_files = list(extract_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in extracted Zensus folder: {extract_dir}")

    # Prefer the canonical 100m population grid file, fallback to first CSV.
    selected_csv = next(
        (
            path for path in csv_files
            if "Bevoelkerungszahl_100m-Gitter" in path.name and "Zensus2022" in path.name
        ),
        csv_files[0],
    )
    logger.info("Selected Zensus CSV %s for conversion.", selected_csv.name)

    data = pd.read_csv(selected_csv, sep=";")
    
    x_col = _find_first_existing_column(data, ["x_mp_100m", "x_mp", "x"])
    y_col = _find_first_existing_column(data, ["y_mp_100m", "y_mp", "y"])
    
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[x_col], data[y_col]), crs="EPSG:3035")
    
    save_path = "data/raw/zensus2022_grid.parquet"
    gdf.to_parquet(save_path)
    logger.info("Saved processed Zensus population parquet to %s.", save_path)
    shutil.rmtree(extract_dir)
    logger.info("Removed extracted Zensus folder %s after parquet conversion.", extract_dir)

def load_zensus_2022() -> gpd.GeoDataFrame:
    """Load Zensus 2022 100m population raster and reproject to EPSG:3035."""
    logger.info("Loading Zensus parquet from data/raw/zensus2022_grid.parquet.")
    return gpd.read_parquet(Path("data/raw/zensus2022_grid.parquet")).to_crs("EPSG:3035")

def download_city_boundary(
    city_name: str
) -> None:
    """Download German city boundary relation from Overpass and save as GeoJSON."""
    output_path = Path("data/raw") / f"city_boundary_{city_name.replace(' ', '_')}.geojson"
    if output_path.exists():
        logger.info("City boundary already exists at %s; skipping Overpass request.", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:90];
    area["ISO3166-1"="DE"][admin_level=2]->.searchArea;
    rel(area.searchArea)
      ["boundary"="administrative"]
      ["type"="boundary"]
      ["name"="{city_name}"];
    out geom;
    """
    headers = {
        # Required by Overpass operators: clearly identify your script.
        "User-Agent": "DEDA-LLM-Spatial-Hotelling/1.0 (research script)",
        "Accept": "application/json",
        "Content-Type": "text/plain; charset=utf-8",
    }
    session = requests.Session()
    response = None
    logger.info("Requesting city boundary for '%s' from Overpass.", city_name)
    for attempt in range(3):
        response = session.post(overpass_url, data=overpass_query, timeout=120, headers=headers)
        if response.status_code in {429, 502, 503, 504}:
            # Keep request pressure low and retry slowly.
            logger.warning(
                "Overpass returned %s for '%s' (attempt %s/3). Retrying after backoff.",
                response.status_code,
                city_name,
                attempt + 1,
            )
            time.sleep(2 * (attempt + 1))
            continue
        break

    if response is None:
        raise RuntimeError("Overpass request did not produce a response.")
    response.raise_for_status()
    elements = response.json().get("elements", [])
    relations = [el for el in elements if el.get("type") == "relation"]

    if not relations:
        raise ValueError(f"No administrative boundary relation found for '{city_name}' in Germany.")

    # If multiple relations match, prefer local-level boundaries.
    relations.sort(key=lambda rel: int(rel.get("tags", {}).get("admin_level", 99)))
    relation = relations[-1]

    outer_lines = []
    inner_lines = []
    for member in relation.get("members", []):
        if member.get("type") != "way":
            continue
        coords = member.get("geometry", [])
        if len(coords) < 2:
            continue
        line = shapely.geometry.LineString([(pt["lon"], pt["lat"]) for pt in coords])
        if member.get("role", "outer") == "inner":
            inner_lines.append(line)
        else:
            outer_lines.append(line)

    if not outer_lines:
        raise ValueError(f"Boundary relation for '{city_name}' has no outer geometry.")

    outer_polygons = list(shapely.ops.polygonize(shapely.ops.unary_union(outer_lines)))
    if not outer_polygons:
        raise ValueError(f"Could not polygonize outer boundary for '{city_name}'.")

    boundary = shapely.ops.unary_union(outer_polygons)
    if inner_lines:
        inner_polygons = list(shapely.ops.polygonize(shapely.ops.unary_union(inner_lines)))
        if inner_polygons:
            boundary = boundary.difference(shapely.ops.unary_union(inner_polygons))

    feature = {
        "type": "Feature",
        "properties": {
            "city_name": city_name,
            "osm_relation_id": relation.get("id"),
            "admin_level": relation.get("tags", {}).get("admin_level"),
            "source": "OpenStreetMap Overpass",
        },
        "geometry": shapely.geometry.mapping(boundary),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(feature, f)
    logger.info("Saved city boundary for '%s' to %s.", city_name, output_path)


def download_relation_boundary(relation_id: int) -> None:
    """Download the boundary of a relation from Overpass.
    
    Args:
        relation_id: ID of the relation
    """
    output_path = Path("data/raw") / f"relation_boundary_{relation_id}.geojson"

    if output_path.exists():
        logger.info("Relation boundary already exists at %s; skipping Overpass request.", output_path)
        return

    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:90];
    relation({relation_id});
    out geom;
    """
    headers = {
        "User-Agent": "DEDA-LLM-Spatial-Hotelling/1.0 (research script)",
        "Accept": "application/json",
        "Content-Type": "text/plain; charset=utf-8",
    }
    response = requests.post(overpass_url, data=overpass_query, timeout=120, headers=headers)
    response.raise_for_status()
    elements = response.json().get("elements", [])
    relations = [el for el in elements if el["type"] == "relation"]
    if not relations:
        raise ValueError(f"{relation_id} relation not found in Overpass response.")
    
    lines = []
    for member in relations[0].get("members", []):
        if member.get("type") != "way":
            continue
        coords = member.get("geometry", [])
        if len(coords) < 2:
            continue
        lines.append(shapely.geometry.LineString(
            [(pt["lon"], pt["lat"]) for pt in coords]
        ))
    
    if not lines:
        raise ValueError(f"No way geometry found in {relation_id} relation.")
    
    # Polygonize the ring — result is the interior of the Ringbahn
    polygons = list(shapely.ops.polygonize(shapely.ops.unary_union(lines)))
    if not polygons:
        raise ValueError(f"Could not polygonize {relation_id} relation geometry.")
    
    # Take the largest polygon (the relation interior)
    relation_polygon = max(polygons, key=lambda p: p.area)
    
    relation_gdf = gpd.GeoDataFrame(geometry=[relation_polygon], crs="EPSG:3035")
    relation_polygon = relation_gdf.to_crs("EPSG:3035").geometry[0]
    
    feature = {
        "type": "Feature",
        "properties": {"source": f"OSM relation/{relation_id}", "crs": "EPSG:3035"},
        "geometry": shapely.geometry.mapping(relation_polygon),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(feature, f)
    logger.info(f"Saved relation {relation_id} boundary to %s.", output_path)
    logger.info(f"Relation {relation_id} boundary is {'closed' if _boundary_is_closed(relation_gdf) else 'open'}.")

def load_boundary(
    path: Path,
) -> gpd.GeoDataFrame:
    """Load boundary from a GeoJSON file."""
    logger.info("Loading boundary geometry from %s.", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") == "Feature":
        logger.info("Loaded boundary from GeoJSON Feature.")
        return gpd.GeoDataFrame(geometry=[shapely.geometry.shape(data["geometry"])], crs="EPSG:3035")
    if data.get("type") == "FeatureCollection":
        logger.info("Loaded boundary from GeoJSON FeatureCollection.")
        return gpd.GeoDataFrame(geometry=[shapely.geometry.shape(data["features"][0]["geometry"])], crs="EPSG:3035")
    logger.info("Loaded boundary from raw GeoJSON geometry object.")
    return gpd.GeoDataFrame(geometry=[shapely.geometry.shape(data)], crs="EPSG:3035")

def download_lor_shapes() -> None:
    """Download LOR shapes from the Senatsverwaltung für Stadtentwicklung (SenStadt) for Berlin.
    """
    logger.info("Starting LOR shapes download and conversion.")
    # url = "https://www.berlin.de/sen/sbw/_assets/stadtdaten/stadtwissen/lebensweltlich-orientierte-raeume/lor_2021-01-01_k3_shapefiles_nur_id.7z?ts=1770289259" new url
    url = "https://www.berlin.de/sen/sbw/_assets/stadtdaten/stadtwissen/lebensweltlich-orientierte-raeume/lor_2019-01-01_shapefiles_nur_id.7z?ts=1770289260"
    save_path = "data/raw/lor_shapes.7z"
    extract_dir = Path("data/raw/lor_shapes")
    urllib.request.urlretrieve(url, save_path)
    logger.info("Downloaded LOR archive to %s.", save_path)
    try:
        import py7zr  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "Missing optional dependency 'py7zr' required to extract .7z archives. "
            "Install it with: pip install py7zr"
        ) from exc

    with py7zr.SevenZipFile(save_path, mode="r") as archive:
        archive.extractall(path=str(extract_dir))
    logger.info("Extracted LOR archive to %s.", extract_dir)
    os.remove(save_path)

    shapefiles = list(extract_dir.glob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(f"No shapefiles found in extracted LOR directory: {extract_dir}")

    # Prefer PLR (most granular LOR level), then PGR, then BZR.
    priority_tokens = ["_PLR_", "_PGR_", "_BZR_"]
    selected = None
    for token in priority_tokens:
        selected = next((path for path in shapefiles if token in path.name), None)
        if selected is not None:
            break
    if selected is None:
        selected = shapefiles[0]

    logger.info("Selected LOR shapefile %s for conversion.", selected.name)
    data = gpd.read_file(selected)
    data = data.to_crs(crs="EPSG:3035")
    logger.info("Reprojected to CRS EPSG:3035.")
    
    parquet_path = Path("data/raw/lor_shapes.parquet")
    data.to_parquet(parquet_path)
    logger.info("Saved LOR parquet to %s.", parquet_path)
    shutil.rmtree(extract_dir)
    logger.info("Removed extracted LOR folder %s after parquet conversion.", extract_dir)

def download_local_shapes() -> None:
    """Download the shapes of the local planning areas. This method can be implemented for other city if needed.
    """
    raise NotImplementedError("This method is not implemented yet.")

def filter_zensus_2022(boundary_path: Path) -> None:
    """Filter Zensus 2022 population grid to a city boundary.
    
    Args:
        boundary_path: Path to the boundary geojson file
    """
    zensus_path = Path("data/raw/zensus2022_grid.parquet")
    zensus = gpd.read_parquet(zensus_path)
    # Geometry col from lon lat
    zensus["geometry"] = gpd.points_from_xy(zensus.x_mp_100m, zensus.y_mp_100m)
    zensus = gpd.GeoDataFrame(zensus, geometry="geometry", crs="EPSG:3035")
    
    # Load geojson boundary
    boundary = load_boundary(boundary_path)
    # Extract the single boundary geometry (boundary is a 1-row GeoDataFrame)
    boundary_geom = boundary.geometry.iloc[0]
    zensus = zensus[zensus.geometry.within(boundary_geom)]
    logger.info("Filtered Zensus 2022 population grid to city boundary.")
    logger.info("Filtered %s population grid rows.", len(zensus))
    parquet_path = Path("data/raw/zensus2022_grid_filtered.parquet")
    zensus.to_parquet(parquet_path)
    logger.info("Saved filtered Zensus 2022 population grid to %s.", parquet_path)

def build_full_grid(
    boundary: gpd.GeoDataFrame,  # must be in EPSG:3035
    zensus: gpd.GeoDataFrame,
    cell_size: float = 100.0,
) -> gpd.GeoDataFrame:
    """Return full INSPIRE 100m grid inside boundary, with 0 for unpopulated cells.
    
    The Zensus 2022 suppresses cells with 0 or very low population. This function
    reconstructs the complete grid by generating all INSPIRE-aligned cell centres
    within the boundary and left-joining the census data.
    """
    import numpy as np

    minx, miny, maxx, maxy = boundary.bounds

    # Snap to INSPIRE grid alignment (multiples of cell_size in EPSG:3035)
    xs = np.arange(
        int(np.floor(minx / cell_size) * cell_size),
        int(np.ceil(maxx  / cell_size) * cell_size) + 1,
        cell_size, dtype=np.int64
    )
    ys = np.arange(
        int(np.floor(miny / cell_size) * cell_size),
        int(np.ceil(maxy  / cell_size) * cell_size) + 1,
        cell_size, dtype=np.int64
    )
    xx, yy = np.meshgrid(xs, ys)
    full = gpd.GeoDataFrame(
        {"x_mp_100m": xx.ravel(), "y_mp_100m": yy.ravel()},
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs="EPSG:3035",
    )

    # Spatial clip to boundary polygon
    full_gdf = full[full.within(boundary)].copy()
    logger.info("Full grid has %s cells inside boundary.", len(full_gdf))

    # Merge census data — unmatched cells get Einwohner = 0
    full_gdf = full_gdf.merge(zensus, on=["x_mp_100m", "y_mp_100m"], how="left")
    full_gdf["Einwohner"] = full_gdf["Einwohner"].fillna(0).astype(np.int32)
    logger.info("Populated: %s cells, Empty: %s cells.", (full_gdf["Einwohner"] > 0).sum(), (full_gdf["Einwohner"] == 0).sum())
    return full_gdf

def main():
    logger.info("Starting raster module main workflow.")
    download_zensus_2022()
    download_city_boundary("Berlin")
    download_relation_boundary(14983) # Berlin outer S-Bahn ring
    download_lor_shapes()
    filter_zensus_2022(Path("data/raw/city_boundary_Berlin.geojson"))
    logger.info("Completed raster module main workflow.")

if __name__ == "__main__":
    main()