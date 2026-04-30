"""Administrative boundary geometry: Overpass download and GeoJSON loading.

Key dependencies: geopandas, shapely, pyproj, requests (optional ``[spatial]`` extra).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import requests
import shapely.geometry
import shapely.ops

logger = logging.getLogger(__name__)


def _boundary_is_closed(boundary: gpd.GeoDataFrame) -> bool:
    """Return True if the boundary's first geometry is a closed ring."""
    return boundary.geometry.iloc[0].is_closed


def download_city_boundary(city_name: str) -> None:
    """Download German city boundary from Overpass and save as GeoJSON in EPSG:3035."""
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

    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:4326")
    boundary = boundary_gdf.to_crs("EPSG:3035").geometry.iloc[0]

    feature = {
        "type": "Feature",
        "properties": {
            "city_name": city_name,
            "osm_relation_id": relation.get("id"),
            "admin_level": relation.get("tags", {}).get("admin_level"),
            "source": "OpenStreetMap Overpass",
            "crs": "EPSG:3035",
        },
        "geometry": shapely.geometry.mapping(boundary),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(feature, f)
    logger.info("Saved city boundary for '%s' to %s.", city_name, output_path)


def download_relation_boundary(relation_id: int) -> None:
    """Download the boundary of an OSM relation from Overpass and save as GeoJSON (EPSG:3035)."""
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
        lines.append(
            shapely.geometry.LineString([(pt["lon"], pt["lat"]) for pt in coords]),
        )

    if not lines:
        raise ValueError(f"No way geometry found in {relation_id} relation.")

    polygons = list(shapely.ops.polygonize(shapely.ops.unary_union(lines)))
    if not polygons:
        raise ValueError(f"Could not polygonize {relation_id} relation geometry.")

    relation_polygon_ll = max(polygons, key=lambda p: p.area)
    relation_gdf = gpd.GeoDataFrame(geometry=[relation_polygon_ll], crs="EPSG:4326")
    relation_polygon = relation_gdf.to_crs("EPSG:3035").geometry.iloc[0]

    feature = {
        "type": "Feature",
        "properties": {"source": f"OSM relation/{relation_id}", "crs": "EPSG:3035"},
        "geometry": shapely.geometry.mapping(relation_polygon),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(feature, f)
    logger.info("Saved relation %s boundary to %s.", relation_id, output_path)
    closed = _boundary_is_closed(gpd.GeoDataFrame(geometry=[relation_polygon], crs="EPSG:3035"))
    logger.info("Relation %s boundary ring is %s.", relation_id, "closed" if closed else "open")


def load_boundary(path: Path) -> gpd.GeoDataFrame:
    """Load boundary geometry from a GeoJSON file (Feature, FeatureCollection, or raw geometry).
    
    Reads CRS from the GeoJSON properties if available, otherwise defaults to EPSG:3035.
    """
    logger.info("Loading boundary geometry from %s.", path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract CRS from properties if available
    crs = "EPSG:3035"  # Default CRS
    if data.get("type") == "Feature":
        crs = data.get("properties", {}).get("crs", crs)
        logger.info("Loaded boundary from GeoJSON Feature with CRS %s.", crs)
        return gpd.GeoDataFrame(geometry=[shapely.geometry.shape(data["geometry"])], crs=crs)
    if data.get("type") == "FeatureCollection":
        crs = data.get("features", [{}])[0].get("properties", {}).get("crs", crs)
        logger.info("Loaded boundary from GeoJSON FeatureCollection with CRS %s.", crs)
        return gpd.GeoDataFrame(
            geometry=[shapely.geometry.shape(data["features"][0]["geometry"])],
            crs=crs,
        )
    logger.info("Loaded boundary from raw GeoJSON geometry object with CRS %s.", crs)
    return gpd.GeoDataFrame(geometry=[shapely.geometry.shape(data)], crs=crs)
