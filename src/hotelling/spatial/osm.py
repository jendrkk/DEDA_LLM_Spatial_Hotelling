"""OSM POI fetcher via Overpass API / osmnx.

Responsibility: fetch and normalize points-of-interest for food retail chains
in a polygon geometry.

Public API: fetch_pois, normalize_chain_name, CHAIN_QID_MAP

Key dependencies: osmnx, geopandas, shapely

References:
    OpenStreetMap contributors;
    osmnx (Boeing 2017).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

# brand:wikidata QID -> canonical chain name mapping (subset for Berlin grocery chains)
CHAIN_QID_MAP: Dict[str, str] = {
    "Q151954": "Rewe",
    "Q11462860": "Penny",
    "Q700965": "Lidl",
    "Q41187": "Aldi Süd",
    "Q125054261": "Aldi Nord",
    "Q685967": "Edeka",
    "Q459358": "Netto",
    "Q2662792": "Kaufland",
    "Q1145968": "dm",
    "Q183538": "Rossmann",
}


def fetch_pois(
    polygon,  # shapely.geometry.Polygon in WGS84
    tags: Optional[Dict[str, object]] = None,
    cache_dir: Optional[Path] = None,
) -> "GeoDataFrame":  # type: ignore[name-defined]
    """Fetch points-of-interest from OpenStreetMap for a given polygon.

    Parameters
    ----------
    polygon : shapely Polygon in WGS84 (EPSG:4326)
    tags : OSM tag filter dict (e.g. {"shop": ["supermarket", "convenience"]})
    cache_dir : if provided, cache Overpass responses as GeoJSON files

    Returns
    -------
    GeoDataFrame with columns: geometry, name, brand, brand:wikidata, chain (normalized)
    """
    raise NotImplementedError


def normalize_chain_name(
    wikidata_qid: Optional[str],
    fallback_name: Optional[str] = None,
) -> Optional[str]:
    """Map a brand:wikidata QID to a canonical chain name.

    Parameters
    ----------
    wikidata_qid : Wikidata QID string (e.g. "Q151954")
    fallback_name : raw brand name to return if QID not found

    Returns
    -------
    Canonical chain name or fallback_name
    """
    raise NotImplementedError
