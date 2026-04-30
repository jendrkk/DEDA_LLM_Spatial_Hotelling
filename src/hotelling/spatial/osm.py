"""OSM POI fetcher via Overpass API.

Responsibility: fetch and normalise points-of-interest from OpenStreetMap for
any city-level area.  All three OSM element types are supported:

* **nodes** — stored as ``Point`` geometries.
* **ways** — closed rings are stored as ``Polygon``; open ways are skipped
  because they cannot represent area features.
* **relations** — reconstructed from member-way geometry as ``Polygon`` or
  ``MultiPolygon`` via Shapely polygonization (outer/inner ring handling).

Every OSM tag found across *any* returned element is preserved as a GeoDataFrame
column; rows that lack a given tag carry ``NaN``.  The column set therefore
grows automatically as new tags appear in the data — no schema is pre-defined.

Each returned row exposes two geometry representations:

``geometry``
    The raw OSM shape (``Point`` / ``Polygon`` / ``MultiPolygon``), used as the
    active GeoDataFrame geometry column.
``point``
    A representative ``Point`` (identical to ``geometry`` for nodes; centroid
    of the polygon for area elements).  Suitable for distance calculations
    where a single coordinate per POI is required.  Stored as plain Shapely
    objects, not as a ``GeoSeries``, so it serialises transparently.

The ``point`` column is always *derived* from ``geometry`` and is not persisted
to the Parquet cache — it is recomputed on every load.

Public API
----------
fetch_pois, normalize_chain_name, CHAIN_QID_MAP

Key dependencies
----------------
geopandas ≥ 0.14, shapely ≥ 2.0, requests ≥ 2.31, pyarrow ≥ 14
(install via ``pip install hotelling[spatial]``)

References
----------
OpenStreetMap contributors — https://www.openstreetmap.org/copyright
Overpass API — https://overpass-api.de
Nominatim geocoding — https://nominatim.org
Boeing G (2017) OSMnx: New methods for acquiring, constructing, analyzing,
    and visualizing complex street networks. *Computers, Environment and Urban
    Systems* 65:126–139.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
import requests
import shapely.geometry
import shapely.ops

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: ``brand:wikidata`` QID → canonical chain name (Berlin grocery / drugstore chains).
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

# ---------------------------------------------------------------------------
# Module-private constants
# ---------------------------------------------------------------------------

_DEFAULT_TAGS: Dict[str, object] = {"shop": ["supermarket", "convenience"]}
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "DEDA-LLM-Spatial-Hotelling/1.0 "
        "(research script; https://github.com/jendrkk/DEDA_LLM_Spatial_Hotelling)"
    ),
    "Accept": "application/json",
    "Content-Type": "text/plain; charset=utf-8",
}
_TRANSIENT_HTTP_CODES = frozenset({429, 502, 503, 504})

# ---------------------------------------------------------------------------
# Internal helpers — Overpass query construction
# ---------------------------------------------------------------------------


def _get_area_id(city: str) -> int:
    """Return the Overpass area ID for *city* via a Nominatim search.

    Overpass area IDs are derived by adding a fixed offset to the raw OSM ID:
    * relation → raw_id + 3 600 000 000
    * way      → raw_id + 2 400 000 000
    * node     → raw_id (no offset)
    """
    params = {
        "q": city,
        "format": "json",
        "limit": 10,
        "featuretype": "city",
        "addressdetails": 0,
    }
    resp = requests.get(
        _NOMINATIM_URL,
        params=params,
        headers={"User-Agent": _HEADERS["User-Agent"]},
        timeout=30,
    )
    resp.raise_for_status()
    results: List[dict] = resp.json()
    if not results:
        raise ValueError(f"Nominatim returned no results for city '{city}'.")

    # Prefer administrative relation results (most accurate area boundary).
    for r in results:
        if r.get("osm_type") == "relation":
            return int(r["osm_id"]) + 3_600_000_000

    r = results[0]
    osm_type = r["osm_type"]
    osm_id = int(r["osm_id"])
    if osm_type == "relation":
        return osm_id + 3_600_000_000
    if osm_type == "way":
        return osm_id + 2_400_000_000
    return osm_id


def _build_tag_filters(tags: Dict[str, object]) -> str:
    """Convert a *tags* dict to a concatenated Overpass QL filter string.

    Each key-value pair becomes one Overpass tag selector:

    * ``list`` value  → regex OR match: ``["key"~"^(v1|v2)$"]``
    * ``True``        → key-exists check: ``["key"]``
    * ``str`` value   → exact match: ``["key"="value"]``

    Examples
    --------
    >>> _build_tag_filters({"shop": ["supermarket", "convenience"]})
    '["shop"~"^(supermarket|convenience)$"]'
    >>> _build_tag_filters({"amenity": "cafe"})
    '["amenity"="cafe"]'
    >>> _build_tag_filters({"healthcare": True})
    '["healthcare"]'
    """
    parts: List[str] = []
    for key, value in tags.items():
        if isinstance(value, (list, tuple)):
            escaped = "|".join(re.escape(str(v)) for v in value)
            parts.append(f'["{key}"~"^({escaped})$"]')
        elif value is True:
            parts.append(f'["{key}"]')
        else:
            parts.append(f'["{key}"="{value}"]')
    return "".join(parts)


def _build_overpass_query(area_id: int, tag_filters: str, timeout: int = 180) -> str:
    """Build a complete Overpass QL query for nodes, ways, and relations.

    Uses ``out geom tags;`` so that:
    * nodes carry ``lat`` / ``lon`` at the element root;
    * ways carry a ``geometry`` list of ``{lat, lon}`` node coordinates;
    * relations carry ``members`` with per-member ``geometry`` lists.
    """
    return (
        f"[out:json][timeout:{timeout}];\n"
        f"area({area_id})->.searchArea;\n"
        f"(\n"
        f"  node{tag_filters}(area.searchArea);\n"
        f"  way{tag_filters}(area.searchArea);\n"
        f"  relation{tag_filters}(area.searchArea);\n"
        f");\n"
        f"out geom tags;\n"
    )


# ---------------------------------------------------------------------------
# Internal helpers — per-element geometry parsers
# ---------------------------------------------------------------------------


def _node_to_geometry(el: dict) -> Optional[shapely.geometry.Point]:
    """Parse an Overpass *node* element to a ``Point(lon, lat)``."""
    lon, lat = el.get("lon"), el.get("lat")
    if lon is None or lat is None:
        return None
    return shapely.geometry.Point(lon, lat)


def _way_to_geometry(
    el: dict,
) -> Optional[Union[shapely.geometry.Polygon, shapely.geometry.LineString]]:
    """Parse an Overpass *way* element (returned by ``out geom``) to a geometry.

    A way is treated as a closed ``Polygon`` only when it has ≥ 4 coordinate
    pairs and the first pair equals the last (i.e. the ring is closed).
    Open ways are returned as ``LineString`` — the caller should discard these
    for area-POI use cases.

    Self-intersecting polygons are repaired via ``buffer(0)``.
    """
    raw: List[dict] = el.get("geometry", [])
    if len(raw) < 2:
        return None
    coords = [(pt["lon"], pt["lat"]) for pt in raw]
    if len(coords) >= 4 and coords[0] == coords[-1]:
        try:
            poly = shapely.geometry.Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)  # type: ignore[assignment]
            return poly
        except Exception:
            return None
    return shapely.geometry.LineString(coords)


def _relation_to_geometry(
    el: dict,
) -> Optional[Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]]:
    """Parse an Overpass *relation* element to a ``Polygon`` or ``MultiPolygon``.

    Member ways with role ``"outer"`` (default when role is absent) form the
    exterior ring(s); member ways with role ``"inner"`` are treated as holes.
    Polygonization is delegated to ``shapely.ops.polygonize``.

    Returns ``None`` when no outer geometry can be constructed.
    """
    outer_lines: List[shapely.geometry.LineString] = []
    inner_lines: List[shapely.geometry.LineString] = []

    for member in el.get("members", []):
        if member.get("type") != "way":
            continue
        raw_geom: List[dict] = member.get("geometry", [])
        if len(raw_geom) < 2:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in raw_geom]
        line = shapely.geometry.LineString(coords)
        if member.get("role") == "inner":
            inner_lines.append(line)
        else:
            outer_lines.append(line)

    if not outer_lines:
        return None

    outer_polys: List[shapely.geometry.Polygon] = list(
        shapely.ops.polygonize(shapely.ops.unary_union(outer_lines))
    )
    if not outer_polys:
        return None

    outer_geom: Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon] = (
        shapely.ops.unary_union(outer_polys) if len(outer_polys) > 1 else outer_polys[0]
    )

    if inner_lines:
        inner_polys = list(shapely.ops.polygonize(shapely.ops.unary_union(inner_lines)))
        if inner_polys:
            outer_geom = outer_geom.difference(shapely.ops.unary_union(inner_polys))

    if outer_geom.is_empty:
        return None
    if not outer_geom.is_valid:
        outer_geom = outer_geom.buffer(0)  # type: ignore[assignment]
    return outer_geom


# ---------------------------------------------------------------------------
# Internal helpers — element parsing and DataFrame assembly
# ---------------------------------------------------------------------------


def _parse_elements(elements: List[dict]) -> List[dict]:
    """Convert raw Overpass API elements to flat record dicts.

    Each record contains at minimum:
    * ``osm_id``   — OSM element ID
    * ``osm_type`` — ``"node"``, ``"way"``, or ``"relation"``
    * ``geometry`` — a Shapely geometry object

    All OSM tags are merged as top-level keys; missing tags in a given record
    will appear as ``NaN`` when a ``GeoDataFrame`` is assembled from the list.

    Open ways (``LineString``) are silently dropped — they cannot represent
    area features.  The ``point`` column is *not* added here; use
    :func:`_add_point_column` after constructing the GeoDataFrame.
    """
    records: List[dict] = []
    for el in elements:
        el_type = el.get("type")
        if el_type not in {"node", "way", "relation"}:
            continue

        geom: Optional[shapely.geometry.base.BaseGeometry] = None

        if el_type == "node":
            geom = _node_to_geometry(el)

        elif el_type == "way":
            candidate = _way_to_geometry(el)
            # Keep only closed ways (Polygon); discard open LineStrings.
            if isinstance(candidate, shapely.geometry.Polygon):
                geom = candidate

        elif el_type == "relation":
            geom = _relation_to_geometry(el)

        if geom is None:
            continue

        record: dict = {
            "osm_id": el.get("id"),
            "osm_type": el_type,
            "geometry": geom,
        }
        # Merge all OSM tags — new keys extend the schema automatically.
        record.update(el.get("tags", {}))
        records.append(record)

    return records


def _add_point_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return *gdf* with a ``point`` column added.

    The column contains plain Shapely ``Point`` objects (not a ``GeoSeries``):

    * For ``Point`` geometries → the geometry itself.
    * For ``Polygon`` / ``MultiPolygon`` geometries → the centroid.

    Because the column is not a ``GeoSeries``, it is transparent to
    ``to_parquet``, ``to_file``, and other serialisers that only handle the
    active geometry column.  The column is therefore *not* persisted in the
    Parquet cache and is recomputed on every cache load.
    """
    gdf = gdf.copy()
    gdf["point"] = gdf.geometry.apply(
        lambda g: g if g.geom_type == "Point" else g.centroid
    )
    return gdf


# ---------------------------------------------------------------------------
# Internal helpers — HTTP with retry
# ---------------------------------------------------------------------------


def _post_with_retry(
    url: str,
    data: bytes,
    timeout: int,
    max_attempts: int = 3,
) -> requests.Response:
    """POST *data* to *url* with exponential back-off on transient HTTP errors.

    Transient codes (429, 502, 503, 504) trigger a retry after ``10 * 2^attempt``
    seconds.  Non-transient HTTP errors are returned immediately so the caller
    can call ``raise_for_status()``.

    Raises
    ------
    requests.RequestException
        If the last attempt raises a network-level exception.
    RuntimeError
        If all attempts return a transient HTTP status code.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, data=data, timeout=timeout, headers=_HEADERS)
            if resp.status_code in _TRANSIENT_HTTP_CODES:
                wait = 10 * (2**attempt)
                logger.warning(
                    "HTTP %d on attempt %d/%d — retrying after %ds.",
                    resp.status_code,
                    attempt + 1,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = 10 * (2**attempt)
                logger.warning(
                    "Request error on attempt %d/%d: %s — retrying after %ds.",
                    attempt + 1,
                    max_attempts,
                    exc,
                    wait,
                )
                time.sleep(wait)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(
        f"Overpass request returned transient HTTP error after {max_attempts} attempts."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_chain_name(
    wikidata_qid: Optional[str],
    fallback_name: Optional[str] = None,
) -> Optional[str]:
    """Map a ``brand:wikidata`` QID to a canonical chain name.

    Parameters
    ----------
    wikidata_qid:
        Wikidata QID string (e.g. ``"Q151954"``).  Pass ``None`` to skip the
        lookup and fall back to *fallback_name*.
    fallback_name:
        Raw brand or name string returned when *wikidata_qid* is ``None`` or
        absent from :data:`CHAIN_QID_MAP`.

    Returns
    -------
    str | None
        Canonical chain name from :data:`CHAIN_QID_MAP`, *fallback_name*, or
        ``None`` when both are unavailable.

    Examples
    --------
    >>> normalize_chain_name("Q151954")
    'Rewe'
    >>> normalize_chain_name("Q999999", fallback_name="MyStore")
    'MyStore'
    >>> normalize_chain_name(None) is None
    True
    """
    if wikidata_qid and wikidata_qid in CHAIN_QID_MAP:
        return CHAIN_QID_MAP[wikidata_qid]
    return fallback_name


def fetch_pois(
    city: str = "Berlin",
    tags: Optional[Dict[str, object]] = None,
    cache_dir: Optional[Path] = None,
    timeout: int = 180,
) -> gpd.GeoDataFrame:
    """Fetch points-of-interest from OpenStreetMap for a given city.

    Elements of all three OSM types (node, way, relation) are returned.  All
    OSM tags found in *any* element are preserved as GeoDataFrame columns; rows
    that lack a given tag carry ``NaN``.  The column set therefore grows
    automatically when new tags appear in the retrieved data.

    Two geometry representations are provided per row:

    ``geometry``
        Active GeoDataFrame geometry column.  ``Point`` for nodes;
        ``Polygon`` / ``MultiPolygon`` for closed ways and relations.
    ``point``
        Representative ``Point`` (equals ``geometry`` for nodes; centroid of
        the polygon for area elements).  Use this column for distance
        calculations that require a single coordinate per POI.

    Results are cached as a Parquet file so that repeated calls for the same
    *city* are fast.  The ``point`` column is re-derived on every load (it is
    not written to the cache file).

    Parameters
    ----------
    city:
        Nominatim place name used to locate the Overpass search area
        (e.g. ``"Berlin"``, ``"Munich"``).
    tags:
        OSM tag filter dict.  Keys are OSM tag keys; values may be:

        * a single string — exact match (e.g. ``{"shop": "supermarket"}``)
        * a list of strings — regex OR (e.g.
          ``{"shop": ["supermarket", "convenience"]}``)
        * ``True`` — any non-null value (e.g. ``{"healthcare": True}``)

        Defaults to ``{"shop": ["supermarket", "convenience"]}``.
    cache_dir:
        Directory in which the Parquet cache file is stored and looked up.
        The filename is ``OSM_POIs_{city}.parquet``.  The directory is created
        automatically if it does not exist.  Defaults to ``Path("data/raw")``.
    timeout:
        Overpass API query timeout in seconds.  Increase for large areas or
        many tag filters.

    Returns
    -------
    geopandas.GeoDataFrame
        CRS: EPSG:4326.  Always includes columns ``osm_id``, ``osm_type``,
        ``geometry``, ``point``, and ``chain``.  Additional columns correspond
        to OSM tag keys found in the data.

    Raises
    ------
    requests.HTTPError
        If the Overpass or Nominatim HTTP request fails with a non-transient
        status code.
    ValueError
        If Nominatim returns no results for *city*.
    RuntimeError
        If the Overpass request fails after all retry attempts with transient
        HTTP errors.

    Examples
    --------
    >>> gdf = fetch_pois("Berlin")  # doctest: +SKIP
    >>> gdf.crs.to_epsg()           # doctest: +SKIP
    4326
    >>> "point" in gdf.columns      # doctest: +SKIP
    True
    """
    effective_cache_dir = cache_dir if cache_dir is not None else Path("data/raw")
    output_path = effective_cache_dir / f"OSM_POIs_{city}.parquet"

    if output_path.exists():
        logger.info("Loading cached OSM POIs from %s.", output_path)
        gdf = gpd.read_parquet(output_path)
        return _add_point_column(gdf)

    effective_tags = tags if tags is not None else _DEFAULT_TAGS

    logger.info("Resolving Overpass area ID for '%s' via Nominatim.", city)
    area_id = _get_area_id(city)
    logger.info("Area ID for '%s': %d.", city, area_id)

    tag_filters = _build_tag_filters(effective_tags)
    query = _build_overpass_query(area_id, tag_filters, timeout=timeout)
    logger.debug("Overpass query:\n%s", query)

    logger.info("Fetching POIs for '%s' from Overpass (timeout=%ds).", city, timeout)
    response = _post_with_retry(
        _OVERPASS_URL,
        query.encode("utf-8"),
        timeout=timeout + 60,
    )
    response.raise_for_status()

    elements: List[dict] = response.json().get("elements", [])
    logger.info("Overpass returned %d raw elements.", len(elements))

    records = _parse_elements(elements)
    logger.info("%d usable POI elements parsed (nodes, closed ways, relations).", len(records))

    if not records:
        logger.warning("No usable POI elements found for '%s' with tags %s.", city, effective_tags)
        empty = gpd.GeoDataFrame(
            columns=["osm_id", "osm_type", "geometry", "chain"],
            geometry="geometry",
            crs="EPSG:4326",
        )
        empty["point"] = pd.Series(dtype=object)
        return empty

    # pandas aligns columns across dicts and fills NaN for missing tag keys.
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    # Derive canonical chain name from brand:wikidata or brand fallback.
    wikidata_col: pd.Series = gdf.get(  # type: ignore[assignment]
        "brand:wikidata", pd.Series(dtype=object)
    ).reindex(gdf.index)
    brand_col: pd.Series = gdf.get(  # type: ignore[assignment]
        "brand", pd.Series(dtype=object)
    ).reindex(gdf.index)
    gdf["chain"] = [
        normalize_chain_name(qid, fallback_name=brand)
        for qid, brand in zip(wikidata_col, brand_col)
    ]

    effective_cache_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path)
    logger.info("Cached %d POIs for '%s' to %s.", len(gdf), city, output_path)

    return _add_point_column(gdf)
