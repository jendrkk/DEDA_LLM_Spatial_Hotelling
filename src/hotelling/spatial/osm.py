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
from typing import Callable, Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
import requests
import shapely.geometry
import shapely.ops

logger = logging.getLogger(__name__)

__all__ = [
    "CHAIN_QID_MAP",
    "CHAIN_TYPE_MAP",
    "fetch_pois",
    "fetch_site_eligibility_signals",
    "normalize_chain_name",
    "process_supermarkets",
]

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: ``brand:wikidata`` QID → canonical chain name.
#: QIDs verified against actual Berlin OSM data (overpass query, May 2026).
CHAIN_QID_MAP: Dict[str, str] = {
    # --- Major grocery chains ---
    "Q701755": "Edeka",                  # Edeka / EDEKA (incl. E-Center, nah und gut)
    "Q16968817": "Rewe",                 # Rewe / REWE / REWE City / REWE Center
    "Q151954": "Lidl",                   # Lidl
    "Q41171373": "Aldi Nord",            # Aldi Nord / ALDI Nord  (Berlin = Aldi Nord territory)
    "Q879858": "Netto",                  # Netto Marken-Discount / Netto City — merged into single "Netto" chain
    "Q284688": "Penny",                  # Penny / PENNY
    "Q552652": "Netto",                  # Netto — merged with Netto Marken-Discount into one "Netto" chain
    "Q685967": "Kaufland",               # Kaufland
    "Q450180": "Norma",                  # Norma / NORMA
    "Q1548713": "HIT",                   # HIT / HIT Ullrich
    "Q15836148": "NP",                   # NP (Netto-brand discount, Edeka group)
    "Q1022827": "CAP",                   # CAP / CAP-Markt
    "Q1963643": "Nah & Frisch",          # Nah & Frisch
    "Q327854": "Mix Markt",              # Mix Markt
    # --- Rewe subsidiaries ---
    "Q57515238": "Rewe",                 # Nahkauf — operated by Rewe Group
    # --- Bio / organic chains ---
    "Q48883773": "Denns BioMarkt",       # Denns BioMarkt
    "Q864179": "Bio Company",            # Bio Company
    "Q876811": "Alnatura",               # Alnatura / Alnatura Super Natur Markt
    "Q107983669": "LPG BioMarkt",        # LPG BioMarkt (formerly LPG Naturkost)
    # --- Speciality chains ---
    "Q102381911": "Go Asia",             # Go Asia
}

#: Vertical-differentiation tier for each canonical chain name returned by
#: :func:`normalize_chain_name`.  Three tiers map to the demand model's
#: consumer-type partition:
#:   "discount"  → price-sensitive H-type consumers have higher WTP here
#:   "standard"  → mainstream full-assortment chains
#:   "bio"       → premium/organic chains attracting high-WTP L-type consumers
CHAIN_TYPE_MAP: Dict[str, str] = {
    # Discount tier
    "Aldi Nord":              "discount",
    "Lidl":                   "discount",
    "Netto":                  "discount",
    "Penny":                  "discount",
    "Norma":                  "discount",
    # Standard tier
    "Edeka":                  "standard",
    "Rewe":                   "standard",
    "Kaufland":               "standard",
    "HIT":                    "standard",
    "CAP":                    "standard",
    "Nah & Frisch":           "standard",
    "Mix Markt":              "standard",
    # Bio / premium tier
    "Denns BioMarkt":         "bio",
    "Bio Company":            "bio",
    "Alnatura":               "bio",
    "LPG BioMarkt":           "bio",
}

#: Canonical chain names to drop from the simulation entirely, regardless of a
#: valid chain_type. Single-store chains are excluded because each distinct
#: chain spawns one LLM-CEO call per strategic epoch; a one-store chain wastes
#: that call without contributing a meaningful strategic agent. Their stores
#: still exist in OSM but are removed here so they enter neither the demand
#: system nor the CEO layer. Add a name to drop it on the next pipeline run.
_EXCLUDED_CHAINS: frozenset[str] = frozenset({
    "Nah & Frisch",   # single store in the inner ring (see scripts/fix_chain_data.py)
})

# ---------------------------------------------------------------------------
# Module-private constants
# ---------------------------------------------------------------------------

# Secondary normalization map: lowercase brand/name string → canonical chain name.
# Applied when brand:wikidata is absent or not found in CHAIN_QID_MAP.
# All keys must be lowercase; lookup uses ``str.strip().lower()``.
_BRAND_NAME_MAP: Dict[str, str] = {
    # Edeka group
    "edeka": "Edeka",
    "e-center": "Edeka",        # Edeka E-Center large-format stores
    "nah und gut": "Edeka",     # Edeka Nah und gut franchise
    "nah & gut": "Edeka",
    # Rewe group
    "rewe": "Rewe",
    "rewe city": "Rewe",
    "rewe center": "Rewe",
    "nahkauf": "Rewe",          # Nahkauf is a Rewe Group subsidiary
    "nahcity": "Rewe",          # NahCity was a Rewe Group convenience format
    # Lidl
    "lidl": "Lidl",
    # Aldi Nord — Berlin lies entirely within Aldi Nord territory
    "aldi nord": "Aldi Nord",
    "aldi": "Aldi Nord",
    # Netto Marken-Discount and Netto are merged into a single "Netto" chain
    "netto marken-discount": "Netto",
    "netto city": "Netto",
    "netto": "Netto",
    # Penny
    "penny": "Penny",
    # Kaufland (Schwarz Gruppe)
    "kaufland": "Kaufland",
    # Norma
    "norma": "Norma",
    # HIT
    "hit": "HIT",
    # NP
    "np": "NP",
    # CAP
    "cap": "CAP",
    "cap-markt": "CAP",
    # Nah & Frisch
    "nah & frisch": "Nah & Frisch",
    "nah und frisch": "Nah & Frisch",
    # Mix Markt
    "mix markt": "Mix Markt",
    # Bio / organic chains
    "denns biomarkt": "Denns BioMarkt",
    "denns bioladen": "Denns BioMarkt",
    "alnatura": "Alnatura",
    "alnatura super natur markt": "Alnatura",
    "bio company": "Bio Company",
    "lpg biomarkt": "LPG BioMarkt",
    "lpg naturkost": "LPG BioMarkt",   # rebranded to LPG BioMarkt
    # Speciality chains
    "go asia": "Go Asia",
}

_DEFAULT_TAGS: Dict[str, object] = {"shop": ["supermarket"]}

# Large Commercial Centre (LCC) anchor stores — derived from a manually curated
# Overpass Turbo query.  Each dict maps to one union block in the generated
# Overpass QL query; entries with `"brand": True` restrict the result to named
# chains and exclude tiny independent shops sharing the same primary tag.
#
# Equivalent Overpass Turbo blocks:
#   ["shop"="mall"]
#   ["shop"="department_store"]
#   ["shop"="chemist"]["brand"]
#   ["shop"="variety_store"]
#   ["shop"="electronics"]["brand"]
#   ["shop"="doityourself"]
#   ["shop"="furniture"]["brand"]
#   ["shop"="sports"]["brand"]
_LCC_TAGS: List[Dict[str, object]] = [
    {"shop": "mall"},
    {"shop": "department_store"},
    {"shop": "chemist", "brand": True},
    {"shop": "variety_store"},
    {"shop": "electronics", "brand": True},
    {"shop": "doityourself"},
    {"shop": "furniture", "brand": True},
    {"shop": "sports", "brand": True},
]

# Railway stations — nodes, ways, and relations tagged railway=station.
# Equivalent Overpass Turbo block:
#   ["railway"="station"]
_STATIONS_TAGS: Dict[str, object] = {"railway": "station"}

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
# Internal helpers — find repository root
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    current_path = Path.cwd()
    for parent in current_path.parents:
        if (parent / '.git').exists():
            return parent
    raise FileNotFoundError("Could not find the repository root ('.git' directory).")


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


def _build_overpass_query(
    area_id: int,
    tag_filter_blocks: List[str],
    timeout: int = 180,
) -> str:
    """Build a complete Overpass QL query for nodes, ways, and relations.

    *tag_filter_blocks* is a list of tag-filter strings (one per logical tag
    dict).  Each block expands to ``node`` / ``way`` / ``relation`` lines;
    all blocks are OR-unioned inside the same parenthesised query clause.

    Uses ``out geom tags;`` so that:
    * nodes carry ``lat`` / ``lon`` at the element root;
    * ways carry a ``geometry`` list of ``{lat, lon}`` node coordinates;
    * relations carry ``members`` with per-member ``geometry`` lists.
    """
    union_lines = "".join(
        f"  node{block}(area.searchArea);\n"
        f"  way{block}(area.searchArea);\n"
        f"  relation{block}(area.searchArea);\n"
        for block in tag_filter_blocks
    )
    return (
        f"[out:json][timeout:{timeout}];\n"
        f"area({area_id})->.searchArea;\n"
        f"(\n"
        f"{union_lines}"
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
    brand: Optional[str] = None,
    *,
    name: Optional[str] = None,
) -> Optional[str]:
    """Map OSM supermarket tags to a canonical chain name.

    Resolution priority (first match wins):

    1. **brand:wikidata QID** — looked up in :data:`CHAIN_QID_MAP`.
       Most reliable; handles inconsistent brand-name spellings automatically.
    2. **brand field** — case-insensitive lookup in ``_BRAND_NAME_MAP``.
       Canonicalises variant spellings (e.g. ``"REWE"`` / ``"Rewe"``,
       ``"ALDI Nord"`` / ``"Aldi Nord"``).
    3. **name field** — same case-insensitive lookup in ``_BRAND_NAME_MAP``.
       Catches the minority of stores that lack a ``brand`` tag but carry the
       chain name as the element name (e.g. ``name="Aldi Nord"`` without
       ``brand:wikidata``).
    4. **raw brand field** — returned as-is for any unrecognised chain so that
       minor/independent stores keep their identity rather than collapsing to
       ``None``.
    5. ``None`` — when all four sources are unavailable.

    Parameters
    ----------
    wikidata_qid:
        Value of the ``brand:wikidata`` OSM tag (e.g. ``"Q151954"``), or
        ``None`` when absent.
    brand:
        Value of the ``brand`` OSM tag, or ``None`` when absent.
    name:
        Value of the ``name`` OSM tag, used only as a tertiary fallback.

    Returns
    -------
    str | None
        Canonical chain name, or ``None`` when the store cannot be identified.

    Examples
    --------
    >>> normalize_chain_name("Q151954")
    'Lidl'
    >>> normalize_chain_name("Q16968817")
    'Rewe'
    >>> normalize_chain_name(None, brand="REWE City")
    'Rewe'
    >>> normalize_chain_name(None, brand=None, name="Aldi Nord")
    'Aldi Nord'
    >>> normalize_chain_name(None, brand="MyStore")
    'MyStore'
    >>> normalize_chain_name(None) is None
    True
    """
    # Pandas serialises missing tag values as float NaN; normalise to None.
    if not isinstance(wikidata_qid, str):
        wikidata_qid = None
    if not isinstance(brand, str):
        brand = None
    if not isinstance(name, str):
        name = None

    # 1. Wikidata QID lookup
    if wikidata_qid and wikidata_qid in CHAIN_QID_MAP:
        return CHAIN_QID_MAP[wikidata_qid]

    # 2. Brand-name normalisation
    if brand:
        canonical = _BRAND_NAME_MAP.get(brand.strip().lower())
        if canonical is not None:
            return canonical

    # 3. Name-field fallback normalisation
    if name:
        canonical = _BRAND_NAME_MAP.get(name.strip().lower())
        if canonical is not None:
            return canonical

    # 4. Raw brand as-is (unrecognised / independent chains)
    if brand:
        return brand

    return None


def process_supermarkets(
    pois_raw: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Clean, normalise, and clip a raw supermarket POI GeoDataFrame.

    Applies chain-type classification, clips to the simulation grid extent,
    and retains only stores with a recognised canonical chain name.
    Logic extracted from GEO_03_OSM.ipynb.

    Steps:
    1. Reproject to grid CRS (EPSG:3035).
    2. Replace geometry with centroids (all features become Points).
    3. Clip to the union of grid cell polygons.
    4. Add ``chain_type`` column via :data:`CHAIN_TYPE_MAP`.
    5. Drop rows where ``chain`` or ``chain_type`` is null (unrecognised
       independents are excluded), then drop any chain in ``_EXCLUDED_CHAINS``.
    6. Return columns: ``geometry``, ``name``, ``chain``, ``chain_type``.

    Parameters
    ----------
    pois_raw:
        Raw GeoDataFrame from :func:`fetch_pois` with ``type="supermarket"``.
        Must contain a ``chain`` column.  CRS EPSG:4326 expected but any CRS
        is handled via reprojection.
    grid:
        Simulation grid GeoDataFrame with polygon geometry in EPSG:3035.

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned supermarkets in the same CRS as ``grid``, with columns
        ``geometry`` (Point), ``name``, ``chain``, ``chain_type``.
    """
    gdf = pois_raw.copy()
    if gdf.crs is None or gdf.crs != grid.crs:
        gdf = gdf.to_crs(grid.crs)
    gdf["geometry"] = gdf.geometry.centroid

    grid_union = grid.geometry.union_all()
    gdf = gdf[gdf.geometry.within(grid_union)].copy()

    gdf["chain_type"] = gdf["chain"].map(CHAIN_TYPE_MAP)
    gdf = gdf[gdf["chain"].notna() & gdf["chain_type"].notna()].copy()

    # Drop explicitly excluded chains (e.g. single-store chains that would
    # otherwise spawn a wasted per-epoch LLM-CEO call). See _EXCLUDED_CHAINS.
    n_excluded = int(gdf["chain"].isin(_EXCLUDED_CHAINS).sum())
    if n_excluded:
        gdf = gdf[~gdf["chain"].isin(_EXCLUDED_CHAINS)].copy()
        logger.info(
            "process_supermarkets: dropped %d store(s) from excluded chains %s.",
            n_excluded, sorted(_EXCLUDED_CHAINS),
        )

    keep = [c for c in ["geometry", "name", "chain", "chain_type"] if c in gdf.columns]
    return gdf[keep].reset_index(drop=True)


def fetch_pois(
    type: str = "supermarket",
    city: str = "Berlin",
    tags: Optional[Union[Dict[str, object], List[Dict[str, object]]]] = None,
    name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    timeout: int = 180,
) -> gpd.GeoDataFrame:
    """Fetch points-of-interest from OpenStreetMap for a given city.

    Three built-in query profiles are available via *type*:

    ``"supermarket"``
        Fetches all ``shop=supermarket`` elements (the original behaviour).
        A canonical ``chain`` column is added by mapping ``brand:wikidata``
        QIDs through :data:`CHAIN_QID_MAP`.

    ``"LCC"``
        Fetches Large Commercial Centre anchor stores — shopping malls,
        department stores, chemist chains, variety stores, electronics chains,
        DIY/home-improvement stores, furniture retailers, and sports retail
        chains.  The full tag set is defined in the module constant
        ``_LCC_TAGS``.  No ``chain`` column is produced for this profile.

    ``"stations"``
        Fetches all elements tagged ``railway=station`` (S-Bahn, U-Bahn,
        regional, and long-distance rail stations).  The tag set is defined
        in ``_STATIONS_TAGS``.  No ``chain`` column is produced.

    For any other *type* value, the *tags* parameter is used directly (or
    ``_DEFAULT_TAGS`` when *tags* is ``None``); a ``chain`` column is not
    added unless ``type == "supermarket"``.

    All OSM tags found in *any* returned element are preserved as GeoDataFrame
    columns; rows that lack a given tag carry ``NaN``.  The column set grows
    automatically — no schema is pre-defined.

    Two geometry representations are provided per row:

    ``geometry``
        Active GeoDataFrame geometry column.  ``Point`` for nodes;
        ``Polygon`` / ``MultiPolygon`` for closed ways and relations.
    ``point``
        Representative ``Point`` (equals ``geometry`` for nodes; centroid of
        the polygon for area elements).  Suitable for distance calculations.

    Results are cached as a Parquet file keyed by *city* and *type* (or
    *name* when explicitly given).  The ``point`` column is re-derived on
    every load and is not written to the Parquet file.

    Parameters
    ----------
    type:
        Query profile — ``"supermarket"`` or ``"LCC"`` for the built-in
        profiles; any other string uses *tags* directly.
    city:
        Nominatim place name used to locate the Overpass search area
        (e.g. ``"Berlin"``, ``"Munich"``).
    tags:
        OSM tag filter(s) used when *type* is not a built-in profile.
        Pass a **dict** for a single filter block or a **list of dicts** to
        OR-union several independent blocks.  Ignored when
        ``type in {"supermarket", "LCC"}``.
    name:
        Override the cache-file stem.  When provided, the Parquet file is
        named ``OSM_POIs_{city}_{name}.parquet`` instead of the default
        ``OSM_POIs_{city}_{type}.parquet``.
    cache_dir:
        Directory for the Parquet cache file (created if absent).
        Defaults to ``<repo_root>/data/raw``.
    timeout:
        Overpass API query timeout in seconds.

    Returns
    -------
    geopandas.GeoDataFrame
        CRS: EPSG:4326.  Always includes ``osm_id``, ``osm_type``,
        ``geometry``, ``point``, plus all OSM tag keys present in the data.
        A ``chain`` column is added only when ``type == "supermarket"``.

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
    >>> gdf = fetch_pois(type="supermarket", city="Berlin")  # doctest: +SKIP
    >>> gdf.crs.to_epsg()                                     # doctest: +SKIP
    4326
    >>> "chain" in gdf.columns                                # doctest: +SKIP
    True
    >>> lcc = fetch_pois(type="LCC", city="Berlin")            # doctest: +SKIP
    >>> "chain" in lcc.columns                                 # doctest: +SKIP
    False
    >>> stn = fetch_pois(type="stations", city="Berlin")       # doctest: +SKIP
    >>> "chain" in stn.columns                                 # doctest: +SKIP
    False
    """
    effective_cache_dir = (
        cache_dir if cache_dir is not None else _find_repo_root() / Path("data/raw")
    )
    cache_stem = name if name is not None else type
    output_path = effective_cache_dir / f"OSM_POIs_{city}_{cache_stem}.parquet"

    if output_path.exists():
        logger.info("Loading cached OSM POIs from %s.", output_path)
        gdf = gpd.read_parquet(output_path)
        return _add_point_column(gdf)

    # ── Determine tag filters based on type ──────────────────────────────
    if type == "LCC":
        effective_tags: Union[Dict[str, object], List[Dict[str, object]]] = _LCC_TAGS
    elif type == "stations":
        effective_tags = _STATIONS_TAGS
    elif type == "supermarket":
        effective_tags = tags if tags is not None else _DEFAULT_TAGS
    else:
        effective_tags = tags if tags is not None else _DEFAULT_TAGS

    tag_list = [effective_tags] if isinstance(effective_tags, dict) else list(effective_tags)
    tag_filter_blocks = [_build_tag_filters(t) for t in tag_list]

    logger.info("Resolving Overpass area ID for '%s' via Nominatim.", city)
    area_id = _get_area_id(city)
    logger.info("Area ID for '%s': %d.", city, area_id)

    query = _build_overpass_query(area_id, tag_filter_blocks, timeout=timeout)
    logger.debug("Overpass query:\n%s", query)

    logger.info(
        "Fetching POIs for '%s' (type=%r) from Overpass (timeout=%ds).",
        city, type, timeout,
    )
    response = _post_with_retry(
        _OVERPASS_URL,
        query.encode("utf-8"),
        timeout=timeout + 60,
    )
    response.raise_for_status()

    elements: List[dict] = response.json().get("elements", [])
    logger.info("Overpass returned %d raw elements.", len(elements))

    records = _parse_elements(elements)
    logger.info(
        "%d usable POI elements parsed (nodes, closed ways, relations).",
        len(records),
    )

    if not records:
        logger.warning(
            "No usable POI elements found for '%s' with type=%r.", city, type
        )
        base_cols = ["osm_id", "osm_type", "geometry"]
        if type == "supermarket":
            base_cols.append("chain")
        empty = gpd.GeoDataFrame(
            columns=base_cols,
            geometry="geometry",
            crs="EPSG:4326",
        )
        empty["point"] = pd.Series(dtype=object)
        return empty

    # pandas aligns columns across dicts and fills NaN for missing tag keys —
    # the resulting GeoDataFrame therefore contains every attribute returned by
    # the Overpass query, regardless of whether all rows carry that tag.
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    # ── Chain normalisation — supermarket only ───────────────────────────
    if type == "supermarket":
        wikidata_col: pd.Series = gdf.get(  # type: ignore[assignment]
            "brand:wikidata", pd.Series(dtype=object)
        ).reindex(gdf.index)
        brand_col: pd.Series = gdf.get(  # type: ignore[assignment]
            "brand", pd.Series(dtype=object)
        ).reindex(gdf.index)
        name_col: pd.Series = gdf.get(  # type: ignore[assignment]
            "name", pd.Series(dtype=object)
        ).reindex(gdf.index)
        gdf["chain"] = [
            normalize_chain_name(qid, brand=brand, name=nm)
            for qid, brand, nm in zip(wikidata_col, brand_col, name_col)
        ]

    effective_cache_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path)
    logger.info("Cached %d POIs for '%s' to %s.", len(gdf), city, output_path)

    return _add_point_column(gdf)


# ---------------------------------------------------------------------------
# Site-eligibility signal constants
# ---------------------------------------------------------------------------

# ── BLOCKER tags ─────────────────────────────────────────────────────────────
# Every currently occupied ground-floor commercial unit.
# Format: list of dicts; each dict is one independent Overpass tag-block
# (OR-unioned in the query).
# Used by fetch_site_eligibility_signals().
_BLOCKER_TAGS: List[Dict[str, object]] = [
    # All retail shops — any value of shop=*
    {"shop": True},
    # Food & drink service
    {"amenity": ["restaurant", "cafe", "fast_food", "bar", "pub",
                 "biergarten", "food_court", "ice_cream", "nightclub"]},
    # Financial services (bank branches occupy significant ground-floor units)
    {"amenity": ["bank", "bureau_de_change", "money_transfer"]},
    # Health & personal care
    {"amenity": ["pharmacy", "doctors", "dentist", "clinic",
                 "veterinary", "optician"]},
    # Postal & telecoms
    {"amenity": ["post_office", "telephone_office"]},
    # Entertainment / leisure — large footprint (gyms, bowling, etc.)
    {"leisure": ["fitness_centre", "gym", "sports_centre",
                 "bowling_alley", "escape_game", "dance",
                 "amusement_arcade"]},
    # Ground-floor offices
    {"office": True},
    # Craft / workshops (Handwerksbetriebe)
    {"craft": True},
    # Cultural / public services mapped inside commercial buildings
    {"amenity": ["theatre", "cinema", "library", "community_centre",
                 "social_facility", "embassy", "courthouse",
                 "townhall"]},
    # Tourism accommodation (hotel lobbies occupy street-front ground floor)
    {"tourism": ["hotel", "hostel", "guest_house", "motel", "apartment"]},
    # Religious buildings — permanent occupant
    {"amenity": ["place_of_worship"]},
]

# ── VACANT tags ───────────────────────────────────────────────────────────────
# Empty or formerly occupied commercial units — highest-priority candidates.
# Format: list of dicts; each dict is one independent Overpass tag-block.
# Used by fetch_site_eligibility_signals().
_VACANT_TAGS: List[Dict[str, object]] = [
    # Explicitly vacant retail unit (mapper confirmed empty shopfront)
    {"shop": "vacant"},
    # Former shops — all types (disused:shop=supermarket is especially relevant)
    {"disused:shop": True},
    # Abandoned shops (longer-term closure)
    {"abandoned:shop": True},
    # Informal former-use prefix
    {"was:shop": True},
    # Disused large-footprint leisure (gyms ~400–2000 m², open plan)
    {"disused:leisure": ["fitness_centre", "gym", "sports_centre",
                         "bowling_alley", "dance", "amusement_arcade"]},
    {"abandoned:leisure": ["fitness_centre", "gym", "sports_centre",
                           "bowling_alley"]},
    # Disused food service & amenities
    {"disused:amenity": ["restaurant", "cafe", "fast_food", "bar", "pub",
                         "biergarten", "cinema", "theatre", "bank",
                         "pharmacy", "post_office", "doctors", "dentist",
                         "nightclub"]},
    {"abandoned:amenity": ["restaurant", "cafe", "fast_food", "bar",
                           "pub", "cinema", "theatre", "bank"]},
    # Disused offices
    {"disused:office": True},
    # Buildings tagged as retail/commercial/supermarket with NO active
    # occupant tag — likely vacant.
    # NOTE: this block uses way-only tags (building=*) — nodes rarely carry
    # building=* so the result is dominated by way elements; that is correct.
    {"building": ["retail", "commercial", "supermarket", "warehouse"]},
]

_ACTIVE_OCCUPANT_COLS = ("amenity", "office", "leisure", "craft", "tourism")

_FOOD_SERVICE_AMENITIES = frozenset({
    "restaurant", "cafe", "fast_food", "bar", "pub", "biergarten",
    "food_court", "ice_cream", "nightclub",
})
_FINANCIAL_AMENITIES = frozenset({"bank", "bureau_de_change", "money_transfer"})
_HEALTH_AMENITIES = frozenset({
    "pharmacy", "doctors", "dentist", "clinic", "veterinary", "optician",
})
_ENTERTAINMENT_AMENITIES = frozenset({"theatre", "cinema"})
_SUPERMARKET_SHOPS = frozenset({"supermarket", "convenience", "discount", "wholesale"})
_COMMERCIAL_BUILDINGS = frozenset({"retail", "commercial", "supermarket", "warehouse"})


def _empty_site_eligibility_gdf(category_col: str) -> gpd.GeoDataFrame:
    """Return an empty site-eligibility GeoDataFrame with required columns."""
    return gpd.GeoDataFrame(
        columns=["osm_id", "osm_type", "geometry", category_col],
        geometry="geometry",
        crs="EPSG:4326",
    )


def _records_to_eligibility_gdf(
    records: List[dict],
    category_col: str,
    city: str,
    layer: str,
) -> gpd.GeoDataFrame:
    """Assemble records into a GeoDataFrame or return an empty frame with warning."""
    if not records:
        logger.warning(
            "No usable %s elements found for '%s'.", layer, city
        )
        return _empty_site_eligibility_gdf(category_col)

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.drop_duplicates(subset=["osm_id"], keep="first")
    if gdf.empty:
        logger.warning(
            "No usable %s elements remain for '%s' after deduplication.",
            layer,
            city,
        )
        return _empty_site_eligibility_gdf(category_col)
    return gdf


def _filter_vacant_active_occupants(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop vacant-layer rows that carry an active ground-floor occupant tag."""
    if not any(c in gdf.columns for c in ("shop", *_ACTIVE_OCCUPANT_COLS)):
        return gdf

    active_mask = pd.Series(False, index=gdf.index)
    if "shop" in gdf.columns:
        active_mask |= gdf["shop"].notna() & (gdf["shop"] != "vacant")
    for col in _ACTIVE_OCCUPANT_COLS:
        if col in gdf.columns:
            active_mask |= gdf[col].notna()
    return gdf[~active_mask].copy()


def _blocker_category_from_row(row: pd.Series) -> str:
    shop = row.get("shop")
    if pd.notna(shop):
        if shop in _SUPERMARKET_SHOPS:
            return "supermarket"
        return "retail_other"

    amenity = row.get("amenity")
    if pd.notna(amenity):
        if amenity in _FOOD_SERVICE_AMENITIES:
            return "food_service"
        if amenity in _FINANCIAL_AMENITIES:
            return "financial"
        if amenity in _HEALTH_AMENITIES:
            return "health"
        if amenity in _ENTERTAINMENT_AMENITIES:
            return "entertainment_large"
        if amenity == "place_of_worship":
            return "religious"

    if pd.notna(row.get("leisure")):
        return "leisure"
    if pd.notna(row.get("office")):
        return "office"
    if pd.notna(row.get("craft")):
        return "craft"
    if pd.notna(row.get("tourism")):
        return "accommodation"
    return "other"


def _vacant_category_from_row(row: pd.Series) -> str:
    if row.get("shop") == "vacant":
        return "vacant_unit"

    for col in ("disused:shop", "abandoned:shop", "was:shop"):
        if pd.notna(row.get(col)):
            return "former_shop"

    for col in ("disused:leisure", "abandoned:leisure"):
        if pd.notna(row.get(col)):
            return "former_leisure"

    for col in ("disused:amenity", "abandoned:amenity"):
        if pd.notna(row.get(col)):
            return "former_amenity"

    if pd.notna(row.get("disused:office")):
        return "former_office"

    building = row.get("building")
    if pd.notna(building) and building in _COMMERCIAL_BUILDINGS:
        return "commercial_building"
    return "other"


def _assign_blocker_categories(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["blocker_category"] = gdf.apply(_blocker_category_from_row, axis=1)
    return gdf


def _assign_vacant_categories(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["vacant_category"] = gdf.apply(_vacant_category_from_row, axis=1)
    return gdf


def _fetch_eligibility_layer(
    *,
    city: str,
    area_id: int,
    tag_defs: List[Dict[str, object]],
    category_col: str,
    layer: str,
    timeout: int,
    post_filter: Optional[Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame]] = None,
) -> gpd.GeoDataFrame:
    """Run one Overpass query and return a processed eligibility GeoDataFrame."""
    tag_filter_blocks = [_build_tag_filters(t) for t in tag_defs]
    query = _build_overpass_query(area_id, tag_filter_blocks, timeout=timeout)
    logger.debug("Overpass query (%s):\n%s", layer, query)

    logger.info(
        "Fetching site-eligibility '%s' layer for '%s' from Overpass (timeout=%ds).",
        layer,
        city,
        timeout,
    )
    response = _post_with_retry(
        _OVERPASS_URL,
        query.encode("utf-8"),
        timeout=timeout + 60,
    )
    response.raise_for_status()

    elements: List[dict] = response.json().get("elements", [])
    logger.info(
        "Overpass returned %d raw elements for '%s' layer.", len(elements), layer
    )

    records = _parse_elements(elements)
    logger.info(
        "%d usable elements parsed for '%s' layer (nodes, closed ways, relations).",
        len(records),
        layer,
    )

    gdf = _records_to_eligibility_gdf(records, category_col, city, layer)
    if gdf.empty:
        return gdf

    if post_filter is not None:
        gdf = post_filter(gdf)
        if gdf.empty:
            logger.warning(
                "No usable %s elements remain for '%s' after post-filtering.",
                layer,
                city,
            )
            return _empty_site_eligibility_gdf(category_col)

    if category_col == "blocker_category":
        gdf = _assign_blocker_categories(gdf)
    else:
        gdf = _assign_vacant_categories(gdf)
    return gdf


# Columns written to the GeoPackage layers.
# Restricted to fields with safe names and known types.
# The colon-namespaced OSM tag columns (brand:wikidata, disused:shop, etc.)
# are intentionally excluded — they are used only for in-memory categorisation
# and their names or dtypes can cause GDAL write failures.
_GPKG_KEEP_BLOCKERS: frozenset[str] = frozenset({
    "osm_id", "osm_type", "geometry", "name",
    "shop", "amenity", "leisure", "office", "craft", "tourism", "building",
    "blocker_category",
})

_GPKG_KEEP_VACANT: frozenset[str] = frozenset({
    "osm_id", "osm_type", "geometry", "name",
    "shop", "amenity", "leisure", "office", "craft", "tourism", "building",
    "vacant_category",
})


def fetch_site_eligibility_signals(
    city: str = "Berlin",
    cache_dir: Optional[Path] = None,
    timeout: int = 180,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Fetch ground-floor commercial occupancy signals from OpenStreetMap.

    Runs two separate Overpass queries for *city* and returns two
    GeoDataFrames that together cover all locations where a new supermarket
    entry is physically and institutionally plausible:

    ``gdf_blockers``
        Currently occupied ground-floor commercial units: all retail shops
        (``shop=*``), food service, banks, gyms, offices, hotels, craft
        workshops, cinemas, and religious buildings.  These confirm that a
        commercially configured unit exists at this location, but the space
        is currently taken.

    ``gdf_vacant``
        Empty or formerly occupied commercial units: ``shop=vacant``,
        ``disused:shop=*``, ``abandoned:shop=*``, ``was:shop=*``, disused
        leisure/amenity/office elements, and ``building=retail|commercial|
        supermarket|warehouse`` with no active occupant tag.  These are the
        highest-priority candidate locations for market entry because the
        physical configuration is confirmed and the space is currently free.

    Both layers are written to a single GeoPackage at
    ``<cache_dir>/OSM_site_eligibility_{city}.gpkg`` for reproducibility.
    Subsequent calls load from cache without hitting the Overpass API.

    Parameters
    ----------
    city:
        Nominatim place name used to resolve the Overpass search area.
        Default: ``"Berlin"``.
    cache_dir:
        Directory for the GeoPackage cache file.  Created if absent.
        Defaults to ``<repo_root>/data/raw``.
    timeout:
        Overpass API query timeout in seconds.  Default: 180.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        ``(gdf_blockers, gdf_vacant)`` — both in CRS EPSG:4326.
        Each row includes ``osm_id``, ``osm_type``, ``geometry``, ``point``
        (Shapely Point, not a GeoSeries), and all OSM tag columns present in
        the data.  ``gdf_blockers`` additionally contains
        ``blocker_category``; ``gdf_vacant`` additionally contains
        ``vacant_category``.

    Notes
    -----
    The ``point`` column is not written to the GeoPackage (it is re-derived
    on every load).  The ``building=*`` elements in the vacant layer have
    active-occupant rows filtered out in post-processing (see source).
    """
    effective_cache_dir = (
        cache_dir if cache_dir is not None else _find_repo_root() / Path("data/raw")
    )
    output_path = effective_cache_dir / f"OSM_site_eligibility_{city}.gpkg"

    if output_path.exists():
        logger.info("Loading cached site-eligibility signals from %s.", output_path)
        gdf_blockers = gpd.read_file(output_path, layer="blockers")
        gdf_vacant = gpd.read_file(output_path, layer="vacant")
        return _add_point_column(gdf_blockers), _add_point_column(gdf_vacant)

    logger.info("Resolving Overpass area ID for '%s' via Nominatim.", city)
    area_id = _get_area_id(city)
    logger.info("Area ID for '%s': %d.", city, area_id)

    gdf_blockers = _fetch_eligibility_layer(
        city=city,
        area_id=area_id,
        tag_defs=_BLOCKER_TAGS,
        category_col="blocker_category",
        layer="blockers",
        timeout=timeout,
    )

    gdf_vacant = _fetch_eligibility_layer(
        city=city,
        area_id=area_id,
        tag_defs=_VACANT_TAGS,
        category_col="vacant_category",
        layer="vacant",
        timeout=timeout,
        post_filter=_filter_vacant_active_occupants,
    )

    effective_cache_dir.mkdir(parents=True, exist_ok=True)

    # Select only curated columns for the GPKG.  Colon-namespaced OSM tag
    # columns (e.g. brand:wikidata, disused:shop, addr:street) are excluded
    # because GDAL may reject their field definitions regardless of any name
    # sanitization.  Categories are already derived in memory so no information
    # required downstream is lost.
    _blockers_out_cols = [
        c for c in gdf_blockers.columns
        if c in _GPKG_KEEP_BLOCKERS
    ]
    _vacant_out_cols = [
        c for c in gdf_vacant.columns
        if c in _GPKG_KEEP_VACANT
    ]

    gdf_blockers[_blockers_out_cols].to_file(
        output_path, layer="blockers", driver="GPKG"
    )
    gdf_vacant[_vacant_out_cols].to_file(
        output_path, layer="vacant", driver="GPKG", mode="a"
    )

    gdf_blockers = _add_point_column(gdf_blockers)
    gdf_vacant = _add_point_column(gdf_vacant)

    logger.info(
        "Cached site-eligibility signals for '%s' to %s "
        "(%d blockers, %d vacant).",
        city,
        output_path,
        len(gdf_blockers),
        len(gdf_vacant),
    )

    return gdf_blockers, gdf_vacant
