"""Commercial entry-site candidate layer (L_commercial_final).

This module identifies disused or former large-format retail buildings inside
the inner-Ringbahn study area that are plausible entry locations for the
simulation's LLM entrant.

Logic extracted from GEO_06_city_parcels.ipynb.

Pipeline executed by :func:`build_commercial_candidates`:

1. Fetch OSM disused large-format retail via
   :func:`~hotelling.spatial.osm.fetch_pois` (type="disused_retail").
2. Resolve OSM shop signal (disused:shop, abandoned:shop, was:shop, building).
3. Clip to inner-Ringbahn boundary.
4. Join each feature to its ALKIS building footprint (reliable area, polygon).
5. Filter: footprint ≥ 400 m², MBR min-dimension ≥ 10 m, aspect ratio ≤ 10:1.
6. Remove locations within 50 m of an active incumbent supermarket.
7. Enrich with BRW land value (€/m²) and Stadtstruktur morphology score.
8. Assemble and save L_commercial_final.

Outputs
-------
``data/processed/L_commercial_final.parquet``
``data/processed/L_commercial_final.gpkg``

Key dependencies: geopandas, shapely, numpy, pandas.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

logger = logging.getLogger(__name__)

__all__ = ["build_commercial_candidates"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER_A_TYPES: list[str] = ["supermarket", "wholesale", "hypermarket"]
TIER_B_TYPES: list[str] = [
    "chemist", "department_store", "doityourself", "hardware",
    "electronics", "furniture", "variety_store",
]
ALL_LARGE_TYPES: list[str] = TIER_A_TYPES + TIER_B_TYPES

# Stadtstruktur morphology scoring tables.
# Source: Berlin Umweltatlas UA_Stadtstruktur layer attribute names
# ``typ_klar`` (detailed type) and ``ststrname`` (structural name).
_SS_SCORE_TYP_KLAR: dict[str, float] = {
    "Kerngebiet": 0.95,
    "Dichte Blockbebauung, geschlossener Hinterhof (1870er - 1918), 5 - 6-geschossig": 0.95,
    "Geschlossene Blockbebauung, Hinterhof (1870er - 1918), 5-geschossig": 0.93,
    "Geschlossene und halboffene Blockbebauung, Schmuck- und Gartenhof (1870er - 1918), 4-geschossig": 0.87,
    "Gewerbe- und Industriegebiet, großflächiger Einzelhandel, dichte Bebauung": 0.88,
    "Gewerbe- und Industriegebiet, großflächiger Einzelhandel, geringe Bebauung": 0.80,
    "Mischgebiet ohne Wohngebietscharakter, dichte Bebauung": 0.80,
    "Mischgebiet ohne Wohngebietscharakter, geringe Bebauung": 0.65,
    "Blockrandbebauung mit Großhöfen (1920er - 1940er), 2 - 5-geschossig": 0.80,
    "Entkernte Blockrandbebauung, Lückenschluss nach 1945": 0.72,
    "Bahnhof und Bahnanlagen ohne Gleiskörper": 0.82,
    "Heterogene, innerstädtische Mischbebauung, Lückenschluss nach 1945": 0.65,
    "Mischbebauung, halboffener und offener Schuppenhof, 2 - 4-geschossig": 0.68,
    "Parallele Zeilenbebauung mit architektonischem Zeilengrün (1920er - 1930er), 2 - 5-geschossig": 0.52,
    "Geschosswohnungsbau der 1990er Jahre und jünger": 0.38,
    "Freie Zeilenbebauung mit landschaftlichem Siedlungsgrün (1950er - 1970er), 2 - 6-geschossig": 0.32,
    "Großsiedlung und Punkthochhäuser (1960er - 1990er), 4 - 11-geschossig und mehr": 0.20,
    "Brachfläche": 0.40,
    "Verwaltung": 0.50,
}

_SS_SCORE_STSTRNAME: dict[str, float] = {
    "Blockbebauung der Gründerzeit mit Seitenflügeln und Hinterhäusern": 0.95,
    "Blockrandbebauung der Gründerzeit mit geringem Anteil von Seiten- und Hintergebäuden": 0.88,
    "Blockrandbebauung der Gründerzeit mit massiven Veränderungen": 0.76,
    "Bebauung mit überwiegender Nutzung durch Handel und Dienstleistung": 0.92,
    "Blockrand- und Zeilenbebauung der 1920er und 1930er Jahre": 0.72,
    "Dichte Bebauung mit überwiegender Nutzung durch Gewerbe und Industrie": 0.68,
    "Geringe Bebauung mit überwiegender Nutzung durch Gewerbe und Industrie": 0.60,
    "Siedlungsbebauung der 1990er Jahre und jünger": 0.36,
    "Zeilenbebauung seit den 1950er Jahren": 0.33,
    "Hohe Bebauung der Nachkriegszeit": 0.22,
}

_DEFAULT_MORPH: float = 0.50

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_shop_signal(row: pd.Series) -> tuple[str, str]:
    """Return (shop_type, shop_signal) from OSM tag columns of one feature.

    Checks the three disused-tag prefixes in priority order, then falls back
    to the structural ``building=supermarket`` tag.

    Parameters
    ----------
    row:
        A single row of the raw OSM GeoDataFrame returned by
        :func:`~hotelling.spatial.osm.fetch_pois`.

    Returns
    -------
    tuple[str, str]
        ``(shop_type, shop_signal)`` — e.g. ``("supermarket", "disused:shop")``.
        Returns ``("unknown", "unknown")`` when no signal is found.
    """
    for prefix in ("disused:shop", "abandoned:shop", "was:shop"):
        val = row.get(prefix)
        if pd.notna(val) and str(val) in ALL_LARGE_TYPES:
            return str(val), prefix
    if str(row.get("building", "")) == "supermarket":
        return "supermarket", "building"
    if str(row.get("shop", "")) == "vacant":
        return "vacant", "shop"
    return "unknown", "unknown"


def _mbr_sides(geom: shapely.Geometry) -> tuple[float, float]:
    """Return (min_side_m, max_side_m) of the minimum-area bounding rectangle.

    Used for the MBR dimension and aspect-ratio filters.

    Parameters
    ----------
    geom:
        Any Shapely geometry (typically a Polygon or MultiPolygon).

    Returns
    -------
    tuple[float, float]
        ``(min_dim, max_dim)`` in metres.  Returns ``(0.0, 0.0)`` on error.
    """
    try:
        mbr = shapely.minimum_rotated_rectangle(geom)
        coords = np.array(mbr.exterior.coords)
        sides = np.array([
            np.linalg.norm(coords[1] - coords[0]),
            np.linalg.norm(coords[2] - coords[1]),
            np.linalg.norm(coords[3] - coords[2]),
            np.linalg.norm(coords[0] - coords[3]),
        ])
        return float(sides.min()), float(sides.max())
    except Exception:
        return 0.0, 0.0


def _morphology_score(typ_klar: object, ststrname: object) -> float:
    """Return a morphology score ∈ [0, 1] from Stadtstruktur attributes.

    Checks ``typ_klar`` first (more fine-grained), then falls back to
    ``ststrname``.  Returns ``_DEFAULT_MORPH`` when neither key is found.

    Parameters
    ----------
    typ_klar:
        Value of the ``typ_klar`` column in the Stadtstruktur layer.
    ststrname:
        Value of the ``ststrname`` column in the Stadtstruktur layer.

    Returns
    -------
    float
        Score in [0, 1].  Higher = more commercially favourable morphology.
    """
    for val, table in [(typ_klar, _SS_SCORE_TYP_KLAR), (ststrname, _SS_SCORE_STSTRNAME)]:
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            score = table.get(str(val).strip())
            if score is not None:
                return score
    return _DEFAULT_MORPH


def _find_best_alkis(
    geom: shapely.Geometry,
    osm_type: str,
    alkis_tree: STRtree,
    alkis_geoms: np.ndarray,
    alkis_ring: gpd.GeoDataFrame,
) -> Optional[int]:
    """Return the ``_alkis_idx`` of the best-matching ALKIS building or None.

    Strategy:
    - **Point (OSM node)**: take the containing building first; fall back to
      nearest within 30 m.
    - **Polygon (OSM way/relation)**: take the ALKIS building with the largest
      intersection area; reject if intersection < 10 % of the smaller polygon.

    Parameters
    ----------
    geom:
        OSM feature geometry (in EPSG:3035).
    osm_type:
        ``"node"``, ``"way"``, or ``"relation"``.
    alkis_tree:
        STRtree built from ``alkis_ring.geometry.values``.
    alkis_geoms:
        Numpy array of ALKIS geometry objects (same order as alkis_ring rows).
    alkis_ring:
        ALKIS buildings pre-clipped to the study area, with columns
        ``_alkis_area`` and ``_alkis_idx``.

    Returns
    -------
    int | None
        Integer positional index into ``alkis_ring`` of the best match,
        or ``None`` if no acceptable match is found.
    """
    buf = geom.buffer(30.0) if geom.geom_type == "Point" else geom.buffer(1.0)
    candidates = alkis_tree.query(buf, predicate="intersects")
    if len(candidates) == 0:
        return None

    if geom.geom_type == "Point":
        containing = [i for i in candidates if alkis_geoms[i].contains(geom)]
        if containing:
            return max(containing, key=lambda i: alkis_ring.loc[i, "_alkis_area"])
        return min(candidates, key=lambda i: alkis_geoms[i].distance(geom))
    else:
        def _iarea(i: int) -> float:
            try:
                return geom.intersection(alkis_geoms[i]).area
            except Exception:
                return 0.0

        best = max(candidates, key=_iarea)
        min_area = min(geom.area, alkis_ring.loc[best, "_alkis_area"]) * 0.10
        return best if _iarea(best) >= min_area else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_commercial_candidates(
    boundary: gpd.GeoDataFrame,
    alkis_path: Path,
    incumbents: gpd.GeoDataFrame,
    brw: gpd.GeoDataFrame,
    stadtstruktur: gpd.GeoDataFrame,
    *,
    min_footprint_m2: float = 400.0,
    min_mbr_dim_m: float = 10.0,
    max_mbr_aspect: float = 10.0,
    incumbent_exclusion_buffer_m: float = 50.0,
    save_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """Build the L_commercial_final candidate location catalogue.

    Runs the full GEO_06 pipeline: OSM fetch → ALKIS join → size filter →
    incumbent exclusion → BRW + morphology enrichment → save.

    The OSM disused-retail query is performed once and cached by
    :func:`~hotelling.spatial.osm.fetch_pois` in ``data/raw``.

    Parameters
    ----------
    boundary:
        Inner-Ringbahn boundary GeoDataFrame in EPSG:3035 (or any metric CRS;
        reprojected internally to ANALYSIS_CRS = EPSG:3035).
    alkis_path:
        Path to ``alkis_full.gpkg``.  Must contain a layer ``gebaeudeflaechen``
        with a ``bezeich`` column (to filter ``AX_Gebaeude`` records).
    incumbents:
        Active incumbent supermarkets GeoDataFrame (e.g. from
        :func:`~hotelling.spatial.osm.fetch_pois` with type="supermarket").
        Used to exclude occupied locations.
    brw:
        Bodenrichtwerte land-value GeoDataFrame (EPSG:3035 or any CRS;
        reprojected internally).  The first numeric column with median > 100
        is used as the BRW value.
    stadtstruktur:
        UA_Stadtstruktur layer GeoDataFrame (EPSG:3035).  Must contain
        columns ``typ_klar`` and/or ``ststrname`` for morphology scoring.
    min_footprint_m2:
        Minimum building footprint area in m².  Default 400.
    min_mbr_dim_m:
        Minimum MBR short-side dimension in metres.  Default 10.
    max_mbr_aspect:
        Maximum MBR aspect ratio (long / short).  Default 10.
    incumbent_exclusion_buffer_m:
        Radius in metres around an incumbent within which candidates are
        excluded.  Default 50.
    save_path:
        Directory where outputs are saved.  If ``None``, defaults to
        ``data/processed/``.  Two files are written:
        ``L_commercial_final.parquet`` and ``L_commercial_final.gpkg``.

    Returns
    -------
    geopandas.GeoDataFrame
        Candidate locations with columns:
        ``id``, ``geometry``, ``osm_type``, ``name``, ``shop_type``,
        ``shop_signal``, ``footprint_m2``, ``mbr_min_dim_m``, ``gfk``,
        ``brw_value``, ``typ_klar``, ``ststrname``, ``morphology_score``,
        ``confidence_tier``.
        CRS: EPSG:3035.
    """
    from hotelling.spatial.osm import fetch_pois  # noqa: PLC0415

    ANALYSIS_CRS = "EPSG:3035"

    # ── 1. OSM fetch (cached) ─────────────────────────────────────────────────
    _TAGS: list[dict] = [
        {"disused:shop":   ALL_LARGE_TYPES},
        {"abandoned:shop": ALL_LARGE_TYPES},
        {"was:shop":       ALL_LARGE_TYPES},
        {"building": "supermarket"},
    ]
    logger.info("Fetching disused large-format retail from Overpass (cached after first run).")
    osm_raw = fetch_pois(
        type="disused_retail",
        city="Berlin",
        tags=_TAGS,
        name="disused_retail",
    )
    logger.info("Fetched %d features (nodes + ways + relations).", len(osm_raw))

    # ── 2. Assign shop signal and tier ────────────────────────────────────────
    _signals = osm_raw.apply(_resolve_shop_signal, axis=1)
    osm_raw = osm_raw.copy()
    osm_raw["shop_type"]   = _signals.map(lambda t: t[0])
    osm_raw["shop_signal"] = _signals.map(lambda t: t[1])
    osm_raw["tier"] = osm_raw["shop_type"].map(
        lambda t: "A" if t in TIER_A_TYPES else "B"
    )

    # ── 3. Reproject and clip to inner-Ringbahn boundary ─────────────────────
    boundary_proj = boundary.to_crs(ANALYSIS_CRS) if boundary.crs.to_epsg() != 3035 else boundary.copy()
    osm_3035 = osm_raw.to_crs(ANALYSIS_CRS).copy()
    boundary_union = boundary_proj.geometry.union_all()
    if boundary_union.geom_type not in ("Polygon", "MultiPolygon"):
        logger.warning(
            "boundary.union_all() yielded %s instead of Polygon/MultiPolygon "
            "(boundary may contain Point/LineString geometry). "
            "Falling back to convex hull — pass a polygon boundary for exact clipping.",
            type(boundary_union).__name__,
        )
        boundary_union = boundary_union.convex_hull

    _centroid_in = osm_3035.geometry.centroid.within(boundary_union)
    _poly_in     = osm_3035.geometry.intersects(boundary_union)
    osm_3035 = osm_3035[_centroid_in | _poly_in].copy().reset_index(drop=True)
    logger.info(
        "After clip to inner Ring: %d features (from %d Berlin-wide).",
        len(osm_3035), len(osm_raw),
    )

    # ── 4. Load ALKIS buildings (pre-clipped to Ring) ─────────────────────────
    logger.info("Loading ALKIS buildings from %s.", alkis_path)
    alkis_bld = gpd.read_file(alkis_path, layer="gebaeudeflaechen")
    alkis_bld = alkis_bld[alkis_bld["bezeich"] == "AX_Gebaeude"].copy()
    alkis_bld = alkis_bld.to_crs(ANALYSIS_CRS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alkis_ring = gpd.clip(alkis_bld, boundary_union).copy().reset_index(drop=True)
    alkis_ring["_alkis_area"] = alkis_ring.geometry.area
    alkis_ring["_alkis_idx"]  = alkis_ring.index
    logger.info("ALKIS buildings in Ring: %d.", len(alkis_ring))

    # ── 5. ALKIS building join ────────────────────────────────────────────────
    alkis_geoms = alkis_ring.geometry.values
    alkis_tree  = STRtree(alkis_geoms)

    logger.info("Joining %d OSM features to ALKIS buildings.", len(osm_3035))
    _alkis_idx_list: list[Optional[int]] = [
        _find_best_alkis(row.geometry, row.osm_type, alkis_tree, alkis_geoms, alkis_ring)
        for _, row in osm_3035.iterrows()
    ]
    osm_3035["_alkis_idx"]   = _alkis_idx_list
    osm_3035["_has_alkis"]   = osm_3035["_alkis_idx"].notna()

    _matched = osm_3035["_has_alkis"]
    osm_3035.loc[_matched, "_alkis_area_m2"] = (
        osm_3035.loc[_matched, "_alkis_idx"].map(
            lambda i: alkis_ring.loc[int(i), "_alkis_area"]
        )
    )
    osm_3035.loc[_matched, "_alkis_geom"] = (
        osm_3035.loc[_matched, "_alkis_idx"].map(
            lambda i: alkis_ring.loc[int(i), "geometry"]
        )
    )
    osm_3035.loc[_matched, "_alkis_gfk"] = (
        osm_3035.loc[_matched, "_alkis_idx"].map(
            lambda i: alkis_ring.loc[int(i), "gfk"] if "gfk" in alkis_ring.columns else None
        )
    )

    _way_mask = osm_3035["osm_type"].isin(["way", "relation"])
    _no_alkis = ~_matched
    osm_3035.loc[_way_mask & _no_alkis, "_alkis_area_m2"] = (
        osm_3035.loc[_way_mask & _no_alkis, "geometry"].area
    )
    osm_3035.loc[_way_mask & _no_alkis, "_alkis_geom"] = (
        osm_3035.loc[_way_mask & _no_alkis, "geometry"]
    )

    n_matched = _matched.sum()
    n_way_unmatched = (_way_mask & _no_alkis).sum()
    n_node_unmatched = (
        (osm_3035["osm_type"] == "node") & _no_alkis
    ).sum()
    logger.info(
        "ALKIS join: %d matched, %d way/rel unmatched (OSM area used), "
        "%d nodes unmatched (will be dropped).",
        n_matched, n_way_unmatched, n_node_unmatched,
    )

    # ── 6. Size filter ────────────────────────────────────────────────────────
    _cands = osm_3035[osm_3035["_alkis_area_m2"].notna()].copy()
    logger.info("After dropping no-area features: %d", len(_cands))
    _cands = _cands[_cands["_alkis_area_m2"] >= min_footprint_m2].copy()
    logger.info("After footprint ≥ %.0f m² filter: %d", min_footprint_m2, len(_cands))

    _bld_geoms = _cands["_alkis_geom"].where(
        _cands["_alkis_geom"].notna(), _cands["geometry"]
    )
    _sides = [_mbr_sides(g) for g in _bld_geoms]
    _cands["_min_dim"] = [s[0] for s in _sides]
    _cands["_max_dim"] = [s[1] for s in _sides]
    _cands["_aspect"]  = np.where(
        _cands["_min_dim"] > 0.01,
        _cands["_max_dim"] / _cands["_min_dim"],
        np.inf,
    )
    _before = len(_cands)
    _cands = _cands[
        (_cands["_min_dim"] >= min_mbr_dim_m) & (_cands["_aspect"] <= max_mbr_aspect)
    ].copy()
    logger.info(
        "After MBR filter (min %.0f m, aspect ≤ %.0f:1): removed %d, %d remain.",
        min_mbr_dim_m, max_mbr_aspect, _before - len(_cands), len(_cands),
    )

    # ── 7. Remove active incumbents ───────────────────────────────────────────
    _inc_pts = incumbents.to_crs(ANALYSIS_CRS).geometry.centroid
    _inc_tree = STRtree(_inc_pts.values)

    _cand_geoms_for_excl = _cands["_alkis_geom"].where(
        _cands["_alkis_geom"].notna(), _cands["geometry"]
    )

    def _near_incumbent(geom: shapely.Geometry) -> bool:
        buf = geom.centroid.buffer(incumbent_exclusion_buffer_m)
        return len(_inc_tree.query(buf, predicate="intersects")) > 0

    _incumbent_flag = [_near_incumbent(g) for g in _cand_geoms_for_excl]
    _before = len(_cands)
    _cands = _cands[~pd.Series(_incumbent_flag, index=_cands.index)].copy()
    logger.info(
        "After removing active incumbent locations: removed %d, %d remain.",
        _before - len(_cands), len(_cands),
    )

    # ── 8. BRW land-value enrichment ──────────────────────────────────────────
    _brw_proj = brw.to_crs(ANALYSIS_CRS) if brw.crs.to_epsg() != 3035 else brw.copy()
    _BRW_COL = next(
        (
            c for c in _brw_proj.columns
            if c not in ("geometry", "fid", "index")
            and pd.api.types.is_numeric_dtype(_brw_proj[c])
            and _brw_proj[c].median() > 100
        ),
        None,
    )

    if _BRW_COL:
        _brw_slim = _brw_proj[["geometry", _BRW_COL]].copy().reset_index(drop=True)
        _cand_cents = _cands.copy()
        _cand_geoms_brw = _cands["_alkis_geom"].where(
            _cands["_alkis_geom"].notna(), _cands["geometry"]
        )
        _cand_cents["geometry"] = _cand_geoms_brw.apply(lambda g: g.centroid)
        _cand_cents = _cand_cents.set_geometry("geometry")
        _brw_sj = gpd.sjoin(
            _cand_cents[["geometry"]].reset_index(names="_cand_idx"),
            _brw_slim[["geometry", _BRW_COL]],
            how="left",
            predicate="within",
        )
        _brw_map = (
            _brw_sj.dropna(subset=[_BRW_COL])
            .groupby("_cand_idx")[_BRW_COL]
            .first()
            .to_dict()
        )
        _cands["brw_value"] = _cands.index.map(_brw_map)
        logger.info(
            "BRW joined: %d/%d matched  (range %.0f–%.0f €/m²).",
            _cands["brw_value"].notna().sum(), len(_cands),
            _cands["brw_value"].min(), _cands["brw_value"].max(),
        )
    else:
        _cands["brw_value"] = np.nan
        logger.warning("BRW value column not detected — brw_value = NaN.")

    # ── 9. Stadtstruktur morphology enrichment ────────────────────────────────
    _ss_proj = stadtstruktur.to_crs(ANALYSIS_CRS) if stadtstruktur.crs.to_epsg() != 3035 else stadtstruktur.copy()
    _ss_cols = [c for c in ["geometry", "typ_klar", "ststrname"] if c in _ss_proj.columns]

    _cand_cents_ss = _cands.copy()
    _cand_geoms_ss = _cands["_alkis_geom"].where(
        _cands["_alkis_geom"].notna(), _cands["geometry"]
    )
    _cand_cents_ss["geometry"] = _cand_geoms_ss.apply(lambda g: g.centroid)
    _cand_cents_ss = _cand_cents_ss.set_geometry("geometry")

    _ss_sj = gpd.sjoin(
        _cand_cents_ss[["geometry"]].reset_index(names="_cand_idx"),
        _ss_proj[_ss_cols],
        how="left",
        predicate="within",
    )
    _ss_lu = _ss_sj.drop_duplicates("_cand_idx").set_index("_cand_idx")

    def _safe_get_ss(idx: int, col: str) -> object:
        try:
            return _ss_lu.loc[idx, col] if col in _ss_lu.columns else None
        except KeyError:
            return None

    _cands["typ_klar"]  = _cands.index.map(lambda i: _safe_get_ss(i, "typ_klar"))
    _cands["ststrname"] = _cands.index.map(lambda i: _safe_get_ss(i, "ststrname"))
    _cands["morphology_score"] = _cands.apply(
        lambda r: _morphology_score(r.get("typ_klar"), r.get("ststrname")), axis=1
    )
    logger.info(
        "Stadtstruktur joined: %d/%d matched.",
        _cands["typ_klar"].notna().sum(), len(_cands),
    )

    # ── 10. Assemble L_commercial_final ───────────────────────────────────────
    def _canonical_geom(row: pd.Series) -> shapely.Geometry:
        g = row.get("_alkis_geom")
        if g is not None and not (isinstance(g, float) and np.isnan(g)):
            return g
        return row["geometry"]

    _final_geoms = [_canonical_geom(r) for _, r in _cands.iterrows()]

    L_final = gpd.GeoDataFrame(
        {
            "id":               _cands["osm_id"].astype(str),
            "geometry":         _final_geoms,
            "osm_type":         _cands["osm_type"],
            "name":             _cands["name"].values if "name" in _cands.columns else None,
            "shop_type":        _cands["shop_type"],
            "shop_signal":      _cands["shop_signal"],
            "footprint_m2":     _cands["_alkis_area_m2"].round(1),
            "mbr_min_dim_m":    _cands["_min_dim"].round(1),
            "gfk":              _cands.get("_alkis_gfk", pd.Series(dtype=object)).values,
            "brw_value":        _cands["brw_value"],
            "typ_klar":         _cands["typ_klar"],
            "ststrname":        _cands["ststrname"],
            "morphology_score": _cands["morphology_score"].round(3),
            "confidence_tier":  _cands["tier"],
        },
        geometry="geometry",
        crs=ANALYSIS_CRS,
    )

    logger.info(
        "L_commercial_final: %d candidate locations  "
        "(Tier A=%d, Tier B=%d).",
        len(L_final),
        (L_final["confidence_tier"] == "A").sum(),
        (L_final["confidence_tier"] == "B").sum(),
    )

    # ── 11. Save ──────────────────────────────────────────────────────────────
    _save_dir = save_path or Path("data/processed")
    _save_dir = Path(_save_dir)
    _save_dir.mkdir(parents=True, exist_ok=True)

    _pq = _save_dir / "L_commercial_final.parquet"
    _gk = _save_dir / "L_commercial_final.gpkg"

    L_final.to_parquet(_pq, index=False)
    L_final.to_file(_gk, driver="GPKG", layer="L_commercial_final")
    logger.info("Saved → %s  (%.1f MB)", _pq, _pq.stat().st_size / 1e6)
    logger.info("Saved → %s", _gk)

    return L_final
