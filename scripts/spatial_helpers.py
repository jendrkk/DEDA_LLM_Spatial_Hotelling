from ctypes import Array
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import STRtree
from collections import defaultdict
from typing import Hashable


def area_coverage_fractions(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    left_id: str,
    right_id: str,
    min_fraction: float = 0.0,
) -> dict[Hashable, dict[Hashable, float]]:
    """
    For each geometry in `left`, compute the fraction of its area covered
    by each intersecting geometry in `right`.

    Parameters
    ----------
    left, right   : GeoDataFrames with matching CRS (must be equal-area, e.g. EPSG:3035)
    left_id       : column name used as key in the output dict
    right_id      : column name used as value-dict keys
    min_fraction  : drop pairs whose coverage fraction is below this threshold

    Returns
    -------
    dict[left_id_value, dict[right_id_value, float]]
        Fraction ∈ [0, 1] of left geometry area covered by right geometry.
        Left rows with no intersections are present with an empty dict.
    """
    if left.crs != right.crs:
        raise ValueError(f"CRS mismatch: {left.crs} vs {right.crs}")

    left_geoms  = left.geometry.to_numpy()
    right_geoms = right.geometry.to_numpy()
    left_keys   = left[left_id].to_numpy()
    right_keys  = right[right_id].to_numpy()

    # --- 1. Bulk spatial index query (pure C, very fast) ---
    tree = STRtree(right_geoms)
    l_idx, r_idx = tree.query(left_geoms, predicate="intersects")
    # l_idx, r_idx are parallel arrays of matching index pairs

    # --- 2. Vectorised intersection area (Shapely 2 bulk op) ---
    inter_geoms = shapely.intersection(left_geoms[l_idx], right_geoms[r_idx])
    inter_areas = shapely.area(inter_geoms)          # shape (n_pairs,)
    left_areas  = shapely.area(left_geoms[l_idx])    # shape (n_pairs,)

    with np.errstate(invalid="ignore", divide="ignore"):
        fractions = np.where(left_areas > 0, inter_areas / left_areas, 0.0)

    # --- 3. Apply threshold & build output dict ---
    mask = fractions > min_fraction
    l_idx, r_idx, fractions = l_idx[mask], r_idx[mask], fractions[mask]

    result: dict = {k: {} for k in left_keys}   # ensure every left row present
    for li, ri, frac in zip(l_idx, r_idx, fractions):
        result[left_keys[li]][right_keys[ri]] = float(frac)

    return result

def multi_sjoin(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    left_id: str,
    right_id: str,
    predicate: str = "intersects",
) -> dict[Hashable, Array[Hashable]]:
    """
    For each geometry in `left`, find all intersecting (or predicate) geometries in `right`
    and return the corresponding values from `right_id`.

    Parameters
    ----------
    left, right   : GeoDataFrames with matching CRS (must be equal-area, e.g. EPSG:3035)
    left_id       : column name used as key in the output dict
    right_id      : column name used as value-dict keys
    predicate     : spatial predicate to use for the query (default: "intersects")

    Returns
    -------
    dict[left_id_value, Array[right_id_value]]
        Array of right_id values for each left_id value.
    """
    if left.crs != right.crs:
        raise ValueError(f"CRS mismatch: {left.crs} vs {right.crs}")

    left_geoms  = left.geometry.to_numpy()
    right_geoms = right.geometry.to_numpy()
    left_keys   = left[left_id].to_numpy()
    right_keys  = right[right_id].to_numpy()

    # --- 1. Bulk spatial index query (pure C, very fast) ---
    tree = STRtree(right_geoms)
    l_idx, r_idx = tree.query(left_geoms, predicate=predicate)
    # l_idx, r_idx are parallel arrays of matching index pairs

    # --- 2. Build output dict ---
    result: dict = {k: [] for k in left_keys}
    for li, ri in zip(l_idx, r_idx):
        result[left_keys[li]].append(right_keys[ri])
    return result

"""
Addition to scripts/spatial_helpers.py
─────────────────────────────────────────────────────────────────────────────
Paste this block at the end of spatial_helpers.py
(after the existing multi_sjoin function).

Provides compute_ground_floor_availability(), the area-aware replacement for
the crude n_blockers count used in GEO_06.
"""

# ── Node blocker estimated footprints (m²) ────────────────────────────────────
# Conservative lower-bound estimates per blocker_category.
# Lower = more inclusive (fewer false-positive "blocked" rulings).
# Calibrated to approximate the smallest typical occupant of that category
# in Berlin inner-ring urban commercial buildings.
NODE_AREA_ESTIMATES_M2: dict[str, float] = {
    "supermarket":         800.0,  # safety net — should be removed upstream
    "retail_other":        120.0,  # Rossmann / Budni / small shop
    "food_service":         80.0,  # restaurant / cafe / Imbiss
    "financial":           150.0,  # bank branch ground floor
    "health":               80.0,  # pharmacy / Arztpraxis
    "leisure":             400.0,  # gym / Fitnessstudio
    "office":              200.0,  # ground-floor office unit
    "craft":               150.0,  # Handwerksbetrieb
    "accommodation":       400.0,  # hotel lobby / Pension
    "entertainment_large": 600.0,  # cinema / theatre
    "religious":           300.0,  # Gebetsraum in commercial building
    "other":               100.0,  # conservative fallback
}


def compute_ground_floor_availability(
    buildings: gpd.GeoDataFrame,
    blockers: gpd.GeoDataFrame,
    building_id_col: str = "id",
    blocker_cat_col: str = "blocker_category",
    min_supermarket_area_m2: float = 400.0,
    analysis_crs: str = "EPSG:3035",
    node_area_estimates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute ground-floor availability for each building in *buildings*.

    Decision rule
    -------------
    For each building with footprint area A (m²):

        poly_occupied  = Σ area(intersection(building, poly_blocker_i))
        node_occupied  = Σ estimated_area(node_blocker_j.blocker_category)
        total_occupied = min(poly_occupied + node_occupied, A)
        available_m2   = A - total_occupied
        is_blocked     = available_m2 < min_supermarket_area_m2

    This is size-adaptive: a 2 000 m² building can absorb several tenants and
    still have room for a supermarket; a 600 m² building cannot.

    Parameters
    ----------
    buildings:
        GeoDataFrame with building polygon geometries and a unique ID column.
    blockers:
        OSM ground-floor blocker GeoDataFrame with mixed geometry types
        (Points from nodes, Polygons from ways) and a category column.
    building_id_col:
        Column in *buildings* used as the key (default ``"id"``).
    blocker_cat_col:
        Column in *blockers* giving the occupancy category
        (default ``"blocker_category"``).
    min_supermarket_area_m2:
        Minimum viable supermarket footprint in m².  Buildings where
        ``available_m2 < this`` are flagged ``is_blocked=True``.
        Default 400 m².
    analysis_crs:
        Equal-area CRS for area calculations (default ``"EPSG:3035"``).
        Both *buildings* and *blockers* are reprojected internally;
        original CRS is not modified.
    node_area_estimates:
        Override the module-level NODE_AREA_ESTIMATES_M2 dict.

    Returns
    -------
    pandas.DataFrame
        One row per building.  Columns:

        * ``{building_id_col}``    — building identifier (matches *buildings*)
        * ``footprint_m2``         — building polygon area in m²
        * ``poly_occupied_m2``     — area covered by polygon (way) blockers
        * ``node_occupied_m2``     — estimated area from node blockers
        * ``n_poly_blockers``      — count of polygon blockers intersecting
        * ``n_node_blockers``      — count of node blockers inside
        * ``total_occupied_m2``    — min(poly + node, footprint)
        * ``available_m2``         — footprint - total_occupied
        * ``is_blocked``           — True when available_m2 < threshold

    Notes
    -----
    All area calculations are performed in *analysis_crs* (EPSG:3035 by
    default), which is an ETRS89-LAEA equal-area projection.  ALKIS buildings
    delivered in EPSG:25833 have <0.1 % area distortion at Berlin's latitude,
    but equal-area is cleaner.

    The function never modifies *buildings* or *blockers* in place.
    """
    node_areas = node_area_estimates or NODE_AREA_ESTIMATES_M2

    # ── 1. Reproject to equal-area CRS for accurate m² calculations ──────────
    bld = buildings[[building_id_col, "geometry"]].to_crs(analysis_crs).copy()
    blk = blockers[["geometry", blocker_cat_col]].to_crs(analysis_crs).copy()

    bld_geoms  = bld.geometry.to_numpy()
    bld_ids    = bld[building_id_col].to_numpy()
    bld_areas  = shapely.area(bld_geoms)          # vectorised, fast

    # ── 2. Split blockers by geometry type ───────────────────────────────────
    is_poly = blk.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).to_numpy()
    is_node = blk.geometry.geom_type == "Point"

    poly_blk       = blk[is_poly].reset_index(drop=True)
    node_blk       = blk[is_node].reset_index(drop=True)
    poly_blk_geoms = poly_blk.geometry.to_numpy()
    node_blk_geoms = node_blk.geometry.to_numpy()
    node_blk_cats  = node_blk[blocker_cat_col].to_numpy()

    # ── 3. Polygon blocker coverage (exact intersection area) ─────────────────
    poly_occupied  = np.zeros(len(bld), dtype=float)
    n_poly_blocked = np.zeros(len(bld), dtype=int)

    if len(poly_blk) > 0:
        tree = STRtree(bld_geoms)
        # p_idx → poly_blk index; b_idx → buildings index
        p_idx, b_idx = tree.query(poly_blk_geoms, predicate="intersects")

        if len(p_idx) > 0:
            inter_geoms = shapely.intersection(poly_blk_geoms[p_idx], bld_geoms[b_idx])
            inter_areas = shapely.area(inter_geoms)

            for bi, area in zip(b_idx, inter_areas):
                poly_occupied[bi]  += area
                n_poly_blocked[bi] += 1

    # ── 4. Node blocker estimated area ────────────────────────────────────────
    node_occupied  = np.zeros(len(bld), dtype=float)
    n_node_blocked = np.zeros(len(bld), dtype=int)

    if len(node_blk) > 0:
        tree = STRtree(bld_geoms)
        n_idx, b_idx = tree.query(node_blk_geoms, predicate="intersects")

        for ni, bi in zip(n_idx, b_idx):
            cat  = node_blk_cats[ni]
            est  = node_areas.get(str(cat) if not isinstance(cat, str) else cat,
                                  node_areas["other"])
            node_occupied[bi]  += est
            n_node_blocked[bi] += 1

    # ── 5. Combine and decide ─────────────────────────────────────────────────
    # Cap total at building footprint area (data-quality safeguard)
    total_occupied = np.minimum(poly_occupied + node_occupied, bld_areas)
    available      = bld_areas - total_occupied
    is_blocked     = available < min_supermarket_area_m2

    return pd.DataFrame({
        building_id_col:    bld_ids,
        "footprint_m2":     bld_areas,
        "poly_occupied_m2": poly_occupied,
        "node_occupied_m2": node_occupied,
        "n_poly_blockers":  n_poly_blocked,
        "n_node_blockers":  n_node_blocked,
        "total_occupied_m2": total_occupied,
        "available_m2":     available,
        "is_blocked":       is_blocked,
    })