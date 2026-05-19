from ctypes import Array
import numpy as np
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