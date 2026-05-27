"""Zensus 2022 100 m grid: download, load, clip to boundary, and full INSPIRE grid merge.

Key dependencies: geopandas, pandas, pyproj, requests, shapely (optional ``[spatial]`` extra).

References:
    Statistisches Bundesamt Zensus 2022;
    Global Human Settlement Layer (GHS-POP) — planned fallback API.
"""
from __future__ import annotations

import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from hotelling.spatial.boundaries import load_boundary

logger = logging.getLogger(__name__)

__all__ = [
    "download_zensus_2022",
    "load_zensus_2022",
    "load_ghs_pop_fallback",
    "filter_zensus_2022",
    "build_full_grid",
    "build_grid_polygons",
    "clip_grid_to_boundary",
    "make_cell_id",
]


def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first existing column name from a prioritized candidate list."""
    normalized_map = {str(col).strip().casefold(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().casefold()
        if key in normalized_map:
            return normalized_map[key]
    raise KeyError(f"None of the candidate columns found: {candidates}")


def download_zensus_2022() -> None:
    """Download the Zensus 2022 100 m population grid from the Destatis portal and save parquet."""
    logger.info("Starting Zensus 2022 download and conversion.")
    link = "https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Bevoelkerungszahl.zip"
    save_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "zensus2022_grid.zip"
    extract_dir = Path(__file__).resolve().parents[3] / "data" / "raw" / "zensus2022_grid"
    urllib.request.urlretrieve(link, save_path)
    logger.info("Downloaded Zensus archive to %s.", save_path)
    with zipfile.ZipFile(save_path, "r") as zip_ref:
        zip_ref.extractall(str(extract_dir))
    logger.info("Extracted Zensus archive to %s.", extract_dir)
    os.remove(save_path)

    csv_files = list(extract_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in extracted Zensus folder: {extract_dir}")

    selected_csv = next(
        (
            path
            for path in csv_files
            if "Bevoelkerungszahl_100m-Gitter" in path.name and "Zensus2022" in path.name
        ),
        csv_files[0],
    )
    logger.info("Selected Zensus CSV %s for conversion.", selected_csv.name)

    data = pd.read_csv(selected_csv, sep=";")

    x_col = _find_first_existing_column(data, ["x_mp_100m", "x_mp", "x"])
    y_col = _find_first_existing_column(data, ["y_mp_100m", "y_mp", "y"])

    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data[x_col], data[y_col]),
        crs="EPSG:3035",
    )

    save_path_parquet = Path(__file__).resolve().parents[3] / "data" / "raw" / "zensus2022_grid.parquet"
    gdf.to_parquet(save_path_parquet)
    logger.info("Saved processed Zensus population parquet to %s.", save_path_parquet)
    shutil.rmtree(extract_dir)
    logger.info("Removed extracted Zensus folder %s after parquet conversion.", extract_dir)


def load_zensus_2022() -> gpd.GeoDataFrame:
    """Load Zensus 2022 100 m population points from parquet (EPSG:3035)."""
    logger.info("Loading Zensus parquet from %s.", Path(__file__).resolve().parents[3] / "data" / "raw" / "zensus2022_grid.parquet")
    return gpd.read_parquet(Path("data/raw/zensus2022_grid.parquet")).to_crs("EPSG:3035")


def load_ghs_pop_fallback() -> gpd.GeoDataFrame:
    """Load Global Human Settlement Layer population as a fallback raster (not yet implemented)."""
    raise NotImplementedError


def filter_zensus_2022(boundary_path: Path) -> None:
    """Filter Zensus 2022 population grid to a city boundary and write filtered parquet."""
    zensus_path = Path("data/raw/zensus2022_grid.parquet")
    zensus = gpd.read_parquet(zensus_path)

    boundary = load_boundary(boundary_path)
    boundary_geom = boundary.geometry.iloc[0]
    zensus = zensus[zensus.geometry.within(boundary_geom)]
    logger.info("Filtered Zensus 2022 population grid to city boundary.")
    logger.info("Filtered %s population grid rows.", len(zensus))
    parquet_path = Path("data/raw/zensus2022_grid_filtered.parquet")
    zensus.to_parquet(parquet_path)
    logger.info("Saved filtered Zensus 2022 population grid to %s.", parquet_path)


def _infer_grid_offsets(series_x: pd.Series, series_y: pd.Series, step: int) -> tuple[int, int]:
    """Return (x_mod, y_mod) such that official grid coordinates satisfy coord ≡ mod (mod step)."""
    sx = (series_x.astype(np.int64) % step).mode()
    sy = (series_y.astype(np.int64) % step).mode()
    x_mod = int(sx.iloc[0]) if len(sx) else 0
    y_mod = int(sy.iloc[0]) if len(sy) else 0
    return x_mod, y_mod


def _aligned_center_range(lo: float, hi: float, step: int, mod: int) -> np.ndarray:
    """Sequence of lattice coordinates between ``lo`` and ``hi`` inclusive, congruent to ``mod`` mod ``step``."""
    if hi < lo:
        return np.array([], dtype=np.int64)
    k0 = int(np.ceil((lo - mod) / step))
    k1 = int(np.floor((hi - mod) / step))
    if k1 < k0:
        return np.array([], dtype=np.int64)
    return (mod + step * np.arange(k0, k1 + 1, dtype=np.int64)).astype(np.int64)


def build_full_grid(
    boundary: gpd.GeoDataFrame,
    zensus: gpd.GeoDataFrame,
    cell_size: float = 100.0,
) -> gpd.GeoDataFrame:
    """Return full INSPIRE 100 m grid inside ``boundary``, with 0 for unpopulated cells.

    Lattice alignment (offsets modulo ``cell_size``) is taken from ``zensus``, not from
    ``boundary`` bounds alone, so merge keys match Destatis grid coordinates (e.g. cell
    centres vs corners).
    """
    if zensus.empty:
        raise ValueError("build_full_grid requires non-empty zensus to infer grid alignment.")

    step = int(round(cell_size))
    if step <= 0:
        raise ValueError("cell_size must be positive.")

    zensus = zensus.copy()
    if zensus.crs != boundary.crs:
        boundary = boundary.to_crs(zensus.crs)

    x_col = _find_first_existing_column(zensus, ["x_mp_100m", "x_mp", "x"])
    y_col = _find_first_existing_column(zensus, ["y_mp_100m", "y_mp", "y"])
    zensus[x_col] = zensus[x_col].astype(np.int64)
    zensus[y_col] = zensus[y_col].astype(np.int64)

    x_mod, y_mod = _infer_grid_offsets(zensus[x_col], zensus[y_col], step)

    minx, miny, maxx, maxy = boundary.total_bounds
    xs = _aligned_center_range(minx, maxx, step, x_mod)
    ys = _aligned_center_range(miny, maxy, step, y_mod)
    if xs.size == 0 or ys.size == 0:
        logger.warning("Aligned grid range empty for boundary bounds; falling back to zensus extent.")
        zx1, zx2 = int(zensus[x_col].min()), int(zensus[x_col].max())
        zy1, zy2 = int(zensus[y_col].min()), int(zensus[y_col].max())
        xs = _aligned_center_range(min(zx1, minx), max(zx2, maxx), step, x_mod)
        ys = _aligned_center_range(min(zy1, miny), max(zy2, maxy), step, y_mod)

    xx, yy = np.meshgrid(xs, ys)
    skeleton = gpd.GeoDataFrame(
        {x_col: xx.ravel(), y_col: yy.ravel()},
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs=zensus.crs,
    )

    boundary_union = boundary.geometry.unary_union
    skeleton = skeleton[skeleton.geometry.within(boundary_union)].copy()
    logger.info("Full grid has %s lattice cells inside boundary (aligned to Zensus).", len(skeleton))

    z_attrs = zensus.drop(columns=["geometry"])
    if x_col != "x_mp_100m":
        z_attrs = z_attrs.rename(columns={x_col: "x_mp_100m"})
    if y_col != "y_mp_100m":
        z_attrs = z_attrs.rename(columns={y_col: "y_mp_100m"})
    sk = skeleton.rename(columns={x_col: "x_mp_100m", y_col: "y_mp_100m"})

    full_gdf = sk.merge(z_attrs, on=["x_mp_100m", "y_mp_100m"], how="left")

    if "Einwohner" not in full_gdf.columns:
        raise KeyError("Zensus data must include an 'Einwohner' column for population.")

    full_gdf["Einwohner"] = full_gdf["Einwohner"].fillna(0).astype(np.int32)
    logger.info(
        "Populated: %s cells, Empty: %s cells.",
        (full_gdf["Einwohner"] > 0).sum(),
        (full_gdf["Einwohner"] == 0).sum(),
    )
    return full_gdf


def build_grid_polygons(
    zensus: gpd.GeoDataFrame,
    cell_size: float = 100.0,
    boundary: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Convert Zensus midpoint geometries to square cell polygons.

    Each point at ``(x_mp_100m, y_mp_100m)`` becomes a ``cell_size × cell_size``
    metre square centred on that midpoint.  If *boundary* is supplied, polygons
    are clipped to its union and degenerate results (points, empty) are dropped.

    Parameters
    ----------
    zensus:
        GeoDataFrame with point geometry in EPSG:3035 and columns
        ``x_mp_100m``, ``y_mp_100m``.
    cell_size:
        Edge length in metres (default 100).
    boundary:
        Optional GeoDataFrame in EPSG:3035 to clip cell polygons against.

    Returns
    -------
    geopandas.GeoDataFrame
        Same schema as *zensus* but with square ``Polygon`` geometries.
    """
    if zensus.empty:
        return zensus.copy()

    out = zensus.copy()
    x_col = _find_first_existing_column(out, ["x_mp_100m", "x_mp", "x"])
    y_col = _find_first_existing_column(out, ["y_mp_100m", "y_mp", "y"])
    half = float(cell_size) / 2.0
    xs = out[x_col].astype(float)
    ys = out[y_col].astype(float)
    out["geometry"] = [box(x - half, y - half, x + half, y + half) for x, y in zip(xs, ys)]

    if boundary is not None:
        b = boundary
        if out.crs is not None and b.crs is not None and out.crs != b.crs:
            b = b.to_crs(out.crs)
        clip_geom = b.geometry.unary_union
        out["geometry"] = out.geometry.intersection(clip_geom)
        keep = ~out.geometry.is_empty & out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        out = out.loc[keep].copy()

    return out


def clip_grid_to_boundary(
    grid: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Clip grid cell polygons to an exact boundary polygon.

    Edge cells that extend beyond *boundary* are trimmed; degenerate results
    (empty geometry, collapsed to point/line) are dropped.

    Parameters
    ----------
    grid:
        GeoDataFrame with polygon geometry, same CRS as *boundary*.
    boundary:
        GeoDataFrame whose union defines the clipping envelope.

    Returns
    -------
    geopandas.GeoDataFrame
        Clipped grid, same CRS as input.
    """
    out = grid.copy()
    b = boundary
    if out.crs is not None and b.crs is not None and out.crs != b.crs:
        b = b.to_crs(out.crs)
    clip_geom = b.geometry.unary_union
    out["geometry"] = out.geometry.intersection(clip_geom)
    keep = ~out.geometry.is_empty & out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    return out.loc[keep].copy()


def make_cell_id(row: pd.Series) -> str:
    """Return the canonical INSPIRE 100 m cell identifier for a grid row.

    Uses the official ``GITTER_ID_100m`` value when present and non-None;
    otherwise synthesises it from the metric EPSG:3035 midpoint coordinates
    following the INSPIRE naming convention::

        CRS3035RES100mN{y_mp_100m}E{x_mp_100m}

    This function is the single authoritative source of cell IDs in the
    hotelling package and replaces the ad-hoc ``gitter_id`` / ``name_grid``
    helpers used in GEO_05 and GEO_07 notebooks.

    Parameters
    ----------
    row:
        A row from a Zensus / population-grid GeoDataFrame.  Must contain
        either ``GITTER_ID_100m`` or both ``x_mp_100m`` and ``y_mp_100m``.

    Returns
    -------
    str
        The cell identifier string.

    Examples
    --------
    >>> import pandas as pd
    >>> row = pd.Series({"GITTER_ID_100m": None, "x_mp_100m": 4493500, "y_mp_100m": 3279200})
    >>> make_cell_id(row)
    'CRS3035RES100mN3279200E4493500'
    """
    gid = row.get("GITTER_ID_100m")
    if gid is not None and not (isinstance(gid, float) and pd.isna(gid)):
        return str(gid)
    return f"CRS3035RES100mN{int(row['y_mp_100m'])}E{int(row['x_mp_100m'])}"