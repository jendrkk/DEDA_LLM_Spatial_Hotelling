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
from typing import Optional


def load_zensus_2022(
    path: Path,
    clip_polygon=None,  # shapely.geometry.Polygon
    epsg: int = 25833,
) -> "xr.DataArray":  # type: ignore[name-defined]
    """Load Zensus 2022 100m population raster, optionally clipped to a polygon.

    Parameters
    ----------
    path : path to GeoTIFF file
    clip_polygon : shapely Polygon in EPSG:epsg for clipping
    epsg : coordinate reference system EPSG code

    Returns
    -------
    xarray.DataArray with dims (y, x), CRS metadata in attrs
    """
    raise NotImplementedError


def load_ghs_pop_fallback(
    path: Path,
    clip_polygon=None,
    epsg: int = 25833,
) -> "xr.DataArray":  # type: ignore[name-defined]
    """Load GHS-POP raster as fallback if Zensus data is unavailable.

    Returns
    -------
    xarray.DataArray resampled to 100m resolution
    """
    raise NotImplementedError
