"""Final simulation-grid assembly: merge all spatial layers into one GeoDataFrame.

This module provides functions that combine the population grid, LOR attributes,
POI data, and socio-economic layers into the single GeoDataFrame consumed by the
simulation engine.

All inputs must be in EPSG:3035.

Key dependencies: geopandas, pandas (optional ``[spatial]`` extra).
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "add_lor_attributes",
    "add_poi_layer",
    "assemble_simulation_grid",
]


def add_lor_attributes(
    grid: gpd.GeoDataFrame,
    lor: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Attach LOR planning-area attributes to each grid cell.

    Performs a spatial join of *grid* cell centroids against *lor* polygons
    and attaches ``PLR_ID``, ``PLR_NAME``, and any additional LOR columns
    (e.g. population-density scores from :func:`~hotelling.spatial.admin.refine_shapes_selection`)
    to *grid*.

    Parameters
    ----------
    grid:
        Population grid GeoDataFrame with polygon geometry in EPSG:3035.
    lor:
        LOR GeoDataFrame in EPSG:3035 with at minimum ``PLR_ID`` and
        ``PLR_NAME`` columns.

    Returns
    -------
    geopandas.GeoDataFrame
        *grid* with ``PLR_ID`` and ``PLR_NAME`` columns added.  Cells that
        do not fall within any LOR polygon receive ``NaN``.
    """
    raise NotImplementedError(
        "add_lor_attributes: compute grid.geometry.centroid, sjoin to lor "
        "with predicate='within', attach PLR_ID and PLR_NAME back to grid."
    )


def add_poi_layer(
    grid: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    chain_col: str = "chain",
) -> gpd.GeoDataFrame:
    """Count and classify POIs (supermarkets) per grid cell.

    Reprojects *pois* to match *grid* CRS, uses the ``point`` column for
    location (falling back to geometry centroid if ``point`` is absent),
    spatial-joins to *grid* polygons, and aggregates per cell.

    Expected output columns added to *grid*:

    * ``poi_count``          — total number of POIs in cell (int)
    * ``poi_chains``         — comma-separated list of chain names (str)
    * one column per chain   — boolean flag ``has_{chain_name}`` (e.g.
                               ``has_Rewe``, ``has_Lidl``, …)

    Parameters
    ----------
    grid:
        Population grid GeoDataFrame with polygon geometry in EPSG:3035.
    pois:
        OSM POI GeoDataFrame from :func:`~hotelling.spatial.osm.fetch_pois`,
        CRS EPSG:4326 (will be reprojected automatically).
    chain_col:
        Column in *pois* that holds the canonical chain name.

    Returns
    -------
    geopandas.GeoDataFrame
        *grid* with POI summary columns added.
    """
    raise NotImplementedError(
        "add_poi_layer: reproject pois to grid.crs, use pois['point'] if available "
        "else geometry.centroid, sjoin to grid, groupby grid index, count and list chains, "
        "create has_{chain} boolean columns, merge back."
    )


def assemble_simulation_grid(
    pop_grid: gpd.GeoDataFrame,
    lor: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Validate schema and finalise the simulation-ready grid.

    This function receives a *pop_grid* that has already been enriched by
    the pipeline (LOR attributes, ESIx/MSS scores, IHK employment, transit
    hub flags, CBD flags, and POI layer have all been joined in prior phases).
    Its job is to:

    1. Verify that all required columns are present, raising ``KeyError``
       with a clear message for any that are missing.
    2. Fill guaranteed-present columns with safe defaults where values are
       ``NaN`` (e.g. ``poi_count`` → 0, ``Einwohner`` → 0).
    3. Reset the index to a clean RangeIndex.
    4. Return the final GeoDataFrame ready for the simulation engine.

    Required output columns (will raise ``KeyError`` if absent):
        ``x_mp_100m``, ``y_mp_100m``, ``geometry``, ``Einwohner``,
        ``PLR_ID``, ``PLR_NAME``, ``poi_count``

    Parameters
    ----------
    pop_grid:
        Fully-enriched grid GeoDataFrame produced by all prior pipeline
        phases.  Must already contain the required columns listed above.
    lor:
        Selected LOR districts (passed for reference / logging only;
        not used for spatial joins here).
    pois:
        OSM POI GeoDataFrame (passed for reference / logging only;
        not used for spatial joins here).

    Returns
    -------
    geopandas.GeoDataFrame
        Schema-validated simulation grid in EPSG:3035, clean RangeIndex.

    Raises
    ------
    KeyError
        If any required column is missing from *pop_grid*.
    """
    raise NotImplementedError(
        "assemble_simulation_grid: validate that required columns exist in pop_grid "
        "(x_mp_100m, y_mp_100m, geometry, Einwohner, PLR_ID, PLR_NAME, poi_count), "
        "fill NaN values with safe defaults (poi_count→0, Einwohner→0), "
        "reset_index(drop=True), and return pop_grid. "
        "Do NOT re-run add_lor_attributes or add_poi_layer — those are done upstream."
    )
