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
    """Assemble the final simulation-ready grid from all spatial layers.

    Calls :func:`add_lor_attributes` and :func:`add_poi_layer` in sequence on
    *pop_grid*, then ensures the output schema is consistent for consumption
    by the simulation engine.

    The output GeoDataFrame is guaranteed to have the following columns:

    ``x_mp_100m``, ``y_mp_100m``
        INSPIRE grid coordinates (int64, EPSG:3035).
    ``geometry``
        100 m × 100 m square Polygon (EPSG:3035).
    ``Einwohner``
        Population count per cell (int32, 0 for uninhabited cells).
    ``PLR_ID``, ``PLR_NAME``
        LOR district identifier and name (str).
    ``poi_count``
        Number of supermarket POIs in cell (int, 0 if none).

    Parameters
    ----------
    pop_grid:
        Population grid from :func:`~hotelling.spatial.census.build_full_grid`,
        with polygon geometry (from :func:`~hotelling.spatial.census.build_grid_polygons`).
    lor:
        Selected LOR districts from :func:`~hotelling.spatial.admin.select_ringbahn_lor`.
    pois:
        OSM POI GeoDataFrame from :func:`~hotelling.spatial.osm.fetch_pois`.

    Returns
    -------
    geopandas.GeoDataFrame
        Simulation-ready grid in EPSG:3035.
    """
    raise NotImplementedError(
        "assemble_simulation_grid: call add_lor_attributes(pop_grid, lor), "
        "then add_poi_layer(result, pois), verify required columns exist, return."
    )
