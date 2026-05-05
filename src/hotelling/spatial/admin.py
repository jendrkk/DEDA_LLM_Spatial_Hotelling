"""Sub-city administrative units (e.g. Berlin LOR) download and normalisation.

Key dependencies: geopandas (optional ``[spatial]`` extra); ``py7zr`` for .7z archives.
"""
from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


def download_lor_shapes(if_old: bool = True) -> None:
    """Download Berlin LOR shapes from SenStadt, extract, reproject to EPSG:3035, save parquet."""
    logger.info("Starting LOR shapes download and conversion.")
    
    if if_old:
        url = (
            "https://www.berlin.de/sen/sbw/_assets/stadtdaten/stadtwissen/"
            "lebensweltlich-orientierte-raeume/lor_2019-01-01_shapefiles_nur_id.7z?ts=1770289260"
        )
        file_name = "lor_shapes_2019.7z"
    else:
        url = ("https://www.berlin.de/sen/sbw/_assets/stadtdaten/stadtwissen/"
               "lebensweltlich-orientierte-raeume/lor_2021-01-01_k3_shapefiles_nur_id.7z?ts=1770289259"
        )
        file_name = "lor_shapes_2021.7z"

    save_path = Path(f"data/raw/{file_name}")
    extract_dir = Path(f"data/raw/{file_name.split('.')[0]}")
    urllib.request.urlretrieve(url, save_path)
    logger.info("Downloaded LOR archive to %s.", save_path)
    try:
        import py7zr  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "Missing optional dependency 'py7zr' required to extract .7z archives. "
            "Install it with: pip install py7zr",
        ) from exc

    with py7zr.SevenZipFile(save_path, mode="r") as archive:
        archive.extractall(path=str(extract_dir))
    logger.info("Extracted LOR archive to %s.", extract_dir)
    os.remove(save_path)

    shapefiles = list(extract_dir.glob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(f"No shapefiles found in extracted LOR directory: {extract_dir}")

    priority_tokens = ["_PLR_", "_PGR_", "_BZR_"]
    selected = None
    for token in priority_tokens:
        selected = next((path for path in shapefiles if token in path.name), None)
        if selected is not None:
            break
    if selected is None:
        selected = shapefiles[0]

    logger.info("Selected LOR shapefile %s for conversion.", selected.name)
    data = gpd.read_file(selected)
    data = data.to_crs(crs="EPSG:3035")
    logger.info("Reprojected to CRS EPSG:3035.")

    parquet_path = Path(f"data/raw/{file_name.split('.')[0]}.parquet")
    data.to_parquet(parquet_path)
    logger.info("Saved LOR parquet to %s.", parquet_path)
    shutil.rmtree(extract_dir)
    logger.info("Removed extracted LOR folder %s after parquet conversion.", extract_dir)


def download_local_shapes() -> None:
    """Download local planning-area shapes for cities other than Berlin (placeholder)."""
    raise NotImplementedError("This method is not implemented yet.")

def equip_lor_with_population(
    lor: gpd.GeoDataFrame, population_grid: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Equip LOR with population."""
    
    if lor.crs is None or population_grid.crs is None:
        raise ValueError("Both 'lor' and 'population_grid' must have a defined CRS.")
    if lor.crs != population_grid.crs:
        raise ValueError(
            f"CRS mismatch: lor has {lor.crs}, population_grid has {population_grid.crs}. "
            "Reproject one to match the other before calling this function."
        )

    if "Einwohner" not in population_grid.columns:
        raise KeyError("population_grid must contain an 'Einwohner' column.")

    lor_with_population = lor.copy()
    population_points = population_grid.copy()
    population_points["geometry"] = population_points.geometry.centroid

    # Match each centroid to the LOR polygon that contains it.
    joined = gpd.sjoin(
        population_points[["Einwohner", "geometry"]],
        lor_with_population[["PLR_ID", "geometry"]],
        how="left",
        predicate="intersects",
    )

    population_sum = joined.groupby("index_right")["Einwohner"].sum()
    lor_with_population["Einwohner"] = (
        population_sum.reindex(lor_with_population.index, fill_value=0).astype(float)
    )
    return lor_with_population

def shapes_around_boundary(
    shapes: gpd.GeoDataFrame, boundary: gpd.GeoSeries, buffer_distance: float = 1000.0
) -> gpd.GeoDataFrame:
    """Return shapes that intersect with the boundary or are within a buffer distance."""
    buffered_boundary = boundary.buffer(buffer_distance)
    return shapes[shapes.intersects(buffered_boundary)]

def refine_shapes_selection(
    shapes: gpd.GeoDataFrame, boundary: gpd.GeoSeries, population_grid: gpd.GeoDataFrame,
    buffer_distance: float = 1000.0, extend_selection_by: int = 10
) -> gpd.GeoDataFrame:
    """Refine shape selection by including those intersecting with an extended buffer."""
    shapes = equip_lor_with_population(shapes, population_grid).copy()
    initial_selection = shapes_around_boundary(shapes, boundary, buffer_distance)
    if initial_selection.empty:
        logger.warning("No shapes found around the boundary with the initial buffer.")
        return initial_selection

    # Compute population per shape and population density
    shapes_with_population = shapes.copy()
    shapes_with_population["initially_selected"] = shapes_with_population['PLR_ID'].isin(initial_selection['PLR_ID'])
    
    # Compute the population density for each shape and normalize it to be in [0, 1]
    shapes_with_population["area"] = shapes_with_population["geometry"].area
    shapes_with_population["population_density"] = shapes_with_population["Einwohner"] / shapes_with_population["geometry"].area
    shapes_with_population["population_density_normalized"] = shapes_with_population["population_density"] / shapes_with_population["population_density"].max()
    shapes_with_population["population_density_normalized_remaining"] = shapes_with_population["population_density"] / shapes_with_population["population_density"][~shapes_with_population["initially_selected"]].max()
    
    # Compute centroid-to-boundary-edge distance (not polygon-to-polygon distance).
    if isinstance(boundary, gpd.GeoDataFrame):
        boundary_geom = boundary.geometry.unary_union
    elif isinstance(boundary, gpd.GeoSeries):
        boundary_geom = boundary.unary_union
    else:
        boundary_geom = boundary
    boundary_edge = boundary_geom.boundary
    shapes_with_population["distance_to_boundary"] = shapes_with_population["geometry"].centroid.distance(boundary_edge)
    shapes_with_population["distance_to_boundary_squared"] = shapes_with_population["distance_to_boundary"].pow(2)
    shapes_with_population["distance_to_boundary_normalized"] = shapes_with_population["distance_to_boundary"] / shapes_with_population["distance_to_boundary"].max()
    shapes_with_population["distance_to_boundary_normalized_remaining"] = shapes_with_population["distance_to_boundary"] / shapes_with_population["distance_to_boundary"][~shapes_with_population["initially_selected"]].max()
    shapes_with_population["distance_to_boundary_normalized_squared"] = shapes_with_population["distance_to_boundary_squared"] / shapes_with_population["distance_to_boundary_squared"].max()
    shapes_with_population["distance_to_boundary_normalized_squared_remaining"] = shapes_with_population["distance_to_boundary_squared"] / shapes_with_population["distance_to_boundary_squared"][~shapes_with_population["initially_selected"]].max()
    
    # Compute the population density normalized by the distance to the boundary
    shapes_with_population["population_distance_density"] = shapes_with_population["population_density_normalized"] * (1 - shapes_with_population["distance_to_boundary_normalized_squared"])
    shapes_with_population["population_distance_density_remaining"] = shapes_with_population["population_density_normalized_remaining"] * (1 - shapes_with_population["distance_to_boundary_normalized_squared_remaining"])
    
    shapes_with_population = shapes_with_population.sort_values("population_distance_density_remaining", ascending=False)

    # Greedy region-growing: at each step pick the highest-scoring shape that already
    # borders the current selection, add it, and expand the frontier.  Repeating this
    # guarantees the extended selection stays connected (monolithic, no holes).
    #
    # A 1 m buffer on the running union handles tiny floating-point edge gaps that can
    # appear between otherwise-adjacent LOR polygons in EPSG:3035.
    current_union = initial_selection.geometry.unary_union
    remaining = shapes_with_population[~shapes_with_population["initially_selected"]].copy()
    additional_ids: list[str] = []

    for _ in range(extend_selection_by):
        if remaining.empty:
            break

        candidates = remaining[remaining.geometry.intersects(current_union.buffer(1.0))]

        if candidates.empty:
            logger.warning(
                "No bordering shapes remain after adding %d additional shapes. Stopping early.",
                len(additional_ids),
            )
            break

        best_idx = candidates["population_distance_density_remaining"].idxmax()
        additional_ids.append(remaining.loc[best_idx, "PLR_ID"])
        current_union = current_union.union(remaining.loc[best_idx, "geometry"])
        remaining = remaining.drop(index=best_idx)

    shapes_with_population["additional_selected"] = shapes_with_population["PLR_ID"].isin(additional_ids)
    shapes_with_population["selected"] = shapes_with_population["initially_selected"] | shapes_with_population["additional_selected"]

    return shapes_with_population

def join_lor_names(if_old: bool = True):
    logger.info("Starting LOR names download and processing.")
    if if_old:
        link = "https://www.berlin.de/sen/sbw/_assets/stadtdaten/stadtwissen/lebensweltlich-orientierte-raeume/lor_2019-01-01_uebersicht_id_namen.xlsx?ts=1770289266"
        valid_year = 2019
    else:
        link = "https://www.berlin.de/sen/sbw/_assets/stadtdaten/stadtwissen/lebensweltlich-orientierte-raeume/lor_2021-01-01_k3_uebersicht_id_namen.xlsx?ts=1770289269"
        valid_year = 2021
    save_path = Path(f"data/raw/lor_names_{valid_year}.xlsx")
    urllib.request.urlretrieve(link, save_path)
    logger.info("LOR names downloaded.")
    
    # Read the sheet "LOR_2019_Übersicht"
    df = pd.read_excel(save_path, sheet_name=f"LOR_{valid_year}_Übersicht")
    
    # Make sure the PLR_ID column is of the same type as in the LOR shapes (e.g. string)
    df["PLR_ID"] = df['PLR_ID'].astype(str).str.zfill(8)
    
    # Load the LOR shapes
    lor_shapes = gpd.read_parquet(f"data/raw/lor_shapes_{valid_year}.parquet")
    
    # Assign the LOR names to the GeoDataFrame of LOR shapes with the PLR_ID column
    lor_shapes["PLR_NAME"] = np.nan
    for _, row in df.iterrows():
        lor_shapes.loc[lor_shapes["PLR_ID"] == row["PLR_ID"], "PLR_NAME"] = row["PLR_NAME"]
    
    logger.info("LOR names processed.")
    
    # Save the file to the parquet file in processed folder
    lor_shapes.to_parquet(f"data/processed/lor_{valid_year}.parquet")
    logger.info("LOR shapes with names saved to %s.", f"data/processed/lor_{valid_year}.parquet")