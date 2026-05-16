"""Sub-city administrative units (e.g. Berlin LOR) download and normalisation.

Key dependencies: geopandas (optional ``[spatial]`` extra); ``py7zr`` for .7z archives.
"""
from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Sequence

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box as shapely_box

logger = logging.getLogger(__name__)

__all__ = [
    "download_lor_shapes",
    "download_local_shapes",
    "equip_lor_with_population",
    "find_optimal_rectangle",
    "join_lor_names",
    "load_lor",
    "refine_shapes_selection",
    "select_ringbahn_lor",
    "shapes_around_boundary",
]


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

def find_optimal_rectangle(
    boundary: gpd.GeoSeries | gpd.GeoDataFrame,
    population_grid: gpd.GeoDataFrame,
    buffer_distance: float = 0.0,
    cell_size: float = 100.0,
    augument_rectangle_by_additional_layers: Sequence[int] = (0, 0, 0, 0),
    max_iterations: int = 10000,
    tolerance: float = 0.01,
) -> gpd.GeoDataFrame:
    """Find the axis-aligned rectangle that optimally encloses a boundary.

    The returned rectangle satisfies:

    * The boundary (optionally expanded by *buffer_distance*) lies **completely
      within** the rectangle.
    * The centre of the rectangle equals the centroid of the **original**
      (pre-buffer) boundary.
    * Width and height are integer multiples of *cell_size*, so that the INSPIRE
      population-grid lattice tiles perfectly without partial edge cells.
    * Among all valid rectangles, the one with the **highest average population
      density** (``population / area``) is chosen.  This naturally balances the
      competing goals of maximising population coverage and minimising area.

    After the optimal rectangle is found, it can be asymmetrically extended via
    *augument_rectangle_by_additional_layers* = ``[top, right, bottom, left]``.

    Parameters
    ----------
    boundary:
        GeoSeries **or** GeoDataFrame that defines the study-area boundary.
        Reprojected internally to EPSG:3035 (metric CRS).
    population_grid:
        GeoDataFrame with **point** geometry in EPSG:3035 and an ``Einwohner``
        column (integer resident counts per 100 m cell).
    buffer_distance:
        Distance in metres by which the boundary is expanded before the
        rectangle is fitted.  The centre is still taken from the original
        (pre-buffer) boundary.  Default ``0.0``.
    cell_size:
        Side length of one grid cell in metres.  Both dimensions of the
        returned rectangle are guaranteed to be integer multiples of this
        value.  Default ``100.0``.
    augument_rectangle_by_additional_layers:
        ``[top, right, bottom, left]`` – number of extra *cell_size*-wide
        layers appended to each side of the optimal rectangle **after**
        optimisation.  Default ``(0, 0, 0, 0)``.
    max_iterations:
        Controls the size of the search grid: ``n_search_cols * n_search_rows
        ≤ max_iterations``.  Both search dimensions are set to
        ``floor(sqrt(max_iterations))``.  Default ``10_000``.
    tolerance:
        Minimum **relative** improvement in population density required to
        prefer a larger rectangle over the current best candidate.  A value
        of ``0.01`` means the challenger must be at least 1 % denser than the
        incumbent.  This bias towards smaller rectangles acts as a soft
        regulariser.  Default ``0.01``.

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame (EPSG:3035) whose ``geometry`` is the final
        rectangle (after augmentation).  Additional attribute columns:

        * ``n_cols``, ``n_rows``      – number of grid cells in each direction
        * ``width_m``, ``height_m``   – rectangle dimensions in metres
        * ``center_x``, ``center_y``  – centroid coordinates (EPSG:3035)
        * ``population``              – total ``Einwohner`` within rectangle
        * ``population_density``      – ``population / area`` [residents / m²]

    Raises
    ------
    ValueError
        If *population_grid* is empty, *cell_size* ≤ 0, or
        *augument_rectangle_by_additional_layers* does not have exactly 4
        elements.
    KeyError
        If *population_grid* has no ``Einwohner`` column.
    """
    _CRS = "EPSG:3035"

    # ── Input validation ──────────────────────────────────────────────────────
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}.")
    if population_grid.empty:
        raise ValueError("population_grid must not be empty.")
    if "Einwohner" not in population_grid.columns:
        raise KeyError("population_grid must contain an 'Einwohner' column.")
    aug = list(augument_rectangle_by_additional_layers)
    if len(aug) != 4:
        raise ValueError(
            f"augument_rectangle_by_additional_layers must have exactly 4 elements "
            f"[top, right, bottom, left], got {len(aug)}."
        )

    # ── 1. Normalise CRS ──────────────────────────────────────────────────────
    if isinstance(boundary, gpd.GeoDataFrame):
        boundary = boundary.geometry
    boundary = boundary.to_crs(_CRS)
    if population_grid.crs is None or population_grid.crs.to_epsg() != 3035:
        population_grid = population_grid.to_crs(_CRS)

    # ── 2. Derive containment envelope (optionally buffered) ──────────────────
    boundary_geom = boundary.unary_union
    containment_geom = (
        boundary_geom.buffer(buffer_distance) if buffer_distance > 0.0 else boundary_geom
    )

    # ── 3. Fixed centre = centroid of ORIGINAL (pre-buffer) boundary ──────────
    centroid = boundary_geom.centroid
    cx: float = centroid.x
    cy: float = centroid.y

    # ── 4. Minimum rectangle dimensions that enclose the containment envelope ─
    # Because the rectangle is centred at (cx, cy), each half-dimension must
    # cover the farther of the two opposing extremes of the envelope.
    bminx, bminy, bmaxx, bmaxy = containment_geom.bounds
    hw_min: float = max(cx - bminx, bmaxx - cx)
    hh_min: float = max(cy - bminy, bmaxy - cy)

    # Round up so that full_width = n_cols * cell_size and full_height = n_rows * cell_size.
    # half_width  = n_cols * cell_size / 2  ≥ hw_min  →  n_cols ≥ 2 * hw_min / cell_size
    n_cols_min: int = max(1, int(np.ceil(2.0 * hw_min / cell_size)))
    n_rows_min: int = max(1, int(np.ceil(2.0 * hh_min / cell_size)))

    logger.info(
        "Minimum enclosing rectangle: %d cols × %d rows "
        "(%.0f m × %.0f m), centre (%.1f, %.1f).",
        n_cols_min, n_rows_min,
        n_cols_min * cell_size, n_rows_min * cell_size,
        cx, cy,
    )

    # ── 5. Flat numpy arrays for fast population queries ──────────────────────
    pop_x: np.ndarray = population_grid.geometry.x.to_numpy(dtype=np.float64)
    pop_y: np.ndarray = population_grid.geometry.y.to_numpy(dtype=np.float64)
    pop_val: np.ndarray = population_grid["Einwohner"].to_numpy(dtype=np.float64)

    # ── 6. Grid search over (n_cols, n_rows) ──────────────────────────────────
    # We try rectangles of size (n_cols * cell_size) × (n_rows * cell_size) for
    # n_cols ∈ [n_cols_min, n_cols_min + search_side) and similarly for n_rows.
    # The search is structured so that total evaluations ≤ max_iterations.
    search_side: int = max(1, int(np.floor(np.sqrt(max_iterations))))

    best_score: float = -np.inf
    best_n_cols: int = n_cols_min
    best_n_rows: int = n_rows_min

    for n_cols in range(n_cols_min, n_cols_min + search_side):
        hw = n_cols * cell_size * 0.5
        # Boolean mask for the x-strip of this column count
        x_mask: np.ndarray = (pop_x >= cx - hw) & (pop_x <= cx + hw)
        sub_y: np.ndarray = pop_y[x_mask]
        sub_val: np.ndarray = pop_val[x_mask]

        if sub_val.size == 0:
            # Zero population in this x-band; score = 0 for all n_rows here.
            # The current best (≥ 0) cannot be beaten; skip the inner loop.
            continue

        for n_rows in range(n_rows_min, n_rows_min + search_side):
            hh = n_rows * cell_size * 0.5
            y_mask: np.ndarray = (sub_y >= cy - hh) & (sub_y <= cy + hh)
            total_pop: float = sub_val[y_mask].sum()

            area: float = float(n_cols) * float(n_rows) * cell_size * cell_size
            score: float = total_pop / area  # population density [residents / m²]

            # Update only if the challenger is strictly better by more than
            # `tolerance`, so that a marginally denser but much larger rectangle
            # does not win over a compact one (bias towards small area).
            threshold = best_score * (1.0 + tolerance) if best_score > 0.0 else best_score
            if score > threshold:
                best_score = score
                best_n_cols = n_cols
                best_n_rows = n_rows

    logger.info(
        "Optimal rectangle (before augmentation): %d cols × %d rows, "
        "density = %.4e residents/m².",
        best_n_cols, best_n_rows, best_score,
    )

    # ── 7. Apply asymmetric augmentation ──────────────────────────────────────
    aug_top, aug_right, aug_bottom, aug_left = aug

    # Optimal rectangle edges (centred at (cx, cy))
    opt_hw = best_n_cols * cell_size * 0.5
    opt_hh = best_n_rows * cell_size * 0.5

    # Extend each edge independently
    final_minx: float = cx - opt_hw - aug_left   * cell_size
    final_maxx: float = cx + opt_hw + aug_right  * cell_size
    final_miny: float = cy - opt_hh - aug_bottom * cell_size
    final_maxy: float = cy + opt_hh + aug_top    * cell_size

    final_n_cols: int = best_n_cols + aug_left + aug_right
    final_n_rows: int = best_n_rows + aug_top  + aug_bottom

    # ── 8. Final population count inside the augmented rectangle ─────────────
    final_mask: np.ndarray = (
        (pop_x >= final_minx) & (pop_x <= final_maxx)
        & (pop_y >= final_miny) & (pop_y <= final_maxy)
    )
    final_pop: int = int(pop_val[final_mask].sum())
    final_area: float = (final_maxx - final_minx) * (final_maxy - final_miny)
    final_density: float = final_pop / final_area if final_area > 0 else 0.0

    logger.info(
        "Final rectangle (after augmentation): %d cols × %d rows "
        "(%.0f m × %.0f m), population = %d, density = %.4e residents/m².",
        final_n_cols, final_n_rows,
        final_maxx - final_minx, final_maxy - final_miny,
        final_pop, final_density,
    )

    # ── 9. Build and return GeoDataFrame ──────────────────────────────────────
    rect_geom = shapely_box(final_minx, final_miny, final_maxx, final_maxy)
    return gpd.GeoDataFrame(
        {
            "n_cols":              [final_n_cols],
            "n_rows":              [final_n_rows],
            "width_m":             [float(final_maxx - final_minx)],
            "height_m":            [float(final_maxy - final_miny)],
            "center_x":            [cx],
            "center_y":            [cy],
            "population":          [final_pop],
            "population_density":  [final_density],
        },
        geometry=[rect_geom],
        crs=_CRS,
    )

def join_lor_names(if_old: bool = True):
    logger.info("Starting LOR names download and processing.")
    if if_old:
        link = "https://www.berlin.de/sen/sbw/_assets/stadtdaten/stadtwissen/lebensweltlich-orientierte-raeume/lor_2019-01-01_uebersicht_id_namen.xlsx?ts=1770289266"
        valid_year = 2019
    else:
        link = "https://www.berlin.de/sen/sbw/_assets/stadtdaten/stadtwissen/lebensweltlich-orientierte-raeume/lor_2021-01-01_k3_uebersicht_id_namen.xlsx?ts=1770289269"
        valid_year = 2021
    save_path = Path(__file__).resolve().parents[3] / "data" / "raw" / f"lor_names_{valid_year}.xlsx"
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


def load_lor(year: int = 2021) -> gpd.GeoDataFrame:
    """Load the processed LOR shapes for *year* and save a canonical ``lor.parquet``.

    Reads ``data/processed/lor_{year}.parquet`` (written by :func:`join_lor_names`),
    copies it to ``data/processed/lor.parquet`` as the canonical file used by
    notebooks and downstream pipeline steps, and returns the GeoDataFrame.

    Parameters
    ----------
    year:
        LOR version year: 2019 (legacy) or 2021 (current, default).

    Returns
    -------
    geopandas.GeoDataFrame
        LOR planning-area polygons in EPSG:3035 with ``PLR_ID`` and ``PLR_NAME``
        columns.

    Raises
    ------
    FileNotFoundError
        If ``data/processed/lor_{year}.parquet`` does not exist (i.e.
        :func:`join_lor_names` has not been run yet).
    """
    src = Path(f"data/processed/lor_{year}.parquet")
    if not src.is_file():
        raise FileNotFoundError(
            f"LOR parquet not found: {src}. Run join_lor_names for year {year} first.",
        )
    gdf = gpd.read_parquet(src)
    canonical = Path("data/processed/lor.parquet")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, canonical)
    logger.info("Canonical LOR copy saved to %s.", canonical)
    return gdf


def select_ringbahn_lor(
    lor: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    population_grid: gpd.GeoDataFrame,
    buffer_distance: float = 500.0,
    extend_selection_by: int = 6,
) -> gpd.GeoDataFrame:
    """Select LOR districts that cover the inner-Ringbahn study area.

    Convenience wrapper around :func:`refine_shapes_selection` that runs the
    full selection algorithm and returns only the rows flagged as ``selected``.
    The returned GeoDataFrame is the recommended input to
    :func:`~hotelling.spatial.census.build_full_grid`.

    Parameters
    ----------
    lor:
        Full LOR GeoDataFrame in EPSG:3035 with ``PLR_ID`` column.
    boundary:
        GeoDataFrame of the inner-Ringbahn polygon (EPSG:3035).
    population_grid:
        Zensus polygon grid (from :func:`~hotelling.spatial.census.build_grid_polygons`)
        in EPSG:3035 with an ``Einwohner`` column.
    buffer_distance:
        Buffer in metres around the Ringbahn boundary for the initial
        candidate selection.  Default 500 m.
    extend_selection_by:
        Number of additional high-density adjacent LOR units to include
        beyond the initial buffer selection.  Default 6.

    Returns
    -------
    geopandas.GeoDataFrame
        Subset of *lor* covering the Ringbahn study area, retaining all
        original columns plus the scoring columns added by
        :func:`refine_shapes_selection`.
    """
    refined = refine_shapes_selection(
        lor,
        boundary.geometry,
        population_grid,
        buffer_distance=buffer_distance,
        extend_selection_by=extend_selection_by,
    )
    if refined.empty or "selected" not in refined.columns:
        return refined
    return refined.loc[refined["selected"]].copy()