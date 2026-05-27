"""Spatial data pipeline orchestrator for the hotelling package.

This module is the single entry point for running the complete Berlin
inner-Ringbahn spatial data pipeline.  It chains all download, filter,
selection, grid-construction, enrichment, and assembly steps into one
reproducible function: :func:`run_default_data_pipeline`.

Pipeline phases
---------------
1. **Download** — census grid, city and Ringbahn boundaries, LOR shapes,
   social-status indices (ESIx / MSS), urban-structure data, transit
   timetables (GTFS / DB), OSM supermarket POIs.
2. **Filter** — clip Zensus 2022 100 m grid to Berlin city boundary.
3. **LOR selection** — select and extend LOR planning districts that cover
   the inner-Ringbahn study area using a population-density-weighted
   greedy algorithm.
4. **Grid construction** — build the full INSPIRE 100 m lattice within the
   selected LOR districts; convert midpoints to square polygons.
5. **Grid enrichment** — attach ESIx / MSS social-status scores, IHK
   employment counts, transit hub flags, and CBD flags to each cell.
6. **POI layer** — assign OSM supermarket POIs to grid cells and create
   per-chain presence columns.
7. **Assembly** — merge all layers, verify schema, and write
   ``data/processed/simulation_grid.parquet``.

Outputs
-------
Written to ``data/raw/``:
    ``zensus2022_grid.parquet``, ``city_boundary_Berlin.geojson``,
    ``relation_boundary_14983.geojson``, ``lor_shapes_2019/2021.parquet``,
    ``esix.gpkg``, ``mss.gpkg``, ``stadtstruktur.gpkg``, ``gebaeude.gpkg``,
    ``zentren.gpkg``, ``db_station_data.csv``, ``OSM_POIs_Berlin.parquet``

Written to ``data/processed/``:
    ``zensus2022_grid_filtered.parquet``, ``lor_2019/2021.parquet``,
    ``lor.parquet`` (canonical), ``lor_ringbahn.parquet``,
    ``pop_grid.parquet``, ``simulation_grid.parquet``

Usage
-----
Run from the command line::

    hotelling-spatial
    # or
    python -m hotelling.spatial.exe

Run from Python::

    from hotelling.spatial import run_default_data_pipeline
    run_default_data_pipeline()

Note
----
IHK business microdata cannot be downloaded automatically.  Place
``2023_12_IHK_Berlin_Gewerbedaten.csv`` in ``data/raw/`` before running the
pipeline, or pass its path explicitly via the ``ihk_path`` parameter.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

from hotelling.spatial.admin import (
    download_lor_shapes,
    download_local_shapes,
    find_optimal_rectangle,
    join_lor_names,
    load_lor,
    select_ringbahn_lor,
)
from hotelling.spatial.assembly import (
    add_lcc_layer,
    add_lor_attributes,
    add_poi_layer,
    assemble_simulation_grid,
    build_demand_grid,
    enrich_supermarkets_with_brw,
    normalize_social_indices,
)
from hotelling.spatial.boundaries import (
    download_city_boundary,
    download_relation_boundary,
    load_boundary,
)
from hotelling.spatial.census import (
    build_full_grid,
    build_grid_polygons,
    download_zensus_2022,
    filter_zensus_2022,
    load_zensus_2022,
)
from hotelling.spatial.city_data import (
    download_index_data,
    download_stadtstruktur,
    download_station_data,
    identify_transport_hubs,
    process_esix_mss_data,
    process_gebaeude_stadtstruktur,
    process_ihk_data,
    run_prime_location_clustering,
)
from hotelling.spatial.osm import fetch_pois, process_supermarkets
from hotelling.spatial.distance import build_transit_travel_times
from hotelling.spatial.parcels import build_commercial_candidates


__all__ = [
    "add_lcc_layer",
    "add_lor_attributes",
    "add_poi_layer",
    "assemble_simulation_grid",
    "build_commercial_candidates",
    "build_demand_grid",
    "build_full_grid",
    "build_grid_polygons",
    "build_transit_travel_times",
    "download_city_boundary",
    "download_index_data",
    "download_local_shapes",
    "download_lor_shapes",
    "download_relation_boundary",
    "download_stadtstruktur",
    "download_station_data",
    "download_zensus_2022",
    "fetch_pois",
    "filter_zensus_2022",
    "find_optimal_rectangle",
    "identify_transport_hubs",
    "join_lor_names",
    "load_boundary",
    "load_lor",
    "load_zensus_2022",
    "process_esix_mss_data",
    "process_gebaeude_stadtstruktur",
    "process_ihk_data",
    "process_supermarkets",
    "enrich_supermarkets_with_brw",
    "run_default_data_pipeline",
    "run_prime_location_clustering",
    "select_ringbahn_lor",
    "normalize_social_indices",
]

logger = logging.getLogger(__name__)

def run_default_data_pipeline(
    lor_year: int = 2021,
    ringbahn_relation_id: int = 14983,
    buffer_distance: float = 500.0,
    extend_selection_by: int = 6,
    ihk_path: Path | None = None,
    rect_buffer_distance: float = 350.0,
    rect_augment_layers: tuple[int, int, int, int] = (2, 0, 4, 2),
    rect_tolerance: float = 0.01,
) -> None:
    """Run the complete Berlin inner-Ringbahn spatial data pipeline.

    Executes all seven phases in sequence:
    download → filter → LOR selection → grid construction →
    grid enrichment → POI layer → final assembly.

    Parameters
    ----------
    lor_year:
        Which LOR version to use as the administrative unit geometry.
        2019 or 2021 (default).
    ringbahn_relation_id:
        OSM relation ID of the inner-Ringbahn boundary polygon.
        Default 14983 (S41/S42 ring, as per ADR-012).
    buffer_distance:
        Buffer in metres around the Ringbahn boundary for initial LOR
        candidate selection.  Default 500 m.
    extend_selection_by:
        Number of additional high-density LOR units to include beyond
        the initial buffer.  Default 6.
    ihk_path:
        Path to the IHK Berlin business microdata CSV.  If ``None`` or the
        file does not exist, the employment-enrichment step is skipped with
        a warning.  Default path checked: ``data/raw/2023_12_IHK_Berlin_Gewerbedaten.csv``.
    rect_buffer_distance:
        Buffer in metres by which the Ringbahn boundary is expanded before
        fitting the optimal rectangular grid boundary.  Default 350.0 m.
    rect_augment_layers:
        ``[top, right, bottom, left]`` extra 100 m grid-cell layers added to
        each side of the optimal rectangle after optimisation.
        Default ``(2, 0, 4, 2)`` matches GEO_01 notebook parameters.
    rect_tolerance:
        Minimum relative improvement in population density to prefer a
        larger rectangle.  Default 0.01 (1%).  See
        :func:`~hotelling.spatial.admin.find_optimal_rectangle`.
    """
    # Resolve default IHK path
    _ihk_path = ihk_path or Path("data/raw/2023_12_IHK_Berlin_Gewerbedaten.csv")

    # ------------------------------------------------------------------
    # PHASE 1 — DOWNLOAD RAW DATA
    # ------------------------------------------------------------------
    logger.info("=== PHASE 1: Downloading raw data ===")

    download_zensus_2022()
    download_city_boundary("Berlin")
    download_relation_boundary(ringbahn_relation_id)

    download_lor_shapes(if_old=True)
    join_lor_names(if_old=True)
    download_lor_shapes(if_old=False)
    join_lor_names(if_old=False)

    download_index_data()       # ESIx 2022, MSS 2023
    download_stadtstruktur()    # Stadtstruktur, Gebaeude, Zentren
    download_station_data()     # DB station list + VBB GTFS

    fetch_pois("Berlin")        # OSM supermarkets → cached parquet

    logger.info("Phase 1 complete: all raw data downloaded.")

    # ------------------------------------------------------------------
    # PHASE 2 — FILTER CENSUS TO BERLIN
    # ------------------------------------------------------------------
    logger.info("=== PHASE 2: Filtering Zensus grid to Berlin boundary ===")

    filter_zensus_2022(Path("data/raw/city_boundary_Berlin.geojson"))

    logger.info("Phase 2 complete.")

    # ------------------------------------------------------------------
    # PHASE 3 — LOR SELECTION (used for study-area scoping, not grid boundary)
    # ------------------------------------------------------------------
    logger.info("=== PHASE 3: Selecting LOR districts for Ringbahn study area ===")

    zensus_filtered = gpd.read_parquet("data/raw/zensus2022_grid_filtered.parquet")
    boundary = load_boundary(Path(f"data/raw/relation_boundary_{ringbahn_relation_id}.geojson"))
    lor = load_lor(year=lor_year)

    zensus_polygons = build_grid_polygons(zensus_filtered)

    lor_ringbahn = select_ringbahn_lor(
        lor=lor,
        boundary=boundary,
        population_grid=zensus_polygons,
        buffer_distance=buffer_distance,
        extend_selection_by=extend_selection_by,
    )
    lor_ringbahn.to_parquet("data/processed/lor_ringbahn.parquet")
    logger.info(
        "Phase 3 complete: %d LOR districts selected.", len(lor_ringbahn)
    )

    # ------------------------------------------------------------------
    # PHASE 4 — BUILD POPULATION GRID
    # ------------------------------------------------------------------
    # IMPORTANT: The grid boundary is an optimal RECTANGLE fitted to the
    # Ringbahn boundary using find_optimal_rectangle, NOT the irregular
    # LOR polygon union. A rectangular boundary guarantees a regular
    # INSPIRE 100 m lattice with no jagged edges — required by the
    # simulation engine. Parameters match GEO_01_lor.ipynb.
    # ------------------------------------------------------------------
    logger.info("=== PHASE 4: Building population grid ===")

    optimal_rect = find_optimal_rectangle(
        boundary=boundary.geometry,
        population_grid=zensus_filtered,
        buffer_distance=rect_buffer_distance,
        cell_size=100.0,
        augument_rectangle_by_additional_layers=list(rect_augment_layers),
        max_iterations=10_000,
        tolerance=rect_tolerance,
    )
    logger.info(
        "Optimal rectangle: %d cols × %d rows (%.0f m × %.0f m).",
        int(optimal_rect["n_cols"].iloc[0]),
        int(optimal_rect["n_rows"].iloc[0]),
        float(optimal_rect["width_m"].iloc[0]),
        float(optimal_rect["height_m"].iloc[0]),
    )

    zensus_full = load_zensus_2022()
    pop_grid = build_full_grid(boundary=optimal_rect, zensus=zensus_full)

    # build_full_grid returns point geometry (midpoints).
    # Convert to 100 m square polygons for spatial joins in later phases.
    pop_grid = build_grid_polygons(pop_grid)

    pop_grid.to_parquet("data/processed/pop_grid.parquet")
    logger.info("Phase 4 complete: %d grid cells.", len(pop_grid))

    # ------------------------------------------------------------------
    # PHASE 5 — ENRICH GRID WITH CITY DATA
    # ------------------------------------------------------------------
    logger.info("=== PHASE 5: Enriching grid with socio-economic layers ===")

    grid = pop_grid.copy()

    # 5a. LOR attributes (PLR_ID, PLR_NAME, etc.)
    grid = add_lor_attributes(grid, lor_ringbahn)
    logger.info("LOR attributes joined.")

    # 5b. ESIx / MSS social-status indices
    grid = process_esix_mss_data(grid)
    logger.info("ESIx/MSS indices joined.")

    # 5c. IHK employment per grid cell (simple cell-level aggregation)
    if _ihk_path.exists():
        grid = process_ihk_data(grid, _ihk_path)
        logger.info("IHK cell employment joined from %s.", _ihk_path)
    else:
        logger.warning("IHK file not found at %s — skipping cell employment.", _ihk_path)

    # 5d. Building-level enrichment + IHK-to-building matching
    #     Produces data/processed/gebaeude_stadtstruktur.parquet
    gebaeude_stadtstruktur = process_gebaeude_stadtstruktur(ihk_path=_ihk_path)
    logger.info("gebaeude_stadtstruktur built (%d buildings).", len(gebaeude_stadtstruktur))

    # 5e. AABPL prime-location clustering
    #     Produces data/processed/prime_location_clusters.parquet
    try:
        run_prime_location_clustering(gebaeude_stadtstruktur)
        logger.info("Prime-location clusters computed.")
    except ImportError:
        logger.warning(
            "aabpl package not installed — skipping prime-location clustering. "
            "Install 'aabpl' and re-run to generate prime_location_clusters.parquet."
        )

    # 5f. Transport hubs (station_count, station_class per cell)
    #     Produces data/processed/grid_with_stations.parquet
    grid = identify_transport_hubs(grid)
    logger.info("Transport hub flags added.")

    logger.info("Phase 5 complete.")

    # ------------------------------------------------------------------
    # PHASE 6 — ADD OSM POI LAYERS
    # ------------------------------------------------------------------
    logger.info("=== PHASE 6: Adding OSM POI layers ===")

    # 6a. Supermarkets: fetch → normalize → clip to grid → produce supermarkets.parquet
    pois_raw = fetch_pois(type="supermarket", city="Berlin")
    supermarkets = process_supermarkets(pois_raw, grid)
    supermarkets.to_parquet("data/processed/supermarkets.parquet", index=False)
    logger.info("Supermarkets: %d in grid after normalisation.", len(supermarkets))

    # 6b-transit. Transit travel-time matrix (r5py; optional)
    #     Produces data/processed/travel_times.parquet
    try:
        travel_times_df = build_transit_travel_times(grid=grid, supermarkets=supermarkets)
        logger.info("Transit travel-time matrix computed (%d rows).", len(travel_times_df))
    except ImportError:
        logger.warning(
            "r5py not installed — skipping transit travel-time computation. "
            "Install with: pip install hotelling[transit]"
        )
    except Exception as exc:
        logger.warning("Transit travel-time computation failed: %s", exc)

    # 6b. POI layer: count and chain flags per grid cell
    grid = add_poi_layer(grid, supermarkets)
    logger.info("POI layer added.")

    # 6c. LCC malls: produces data/processed/grid_malls.parquet
    lcc_gdf = fetch_pois(type="LCC", city="Berlin")
    grid = add_lcc_layer(grid, lcc_gdf)
    logger.info("LCC mall layer added.")

    # 6d. Commercial entry-site candidates (L_commercial_final)
    #     Produces data/processed/L_commercial_final.parquet + .gpkg
    _alkis_path = Path("data/raw/alkis_full.gpkg")
    if _alkis_path.exists():
        try:
            _brw = gpd.read_file("data/raw/brw_2025.gpkg").to_crs("EPSG:3035")
            _ss  = gpd.read_file("data/raw/stadtstruktur.gpkg").to_crs("EPSG:3035")
            _incumbents_raw = fetch_pois(type="supermarket", city="Berlin")
            build_commercial_candidates(
                boundary=boundary,
                alkis_path=_alkis_path,
                incumbents=_incumbents_raw,
                brw=_brw,
                stadtstruktur=_ss,
            )
            logger.info("Commercial entry-site candidates built.")
        except Exception as exc:
            logger.warning("build_commercial_candidates failed: %s", exc)
    else:
        logger.warning(
            "ALKIS file not found at %s — skipping L_commercial_final. "
            "Run download_alkis_data() first.", _alkis_path
        )

    logger.info("Phase 6 complete.")

    # ------------------------------------------------------------------
    # 6e. Final demand grid assembly
    #     Requires: travel_times.parquet to exist on disk.
    _travel_times_path = Path("data/processed/travel_times.parquet")
    _empl_clusters_path = Path("data/processed/employment_clusters.parquet")
    if _travel_times_path.exists() and _empl_clusters_path.exists():
        try:
            import pandas as _pd
            _tt = _pd.read_parquet(_travel_times_path)
            _ec = gpd.read_parquet(_empl_clusters_path).to_crs("EPSG:3035")
            _grid_malls_path = Path("data/processed/grid_malls.parquet")
            _grid_stations_path = Path("data/processed/grid_with_stations.parquet")
            if _grid_malls_path.exists() and _grid_stations_path.exists():
                _gm = gpd.read_parquet(_grid_malls_path).to_crs("EPSG:3035")
                _gs = gpd.read_parquet(_grid_stations_path).to_crs("EPSG:3035")
                build_demand_grid(
                    grid=grid,
                    grid_malls=_gm,
                    grid_with_stations=_gs,
                    travel_times=_tt,
                    employment_clusters=_ec,
                )
                logger.info("Demand grid assembled.")
                _brw_for_sm = gpd.read_file("data/raw/brw_2025.gpkg").to_crs("EPSG:3035")
                enrich_supermarkets_with_brw(supermarkets, _brw_for_sm)
                logger.info("Supermarkets enriched with BRW data.")
            else:
                logger.warning(
                    "grid_malls.parquet or grid_with_stations.parquet missing — "
                    "skipping demand grid assembly."
                )
        except Exception as exc:
            logger.warning("Demand grid assembly failed: %s", exc)
    else:
        logger.warning(
            "travel_times.parquet or employment_clusters.parquet not found — "
            "skipping demand grid assembly. Run build_transit_travel_times first."
        )

    # ------------------------------------------------------------------
    # PHASE 7 — ASSEMBLE & SAVE FINAL GRID
    # ------------------------------------------------------------------
    logger.info("=== PHASE 7: Assembling and saving simulation-ready grid ===")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    simulation_grid = assemble_simulation_grid(
        pop_grid=grid,
        lor=lor_ringbahn,
        pois=supermarkets,
    )
    simulation_grid.to_parquet("data/processed/simulation_grid.parquet")

    logger.info(
        "Pipeline complete. Simulation grid saved to data/processed/simulation_grid.parquet "
        "(%d cells, %d columns).",
        len(simulation_grid),
        len(simulation_grid.columns),
    )


def main() -> None:
    """Execute the default spatial data pipeline (delegates to :func:`run_default_data_pipeline`)."""
    run_default_data_pipeline()


if __name__ == "__main__":
    main()
