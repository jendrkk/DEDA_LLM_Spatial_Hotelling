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
    join_lor_names,
    load_lor,
    select_ringbahn_lor,
)
from hotelling.spatial.assembly import (
    add_lor_attributes,
    add_poi_layer,
    assemble_simulation_grid,
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
    identify_cbd,
    identify_transport_hubs,
    process_esix_mss_data,
    process_ihk_data,
)
from hotelling.spatial.osm import fetch_pois


__all__ = [
    # Re-exported for convenience — import from here or from the sub-modules
    "add_lor_attributes",
    "add_poi_layer",
    "assemble_simulation_grid",
    "build_full_grid",
    "build_grid_polygons",
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
    "identify_cbd",
    "identify_transport_hubs",
    "join_lor_names",
    "load_boundary",
    "load_lor",
    "load_zensus_2022",
    "process_esix_mss_data",
    "process_ihk_data",
    "run_default_data_pipeline",
    "select_ringbahn_lor",
]

logger = logging.getLogger(__name__)

def run_default_data_pipeline(
    lor_year: int = 2021,
    ringbahn_relation_id: int = 14983,
    buffer_distance: float = 500.0,
    extend_selection_by: int = 6,
    ihk_path: Path | None = None,
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
    # PHASE 3 — LOR SELECTION
    # ------------------------------------------------------------------
    logger.info("=== PHASE 3: Selecting LOR districts for Ringbahn study area ===")

    zensus_filtered = gpd.read_parquet("data/raw/zensus2022_grid_filtered.parquet")
    boundary = load_boundary(Path(f"data/raw/relation_boundary_{ringbahn_relation_id}.geojson"))
    lor = load_lor(year=lor_year)

    # Convert Zensus midpoints → square polygon cells for population join
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
        "Phase 3 complete: %d LOR districts selected for Ringbahn area.",
        len(lor_ringbahn),
    )

    # ------------------------------------------------------------------
    # PHASE 4 — BUILD POPULATION GRID
    # ------------------------------------------------------------------
    logger.info("=== PHASE 4: Building population grid ===")

    zensus_full = load_zensus_2022()
    pop_grid = build_full_grid(boundary=lor_ringbahn, zensus=zensus_full)

    # Convert midpoint grid to polygon grid (100m squares) for spatial joins
    pop_grid = build_grid_polygons(pop_grid)

    pop_grid.to_parquet("data/processed/pop_grid.parquet")
    logger.info("Phase 4 complete: %d grid cells built.", len(pop_grid))

    # ------------------------------------------------------------------
    # PHASE 5 — ENRICH GRID WITH CITY DATA
    # ------------------------------------------------------------------
    logger.info("=== PHASE 5: Enriching grid with socio-economic layers ===")

    grid = pop_grid.copy()

    grid = add_lor_attributes(grid, lor_ringbahn)
    logger.info("LOR attributes joined.")

    grid = process_esix_mss_data(grid)
    logger.info("ESIx / MSS social-status indices joined.")

    if _ihk_path.exists():
        grid = process_ihk_data(grid, _ihk_path)
        logger.info("IHK employment data joined from %s.", _ihk_path)
    else:
        logger.warning(
            "IHK data not found at %s — skipping employment enrichment. "
            "Place the file there and re-run to include it.",
            _ihk_path,
        )

    grid = identify_transport_hubs(grid)
    logger.info("Transit hub flags added.")

    grid = identify_cbd(grid)
    logger.info("CBD flags added.")

    logger.info("Phase 5 complete.")

    # ------------------------------------------------------------------
    # PHASE 6 — ADD POI LAYER
    # ------------------------------------------------------------------
    logger.info("=== PHASE 6: Adding OSM POI layer ===")

    pois = fetch_pois("Berlin")
    grid = add_poi_layer(grid, pois)
    logger.info("POI layer added: %d POIs assigned to grid.", len(pois))

    # ------------------------------------------------------------------
    # PHASE 7 — ASSEMBLE & SAVE FINAL GRID
    # ------------------------------------------------------------------
    logger.info("=== PHASE 7: Assembling and saving simulation-ready grid ===")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    simulation_grid = assemble_simulation_grid(
        pop_grid=grid,
        lor=lor_ringbahn,
        pois=pois,
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
