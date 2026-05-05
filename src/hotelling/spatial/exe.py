"""Backward-compatible re-exports for the former monolithic ``raster`` module.

Prefer importing from :mod:`hotelling.spatial.census`,
:mod:`hotelling.spatial.boundaries`, or :mod:`hotelling.spatial.admin`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from hotelling.spatial.admin import download_local_shapes, download_lor_shapes
from hotelling.spatial.boundaries import (
    download_city_boundary,
    download_relation_boundary,
    load_boundary,
)
from hotelling.spatial.census import (
    build_full_grid,
    download_zensus_2022,
    filter_zensus_2022,
    load_ghs_pop_fallback,
    load_zensus_2022,
)
from hotelling.spatial.city_data import (
    download_index_data,
    download_stadtstruktur,
    download_station_data,
)


__all__ = [
    "build_full_grid",
    "download_city_boundary",
    "download_local_shapes",
    "download_lor_shapes",
    "download_relation_boundary",
    "download_zensus_2022",
    "filter_zensus_2022",
    "load_boundary",
    "load_ghs_pop_fallback",
    "load_zensus_2022",
    "download_index_data",
    "download_stadtstruktur",
    "download_station_data",
]

logger = logging.getLogger(__name__)

def run_default_data_pipeline() -> None:
    """Run the default Berlin-area data download and filter workflow (for scripts / demos)."""
    from hotelling.spatial.admin import download_lor_shapes, join_lor_names
    from hotelling.spatial.boundaries import download_city_boundary, download_relation_boundary

    logger.info("Starting census module default data pipeline.")
    download_zensus_2022()
    download_city_boundary("Berlin")
    download_relation_boundary(14983)
    download_lor_shapes(if_old=True)
    join_lor_names(if_old=True)
    download_lor_shapes(if_old=False)
    join_lor_names(if_old=False)
    filter_zensus_2022(Path("data/raw/city_boundary_Berlin.geojson"))
    
    logger.info("Completed census module default data pipeline.")


def main() -> None:
    """Execute the default spatial data pipeline (delegates to :func:`run_default_data_pipeline`)."""
    run_default_data_pipeline()


if __name__ == "__main__":
    main()
