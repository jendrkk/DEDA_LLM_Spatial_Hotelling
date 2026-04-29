"""Backward-compatible re-exports for the former monolithic ``raster`` module.

Prefer importing from :mod:`hotelling.spatial.census`,
:mod:`hotelling.spatial.boundaries`, or :mod:`hotelling.spatial.admin`.
"""
from __future__ import annotations

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
    run_default_data_pipeline,
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
    "run_default_data_pipeline",
]


def main() -> None:
    """Run the default spatial data pipeline (delegates to :func:`run_default_data_pipeline`)."""
    run_default_data_pipeline()


if __name__ == "__main__":
    main()
