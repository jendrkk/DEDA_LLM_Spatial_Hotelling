"""Spatial building blocks: grids, distances, OSM POIs, census rasters, boundaries.

Heavy GIS dependencies (GeoPandas, Rasterio, OSMnx, …) are optional extras
(``pip install hotelling[spatial]``). Symbols that need them are loaded lazily
so ``from hotelling.spatial import SquareGrid`` works with only NumPy installed.
"""
from __future__ import annotations

import importlib
from typing import Any

from hotelling.spatial.distance import euclidean_distance_matrix, network_distance_matrix
from hotelling.spatial.grid import SquareGrid
from hotelling.spatial.osm import CHAIN_QID_MAP, fetch_pois, normalize_chain_name

__all__ = [
    "CHAIN_QID_MAP",
    "SquareGrid",
    "build_full_grid",
    "download_city_boundary",
    "download_local_shapes",
    "download_lor_shapes",
    "download_relation_boundary",
    "download_zensus_2022",
    "euclidean_distance_matrix",
    "fetch_pois",
    "filter_zensus_2022",
    "load_boundary",
    "load_ghs_pop_fallback",
    "load_zensus_2022",
    "network_distance_matrix",
    "normalize_chain_name",
    "run_default_data_pipeline",
]

_LAZY_GEO: dict[str, tuple[str, str]] = {
    "build_full_grid": ("hotelling.spatial.census", "build_full_grid"),
    "download_zensus_2022": ("hotelling.spatial.census", "download_zensus_2022"),
    "filter_zensus_2022": ("hotelling.spatial.census", "filter_zensus_2022"),
    "load_ghs_pop_fallback": ("hotelling.spatial.census", "load_ghs_pop_fallback"),
    "load_zensus_2022": ("hotelling.spatial.census", "load_zensus_2022"),
    "run_default_data_pipeline": ("hotelling.spatial.census", "run_default_data_pipeline"),
    "download_city_boundary": ("hotelling.spatial.boundaries", "download_city_boundary"),
    "download_relation_boundary": ("hotelling.spatial.boundaries", "download_relation_boundary"),
    "load_boundary": ("hotelling.spatial.boundaries", "load_boundary"),
    "download_lor_shapes": ("hotelling.spatial.admin", "download_lor_shapes"),
    "download_local_shapes": ("hotelling.spatial.admin", "download_local_shapes"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_GEO:
        module_name, attr = _LAZY_GEO[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(__all__)
