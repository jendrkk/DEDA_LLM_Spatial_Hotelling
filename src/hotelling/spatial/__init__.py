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

__all__ = [
    "CHAIN_QID_MAP",
    "CHAIN_TYPE_MAP",
    "SquareGrid",
    "add_lcc_layer",
    "add_lor_attributes",
    "add_poi_layer",
    "assemble_simulation_grid",
    "build_commercial_candidates",
    "build_demand_grid",
    "build_full_grid",
    "build_grid_polygons",
    "build_transit_travel_times",
    "chain_type_to_quality",
    "clip_grid_to_boundary",
    "download_city_boundary",
    "download_index_data",
    "download_local_shapes",
    "download_lor_shapes",
    "download_relation_boundary",
    "download_stadtstruktur",
    "download_station_data",
    "download_zensus_2022",
    "enrich_supermarkets_with_brw",
    "euclidean_distance_matrix",
    "fetch_pois",
    "filter_zensus_2022",
    "identify_cbd",
    "identify_transport_hubs",
    "join_lor_names",
    "load_boundary",
    "load_berlin_city",
    "load_ghs_pop_fallback",
    "load_lor",
    "load_zensus_2022",
    "make_cell_id",
    "network_distance_matrix",
    "normalize_chain_name",
    "normalize_social_indices",
    "process_esix_mss_data",
    "process_gebaeude_stadtstruktur",
    "process_ihk_data",
    "process_supermarkets",
    "run_default_data_pipeline",
    "run_prime_location_clustering",
    "select_ringbahn_lor",
]

_LAZY_GEO: dict[str, tuple[str, str]] = {
    "CHAIN_QID_MAP": ("hotelling.spatial.osm", "CHAIN_QID_MAP"),
    "CHAIN_TYPE_MAP": ("hotelling.spatial.osm", "CHAIN_TYPE_MAP"),
    "process_supermarkets": ("hotelling.spatial.osm", "process_supermarkets"),
    "build_commercial_candidates": ("hotelling.spatial.parcels", "build_commercial_candidates"),
    "build_demand_grid": ("hotelling.spatial.assembly", "build_demand_grid"),
    "build_transit_travel_times": ("hotelling.spatial.distance", "build_transit_travel_times"),
    "chain_type_to_quality": ("hotelling.spatial.loader", "chain_type_to_quality"),
    "enrich_supermarkets_with_brw": ("hotelling.spatial.assembly", "enrich_supermarkets_with_brw"),
    "make_cell_id": ("hotelling.spatial.census", "make_cell_id"),
    "normalize_social_indices": ("hotelling.spatial.assembly", "normalize_social_indices"),
    "add_lcc_layer": ("hotelling.spatial.assembly", "add_lcc_layer"),
    "add_lor_attributes": ("hotelling.spatial.assembly", "add_lor_attributes"),
    "add_poi_layer": ("hotelling.spatial.assembly", "add_poi_layer"),
    "assemble_simulation_grid": ("hotelling.spatial.assembly", "assemble_simulation_grid"),
    "build_full_grid": ("hotelling.spatial.census", "build_full_grid"),
    "build_grid_polygons": ("hotelling.spatial.census", "build_grid_polygons"),
    "clip_grid_to_boundary": ("hotelling.spatial.census", "clip_grid_to_boundary"),
    "download_zensus_2022": ("hotelling.spatial.census", "download_zensus_2022"),
    "filter_zensus_2022": ("hotelling.spatial.census", "filter_zensus_2022"),
    "load_ghs_pop_fallback": ("hotelling.spatial.census", "load_ghs_pop_fallback"),
    "load_zensus_2022": ("hotelling.spatial.census", "load_zensus_2022"),
    "run_default_data_pipeline": ("hotelling.spatial.exe", "run_default_data_pipeline"),
    "download_city_boundary": ("hotelling.spatial.boundaries", "download_city_boundary"),
    "download_relation_boundary": ("hotelling.spatial.boundaries", "download_relation_boundary"),
    "load_boundary": ("hotelling.spatial.boundaries", "load_boundary"),
    "load_berlin_city": ("hotelling.spatial.loader", "load_berlin_city"),
    "download_index_data": ("hotelling.spatial.city_data", "download_index_data"),
    "download_stadtstruktur": ("hotelling.spatial.city_data", "download_stadtstruktur"),
    "download_station_data": ("hotelling.spatial.city_data", "download_station_data"),
    "process_ihk_data": ("hotelling.spatial.city_data", "process_ihk_data"),
    "process_esix_mss_data": ("hotelling.spatial.city_data", "process_esix_mss_data"),
    "process_gebaeude_stadtstruktur": ("hotelling.spatial.city_data", "process_gebaeude_stadtstruktur"),
    "run_prime_location_clustering": ("hotelling.spatial.city_data", "run_prime_location_clustering"),
    "identify_transport_hubs": ("hotelling.spatial.city_data", "identify_transport_hubs"),
    "identify_cbd": ("hotelling.spatial.city_data", "identify_cbd"),
    "download_lor_shapes": ("hotelling.spatial.admin", "download_lor_shapes"),
    "download_local_shapes": ("hotelling.spatial.admin", "download_local_shapes"),
    "join_lor_names": ("hotelling.spatial.admin", "join_lor_names"),
    "load_lor": ("hotelling.spatial.admin", "load_lor"),
    "select_ringbahn_lor": ("hotelling.spatial.admin", "select_ringbahn_lor"),
    "fetch_pois": ("hotelling.spatial.osm", "fetch_pois"),
    "normalize_chain_name": ("hotelling.spatial.osm", "normalize_chain_name"),
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
