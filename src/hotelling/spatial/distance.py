"""Distance matrix computation: KDTree Euclidean + OSRM network distance.

Responsibility: compute pairwise distance matrices between consumer and firm
locations. Supports both Euclidean (fast) and network routing (realistic).

Public API: euclidean_distance_matrix, network_distance_matrix

Key dependencies: numpy, scipy.spatial

All distances in metres.

References:
    scipy.spatial.KDTree;
    OSRM (Luxen & Vetter 2011).
"""
from __future__ import annotations

import csv
import logging
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "euclidean_distance_matrix",
    "network_distance_matrix",
    "build_transit_travel_times",
]


def euclidean_distance_matrix(
    locations_a: np.ndarray,
    locations_b: np.ndarray,
) -> np.ndarray:
    """Compute pairwise Euclidean distances using KDTree.

    Parameters
    ----------
    locations_a : shape (M, 2) array of (x, y) coordinates in metres
    locations_b : shape (N, 2) array of (x, y) coordinates in metres

    Returns
    -------
    np.ndarray shape (M, N) - distances in metres
    """
    a = np.asarray(locations_a, dtype=np.float64)
    b = np.asarray(locations_b, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("locations_a must have shape (M, 2).")
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("locations_b must have shape (N, 2).")
    from scipy.spatial.distance import cdist
    return cdist(a, b)


def network_distance_matrix(
    locations_a: np.ndarray,
    locations_b: np.ndarray,
    osrm_base_url: str = "http://router.project-osrm.org",
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """Compute pairwise network (routing) distances via OSRM table API.

    Parameters
    ----------
    locations_a : shape (M, 2) array of (lon, lat) in WGS84
    locations_b : shape (N, 2) array of (lon, lat) in WGS84
    osrm_base_url : OSRM server base URL
    cache_path : parquet file to cache results

    Returns
    -------
    np.ndarray shape (M, N) - network distances in metres
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Private helpers for build_transit_travel_times
# ---------------------------------------------------------------------------

def _gtfs_file_has_data_rows(path: Path) -> bool:
    """Return True only if a GTFS CSV file has at least one non-header data row."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return False
        for row in reader:
            if any(cell.strip() for cell in row):
                return True
    return False


def _build_gtfs_zip_from_dir(gtfs_dir: Path, gtfs_zip: Path) -> None:
    """Pack all non-empty GTFS .txt files from *gtfs_dir* into *gtfs_zip*.

    Header-only files (e.g. ``frequencies.txt``) are excluded to prevent
    r5py from failing on empty tables.

    Parameters
    ----------
    gtfs_dir : Path
        Directory containing unpacked GTFS ``.txt`` files.
    gtfs_zip : Path
        Destination ``.zip`` path (overwritten if it already exists).
    """
    if gtfs_zip.exists():
        gtfs_zip.unlink()

    included: list[str] = []
    skipped: list[str] = []

    for file in sorted(gtfs_dir.iterdir()):
        if not file.is_file() or file.suffix.lower() != ".txt":
            continue
        if _gtfs_file_has_data_rows(file):
            included.append(file.name)
        else:
            skipped.append(file.name)

    if not included:
        raise RuntimeError(
            f"No non-empty GTFS .txt files found in {gtfs_dir}. "
            "Ensure the GTFS feed is complete before building the zip."
        )

    with zipfile.ZipFile(gtfs_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in included:
            zf.write(gtfs_dir / name, arcname=name)

    logger.info(
        "GTFS zip created at %s  (included: %s; skipped: %s).",
        gtfs_zip,
        included,
        skipped,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_transit_travel_times(
    grid: "gpd.GeoDataFrame",
    supermarkets: "gpd.GeoDataFrame",
    osm_pbf_path: Optional[Path] = None,
    gtfs_dir: Optional[Path] = None,
    gtfs_zip: Optional[Path] = None,
    output_path: Optional[Path] = None,
    departure: Optional[datetime] = None,
    departure_time_window_hours: float = 2.0,
    max_time_minutes: int = 60,
    max_walking_minutes: int = 10,
    walking_speed_kmh: float = 4.8,
    jvm_max_memory: str = "10G",
) -> pd.DataFrame:
    """Compute a transit + walk travel-time matrix from grid cells to supermarkets.

    Uses r5py (https://r5py.readthedocs.io) with a multimodal transport
    network built from an OSM PBF extract and a VBB GTFS feed.

    Origins  — centroid of each grid cell (column ``GITTER_ID_100m`` used as ID).
    Destinations — centroid of each supermarket (integer index used as ID).

    Parameters
    ----------
    grid:
        Population grid GeoDataFrame in EPSG:3035 with polygon geometry.
        Must contain ``GITTER_ID_100m`` or ``x_mp_100m`` / ``y_mp_100m``
        columns so that :func:`~hotelling.spatial.census.make_cell_id` can
        assign cell IDs.
    supermarkets:
        Supermarket POI GeoDataFrame in any CRS (will be reprojected to EPSG:3035).
    osm_pbf_path:
        Path to the Berlin OSM PBF extract.
        Default: ``<repo_root>/data/raw/berlin-*.osm.pbf`` (auto-detected).
    gtfs_dir:
        Directory with unpacked GTFS ``.txt`` files.
        Default: ``<repo_root>/data/raw/gtfs/``.
    gtfs_zip:
        Path where the r5py-ready GTFS zip will be written/cached.
        Default: ``<repo_root>/data/raw/gtfs_berlin.zip``.
    output_path:
        Parquet path where the result is saved.
        Default: ``<repo_root>/data/processed/travel_times.parquet``.
    departure:
        ``datetime`` for the transit departure.
        Default: ``datetime(2025, 10, 7, 10, 0)`` (Tuesday, representative
        weekday morning, chosen to match the GEO_05 notebook).
    departure_time_window_hours:
        Width of the departure-time window in hours.  r5py samples multiple
        departures within this window.  Default 2.0.
    max_time_minutes:
        Maximum travel time cap in minutes.  Journeys longer than this are
        returned as ``NaN``.  Default 60.
    max_walking_minutes:
        Maximum pure-walking leg duration in minutes.  Default 10.
    walking_speed_kmh:
        Walking speed used by r5py.  Default 4.8 km/h.
    jvm_max_memory:
        JVM heap size string passed to r5py (via JAVA_TOOL_OPTIONS).
        Default ``"10G"``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``from_id`` (str, INSPIRE cell ID), ``to_id`` (str, store
        integer index as str), ``travel_time`` (int, minutes, NaN if
        unreachable).

    Raises
    ------
    ImportError
        If ``r5py`` is not installed.  Install with::

            pip install hotelling[transit]

    Notes
    -----
    r5py requires a Java Runtime Environment (JDK 11+) to be installed
    on the host machine.  The ``JAVA_TOOL_OPTIONS`` environment variable is
    set automatically before the JVM is started.

    The first call builds the transport network and takes several minutes;
    subsequent calls with the same network parameters are fast because r5py
    caches the built network in ``~/.cache/r5py``.
    """
    try:
        import r5py  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "r5py is not installed.  Install it with:\n"
            "    pip install hotelling[transit]\n"
            "Note: r5py also requires a Java Runtime Environment (JDK 11+)."
        ) from exc

    import geopandas as gpd  # lazy spatial import
    from hotelling.spatial.census import make_cell_id  # noqa: PLC0415

    # ── Resolve default paths from repo root ─────────────────────────────────
    def _repo_root() -> Path:
        """Walk parents until data/raw exists."""
        for p in [Path.cwd(), *Path.cwd().parents]:
            if (p / "data" / "raw").exists():
                return p
        raise FileNotFoundError(
            "Cannot locate repo root (data/raw not found). "
            "Run from within the DEDA_LLM_Spatial_Hotelling repository."
        )

    root = _repo_root()

    if osm_pbf_path is None:
        pbf_candidates = list((root / "data" / "raw").glob("*.osm.pbf"))
        if not pbf_candidates:
            raise FileNotFoundError(
                f"No *.osm.pbf file found in {root / 'data' / 'raw'}. "
                "Download a Berlin OSM extract first."
            )
        osm_pbf_path = pbf_candidates[0]

    _gtfs_dir = gtfs_dir or (root / "data" / "raw" / "gtfs")
    _gtfs_zip = gtfs_zip or (root / "data" / "raw" / "gtfs_berlin.zip")
    _output_path = output_path or (root / "data" / "processed" / "travel_times.parquet")
    _departure = departure or datetime(2025, 10, 7, 10, 0)

    if not _gtfs_dir.exists():
        raise FileNotFoundError(
            f"GTFS directory not found at {_gtfs_dir}. "
            "Run download_station_data() to fetch the GTFS feed first."
        )

    # ── JVM configuration (must be set before r5py import / JVM start) ──────
    os.environ["JAVA_TOOL_OPTIONS"] = f"-Xmx{jvm_max_memory} -XX:+UseG1GC"

    # ── Build GTFS zip from directory ────────────────────────────────────────
    logger.info("Building GTFS zip from %s → %s.", _gtfs_dir, _gtfs_zip)
    _build_gtfs_zip_from_dir(_gtfs_dir, _gtfs_zip)

    # ── Build transport network ───────────────────────────────────────────────
    logger.info(
        "Building r5py transport network from PBF=%s, GTFS=%s.",
        osm_pbf_path.name, _gtfs_zip.name,
    )
    transport_network = r5py.TransportNetwork(
        osm_pbf=osm_pbf_path,
        gtfs=[_gtfs_zip],
    )

    # ── Prepare origins (grid cell centroids) ────────────────────────────────
    grid_copy = grid.copy()
    grid_copy["GITTER_ID_100m"] = grid_copy.apply(make_cell_id, axis=1)
    origins = grid_copy[["GITTER_ID_100m", "geometry"]].copy()
    origins["geometry"] = origins.geometry.centroid
    origins = origins.rename(columns={"GITTER_ID_100m": "id"}).to_crs("EPSG:3035")

    # ── Prepare destinations (supermarket centroids) ──────────────────────────
    sm = supermarkets.copy()
    sm = sm.to_crs("EPSG:3035") if sm.crs != origins.crs else sm
    sm["store_id"] = sm.index.astype(str)
    destinations = sm[["store_id", "geometry"]].rename(columns={"store_id": "id"})

    # ── Compute travel-time matrix ────────────────────────────────────────────
    logger.info(
        "Computing transit travel-time matrix: %d origins × %d destinations.",
        len(origins), len(destinations),
    )
    ttm = r5py.TravelTimeMatrix(
        transport_network,
        origins=origins,
        destinations=destinations,
        departure=_departure,
        departure_time_window=timedelta(hours=departure_time_window_hours),
        transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
        max_time=timedelta(minutes=max_time_minutes),
        max_time_walking=timedelta(minutes=max_walking_minutes),
        speed_walking=walking_speed_kmh,
        percentiles=[50],
    )

    # ── Save and return ────────────────────────────────────────────────────────
    _output_path.parent.mkdir(parents=True, exist_ok=True)
    ttm.to_parquet(_output_path, index=False)
    logger.info(
        "Travel-time matrix saved → %s  (%d rows).", _output_path, len(ttm)
    )
    return ttm
