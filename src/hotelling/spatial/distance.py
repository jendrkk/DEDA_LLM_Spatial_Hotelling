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

from pathlib import Path
from typing import Optional

import numpy as np


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
    raise NotImplementedError


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
