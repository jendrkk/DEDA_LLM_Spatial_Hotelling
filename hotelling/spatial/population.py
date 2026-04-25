"""
Population grid for real-world spatial population data.

Wraps raster population datasets (e.g. GPWv4, WorldPop, GHS-POP) and
provides methods to sample consumer locations proportional to population
density.  Falls back to uniform sampling when the raster data is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PopulationGrid:
    """Real-world population density mapped to model space.

    Parameters
    ----------
    raster_path:
        Path to a GeoTIFF raster file containing population counts.
    map_loader:
        An optional :class:`~hotelling.spatial.map_loader.MapLoader` instance
        used to normalise coordinates.
    """

    def __init__(
        self,
        raster_path: Optional[str | Path] = None,
        map_loader=None,  # noqa: ANN001
    ) -> None:
        self.map_loader = map_loader
        self._density: Optional[np.ndarray] = None
        self._transform = None

        if raster_path is not None:
            self._load_raster(Path(raster_path))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_raster(self, path: Path) -> None:
        try:
            import rasterio  # noqa: PLC0415

            with rasterio.open(path) as src:
                self._density = src.read(1).astype(float)
                self._density = np.maximum(self._density, 0.0)
                self._transform = src.transform
            logger.info("Loaded population raster from %s", path)
        except ImportError:
            logger.warning("rasterio not installed; falling back to uniform distribution.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load raster: %s", exc)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_locations(
        self,
        n: int,
        width: float = 1.0,
        height: float = 1.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sample *n* consumer locations in model space.

        If a population raster is available, locations are sampled
        proportionally to population density.  Otherwise they are sampled
        uniformly.

        Returns
        -------
        np.ndarray
            Shape *(n, 2)* array of *(x, y)* model-space coordinates.
        """
        rng = np.random.default_rng(seed)

        if self._density is None:
            return rng.uniform(low=[0.0, 0.0], high=[width, height], size=(n, 2))

        rows, cols = self._density.shape
        weights = self._density.ravel()
        total = weights.sum()
        if total <= 0:
            return rng.uniform(low=[0.0, 0.0], high=[width, height], size=(n, 2))

        probs = weights / total
        flat_indices = rng.choice(len(probs), size=n, p=probs)
        row_idx, col_idx = np.unravel_index(flat_indices, (rows, cols))

        # Convert raster pixel indices to model space
        xs = (col_idx + rng.uniform(size=n)) / cols * width
        ys = (row_idx + rng.uniform(size=n)) / rows * height
        return np.column_stack([xs, ys])
