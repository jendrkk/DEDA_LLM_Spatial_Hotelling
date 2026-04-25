"""
Map loader for real GIS / spatial data.

Supports loading geographic boundary shapefiles (via ``geopandas``) and
normalising coordinates to the unit square used by the Hotelling model.

``geopandas`` and ``shapely`` are optional dependencies; if they are absent
the loader falls back to rectangular synthetic boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MapLoader:
    """Load and normalise geographic boundary data.

    Parameters
    ----------
    filepath:
        Path to a shapefile, GeoJSON, or any format supported by
        ``geopandas.read_file()``.
    target_width:
        Width (in model units) of the normalised output boundary.
    target_height:
        Height (in model units) of the normalised output boundary.
    """

    def __init__(
        self,
        filepath: Optional[str | Path] = None,
        target_width: float = 1.0,
        target_height: float = 1.0,
    ) -> None:
        self.filepath = Path(filepath) if filepath else None
        self.target_width = target_width
        self.target_height = target_height
        self._gdf = None
        self._bounds: Optional[Tuple[float, float, float, float]] = None

        if self.filepath is not None:
            self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            import geopandas as gpd  # noqa: PLC0415

            self._gdf = gpd.read_file(self.filepath)
            minx, miny, maxx, maxy = self._gdf.total_bounds
            self._bounds = (float(minx), float(miny), float(maxx), float(maxy))
            logger.info("Loaded map from %s (bounds=%s)", self.filepath, self._bounds)
        except ImportError:
            logger.warning("geopandas not installed; using rectangular boundary.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load map: %s", exc)

    # ------------------------------------------------------------------
    # Coordinate normalisation
    # ------------------------------------------------------------------

    def normalise(self, x: float, y: float) -> Tuple[float, float]:
        """Map real-world *(x, y)* to model-space coordinates.

        If no geographic data has been loaded the coordinates are assumed to
        already be in model space and are returned unchanged.
        """
        if self._bounds is None:
            return x, y
        minx, miny, maxx, maxy = self._bounds
        nx = (x - minx) / (maxx - minx) * self.target_width
        ny = (y - miny) / (maxy - miny) * self.target_height
        return float(np.clip(nx, 0.0, self.target_width)), float(
            np.clip(ny, 0.0, self.target_height)
        )

    def normalise_array(self, coords: np.ndarray) -> np.ndarray:
        """Vectorised version of :meth:`normalise` for *(N, 2)* arrays."""
        if self._bounds is None:
            return coords
        minx, miny, maxx, maxy = self._bounds
        out = coords.copy().astype(float)
        out[:, 0] = np.clip(
            (out[:, 0] - minx) / (maxx - minx) * self.target_width, 0.0, self.target_width
        )
        out[:, 1] = np.clip(
            (out[:, 1] - miny) / (maxy - miny) * self.target_height, 0.0, self.target_height
        )
        return out

    @property
    def geodataframe(self):  # noqa: ANN201
        """The underlying ``geopandas.GeoDataFrame`` (may be *None*)."""
        return self._gdf
