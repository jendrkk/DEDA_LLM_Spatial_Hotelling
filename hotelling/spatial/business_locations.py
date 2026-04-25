"""
Real business location data loader.

Loads point-of-interest (POI) or business location datasets (CSV, GeoJSON,
etc.) and can use them to initialise firm positions in the Hotelling model
with empirically observed starting locations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BusinessLocations:
    """Load and query real-world business location data.

    Parameters
    ----------
    filepath:
        Path to a CSV (with *lon* / *lat* columns) or a GeoJSON file.
    map_loader:
        An optional :class:`~hotelling.spatial.map_loader.MapLoader` used to
        normalise coordinates into model space.
    """

    def __init__(
        self,
        filepath: Optional[str | Path] = None,
        map_loader=None,  # noqa: ANN001
    ) -> None:
        self.map_loader = map_loader
        self._locations: List[Tuple[float, float]] = []

        if filepath is not None:
            self._load(Path(filepath))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                self._load_csv(path)
            elif suffix in (".geojson", ".json", ".gpkg", ".shp"):
                self._load_geo(path)
            else:
                logger.warning("Unsupported file format: %s", suffix)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load business locations: %s", exc)

    def _load_csv(self, path: Path) -> None:
        import csv  # noqa: PLC0415

        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                lon_key = next((k for k in row if k.lower() in ("lon", "longitude", "x")), None)
                lat_key = next((k for k in row if k.lower() in ("lat", "latitude", "y")), None)
                if lon_key and lat_key:
                    self._locations.append((float(row[lon_key]), float(row[lat_key])))
        logger.info("Loaded %d business locations from CSV.", len(self._locations))

    def _load_geo(self, path: Path) -> None:
        try:
            import geopandas as gpd  # noqa: PLC0415

            gdf = gpd.read_file(path)
            for geom in gdf.geometry:
                self._locations.append((geom.x, geom.y))
            logger.info("Loaded %d business locations from %s.", len(self._locations), path.suffix)
        except ImportError:
            logger.warning("geopandas not installed; cannot load geographic business data.")

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def normalised_locations(self) -> List[Tuple[float, float]]:
        """Return all business locations normalised to model space."""
        if self.map_loader is None:
            return list(self._locations)
        return [self.map_loader.normalise(x, y) for x, y in self._locations]

    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> List[Tuple[float, float]]:
        """Return *n* randomly sampled (normalised) business locations."""
        locs = self.normalised_locations()
        if not locs:
            logger.warning("No business locations loaded; returning empty sample.")
            return []
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(locs), size=min(n, len(locs)), replace=False)
        return [locs[i] for i in indices]

    def __len__(self) -> int:
        return len(self._locations)
