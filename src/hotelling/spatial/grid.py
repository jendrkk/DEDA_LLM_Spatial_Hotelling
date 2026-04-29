"""SquareGrid with population PropertyLayer.

Responsibility: provide a 2-D grid representation of the city with population
density used for consumer sampling.

Public API: SquareGrid

Key dependencies: numpy, dataclasses

References: Mesa-Geo (not used); custom implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import urllib.request
import zipfile
import os
import pandas as pd


@dataclass
class Grid:
    """2-D grid with optional population density layer.

    Parameters
    ----------
    width: int - number of cells horizontally
    height: int - number of cells vertically
    cell_size: float - physical size of each cell in metres
    population: Optional[np.ndarray] - (height, width) population density per cell
    """

    width: int = 50
    height: int = 50
    cell_size: float = 100.0  # 100m cells (Zensus 2022 resolution)
    population: Optional[pd.DataFrame] = field(default=None, repr=False)
    crs: Optional[str] = None

    def __post_init__(self) -> None:
        if self.population is None:
            self.population = np.ones((self.height, self.width))
            
    def download_population_grid(self) -> None:
        """Download the population grid from the Zensus 2022 dataset."""
        
        link = "https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Bevoelkerungszahl.zip"
        save_path = "data/raw/zensus2022_grid.zip"
        
        urllib.request.urlretrieve(link, save_path)
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall("data/raw/zensus2022_grid")
        os.remove(save_path)
        
        data = pd.read_csv("data/raw/zensus2022_grid/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv")
        
        from pyproj import Transformer
        
        transformer = Transformer.from_crs("EPSG:25833", "EPSG:3035", always_xy=True)
        data["lon"], data["lat"] = transformer.transform(data["x_mp_100m"].to_numpy(), data["y_mp_100m"].to_numpy())
        
        pop = data[['lon', 'lat', 'Einwohner']]
        
        self.population = data
        
        
        self.crs = "EPSG:3035"
        
        # Has columns: "GITTER_ID_100m", "x_mp_100m", "y_mp_100m", "Einwohner"
        self.width = self.population["x_mp_100m"].max() - self.population["x_mp_100m"].min()
        self.height = self.population["y_mp_100m"].max() - self.population["y_mp_100m"].min()
        self.cell_size = 100.0
        
        
        
    def total_population(self) -> float:
        """Return total population across all cells."""
        return float(np.sum(self.population))

    def sample_locations(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n consumer locations proportional to population density.

        Returns
        -------
        np.ndarray shape (n, 2) in cell coordinates
        """
        raise NotImplementedError

    def cell_to_metres(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid cell (row, col) to (x, y) in metres from origin."""
        raise NotImplementedError
    