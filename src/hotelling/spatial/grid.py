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


@dataclass
class SquareGrid:
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
    population: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.population is None:
            self.population = np.ones((self.height, self.width))

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
