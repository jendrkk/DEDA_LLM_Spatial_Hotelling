"""Regular lattice with an optional population density layer per cell.

Used for consumer sampling and distance games on a 2-D cell grid.

Key dependency: numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class SquareGrid:
    """Square cell grid with optional population weights per cell.

    Parameters
    ----------
    width
        Number of cells along the horizontal axis.
    height
        Number of cells along the vertical axis.
    cell_size
        Edge length of each square cell in metres.
    population
        Array of shape ``(height, width)`` with non-negative weights per cell.
        If omitted, weights are uniform ones.
    crs
        Optional coordinate reference system identifier for downstream GIS use.
    """

    width: int = 50
    height: int = 50
    cell_size: float = 100.0
    population: Optional[np.ndarray] = field(default=None, repr=False)
    crs: Optional[str] = None

    def __post_init__(self) -> None:
        if self.population is None:
            self.population = np.ones((self.height, self.width), dtype=np.float64)
        else:
            arr = np.asarray(self.population, dtype=np.float64)
            if arr.shape != (self.height, self.width):
                msg = f"population shape {arr.shape} must equal (height, width) = ({self.height}, {self.width})"
                raise ValueError(msg)
            self.population = arr

    def total_population(self) -> float:
        """Sum of population weights over all cells."""
        return float(np.sum(self.population))

    def sample_locations(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample ``n`` cell indices proportional to population weights (not implemented)."""
        raise NotImplementedError

    def cell_to_metres(self, row: int, col: int) -> Tuple[float, float]:
        """Map cell ``(row, col)`` to ``(x, y)`` metres from the grid origin (not implemented)."""
        raise NotImplementedError
