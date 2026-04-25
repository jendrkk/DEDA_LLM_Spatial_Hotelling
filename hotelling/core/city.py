"""
2-dimensional Hotelling city model.

The city defines the spatial arena in which firms compete and consumers are
distributed.  By default consumers are placed on a 2-D unit square, but the
class accepts an arbitrary convex polygon (e.g. derived from real GIS data) as
the market boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class City:
    """Represents the spatial market (the 'city') for a Hotelling model.

    Parameters
    ----------
    width:
        Horizontal extent of the market space (default 1.0).
    height:
        Vertical extent of the market space (default 1.0).
    n_consumers:
        Number of consumers uniformly distributed in the city.
    seed:
        Random seed for reproducibility.
    """

    width: float = 1.0
    height: float = 1.0
    n_consumers: int = 1000
    seed: Optional[int] = None

    # Populated after __post_init__
    consumer_locations: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.consumer_locations = rng.uniform(
            low=[0.0, 0.0],
            high=[self.width, self.height],
            size=(self.n_consumers, 2),
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def area(self) -> float:
        """Return the area of the (rectangular) market space."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Return the geometric center of the city."""
        return self.width / 2, self.height / 2

    def is_valid_location(self, x: float, y: float) -> bool:
        """Return True if *(x, y)* lies within the city boundaries."""
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height

    def random_location(self, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
        """Sample a uniformly random location inside the city."""
        if rng is None:
            rng = np.random.default_rng()
        x, y = rng.uniform([0.0, 0.0], [self.width, self.height])
        return float(x), float(y)

    def __repr__(self) -> str:
        return (
            f"City(width={self.width}, height={self.height}, "
            f"n_consumers={self.n_consumers})"
        )
