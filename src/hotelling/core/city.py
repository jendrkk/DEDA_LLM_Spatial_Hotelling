"""2-D city container with population grid and distance matrix.

Responsibility: hold spatial market geometry and the firms located in it.

Public API: City

Key dependencies: numpy, hotelling.core.firm

References: Anderson, de Palma, Thisse (1992) Ch.3; Tabuchi (1994).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from hotelling.core.firm import Firm


@dataclass
class City:
    """2-D spatial market container.

    Parameters
    ----------
    boundary: Tuple[float, float, float, float] - (xmin, ymin, xmax, ymax) in metres
    population_grid: Optional[np.ndarray] - 2-D population density array (H×W)
    firms: List[Firm] - firms currently in the market
    """

    boundary: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    population_grid: Optional[np.ndarray] = field(default=None, repr=False)
    firms: List[Firm] = field(default_factory=list)

    @property
    def width(self) -> float:
        """Horizontal extent of the market space."""
        return self.boundary[2] - self.boundary[0]

    @property
    def height(self) -> float:
        """Vertical extent of the market space."""
        return self.boundary[3] - self.boundary[1]

    @property
    def center(self) -> Tuple[float, float]:
        """Geometric center of the city."""
        return (self.boundary[0] + self.width / 2, self.boundary[1] + self.height / 2)

    @property
    def area(self) -> float:
        """Area of the city."""
        return self.width * self.height
