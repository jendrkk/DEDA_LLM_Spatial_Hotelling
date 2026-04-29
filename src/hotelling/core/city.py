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

    boundary: Tuple[float, float, float, float]
    population_grid: Optional[np.ndarray]
    firms: List[Firm]
    
    # precomputed once from population_grid + firms:
    dist2_km2: np.ndarray        # (M, N) — squared network distances, Parquet-cached
    cell_pop: np.ndarray         # (M,)
    lambda_phi: np.ndarray       # (M,)
    pi_H: np.ndarray             # (M,)
    pi_H_lambda_phi: np.ndarray  # (M,)
    
    # model parameters (from Hydra config, attached at env init):
    alpha: np.ndarray            # (2,) — [α_L, α_H]
    beta: float
    mu: float = 0.25
    a0: float = 0.0
    
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

