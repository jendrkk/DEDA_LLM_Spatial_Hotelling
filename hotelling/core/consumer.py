"""
Consumer model for the Hotelling spatial competition simulation.

Each consumer is located at a fixed point in the city and purchases from the
firm that minimises her total cost (price + transport cost).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from hotelling.core.firm import Firm


@dataclass
class Consumer:
    """A single consumer located at a point in 2-D space.

    Parameters
    ----------
    location:
        *(x, y)* coordinates of the consumer.
    transport_cost:
        Per-unit-distance cost of travelling to a firm (linear transport cost).
    """

    location: Tuple[float, float]
    transport_cost: float = 1.0

    def distance_to(self, x: float, y: float) -> float:
        """Euclidean distance from consumer to a point *(x, y)*."""
        dx = self.location[0] - x
        dy = self.location[1] - y
        return float(np.sqrt(dx**2 + dy**2))

    def total_cost(self, firm: "Firm") -> float:
        """Full delivered price: firm price + transport cost."""
        dist = self.distance_to(*firm.location)
        return firm.price + self.transport_cost * dist

    def preferred_firm(self, firms: list["Firm"]) -> Optional["Firm"]:
        """Return the firm with the lowest total cost.

        Returns ``None`` if *firms* is empty.
        """
        if not firms:
            return None
        return min(firms, key=self.total_cost)
