"""
Market dynamics for the Hotelling spatial competition model.

The :class:`Market` class holds a collection of :class:`~hotelling.core.firm.Firm`
objects, resolves consumer choices given the current city state, and computes
per-firm market shares and profits.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from hotelling.core.city import City
from hotelling.core.consumer import Consumer
from hotelling.core.firm import Firm


class Market:
    """Manages firms and consumer assignment in the Hotelling model.

    Parameters
    ----------
    city:
        The :class:`~hotelling.core.city.City` instance defining the market.
    firms:
        Initial list of competing firms.
    transport_cost:
        Per-unit-distance transport cost applied to all consumers.
    """

    def __init__(
        self,
        city: City,
        firms: Optional[List[Firm]] = None,
        transport_cost: float = 1.0,
    ) -> None:
        self.city = city
        self.firms: List[Firm] = firms or []
        self.transport_cost = transport_cost
        self._consumers: List[Consumer] = self._build_consumers()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_consumers(self) -> List[Consumer]:
        return [
            Consumer(location=(float(x), float(y)), transport_cost=self.transport_cost)
            for x, y in self.city.consumer_locations
        ]

    def add_firm(self, firm: Firm) -> None:
        """Add a firm to the market."""
        self.firms.append(firm)

    def remove_firm(self, firm_id: str) -> None:
        """Remove a firm by its ID."""
        self.firms = [f for f in self.firms if f.firm_id != firm_id]

    # ------------------------------------------------------------------
    # Core market resolution
    # ------------------------------------------------------------------

    def resolve(self) -> Dict[str, float]:
        """Assign consumers to firms and compute market shares and profits.

        Returns
        -------
        dict
            Mapping ``{firm_id: market_share}`` for all firms.
        """
        if not self.firms:
            return {}

        # Count purchases per firm
        counts: Dict[str, int] = {f.firm_id: 0 for f in self.firms}
        revenues: Dict[str, float] = {f.firm_id: 0.0 for f in self.firms}

        for consumer in self._consumers:
            chosen = consumer.preferred_firm(self.firms)
            if chosen is not None:
                counts[chosen.firm_id] += 1
                revenues[chosen.firm_id] += chosen.price

        n_consumers = len(self._consumers)
        shares: Dict[str, float] = {}
        for firm in self.firms:
            share = counts[firm.firm_id] / n_consumers if n_consumers > 0 else 0.0
            firm.market_share = share
            firm.profit = revenues[firm.firm_id]
            shares[firm.firm_id] = share

        return shares

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of current market state."""
        lines = ["Market summary:", f"  City: {self.city}"]
        for firm in self.firms:
            lines.append(
                f"  {firm.name}: loc={firm.location}, "
                f"price={firm.price:.3f}, share={firm.market_share:.3%}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Market(n_firms={len(self.firms)}, city={self.city})"
