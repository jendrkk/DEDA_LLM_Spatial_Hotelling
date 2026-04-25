"""
Firm (market actor) base model for the Hotelling simulation.

Concrete agent types (LLM, Q-learning, naive) inherit from :class:`Firm` and
override :meth:`decide_location` and :meth:`decide_price`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Firm:
    """A firm competing in a 2-D Hotelling market.

    Parameters
    ----------
    firm_id:
        Unique identifier for this firm.
    location:
        Initial *(x, y)* position in the market.
    price:
        Current price charged to consumers.
    name:
        Optional human-readable label (defaults to ``firm_id``).
    """

    firm_id: str
    location: Tuple[float, float] = (0.5, 0.5)
    price: float = 1.0
    name: Optional[str] = None

    # Runtime statistics
    market_share: float = field(default=0.0, init=False, repr=False)
    profit: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.firm_id

    # ------------------------------------------------------------------
    # Decision interface – override in subclasses
    # ------------------------------------------------------------------

    def decide_location(self, city, market) -> Tuple[float, float]:  # noqa: ANN001
        """Return the next location for this firm.

        The default implementation returns the current location (no movement).
        Subclasses should override this method to implement strategic behaviour.
        """
        return self.location

    def decide_price(self, city, market) -> float:  # noqa: ANN001
        """Return the next price for this firm.

        The default implementation returns the current price.
        Subclasses should override this method to implement pricing strategies.
        """
        return self.price

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def update(self, city, market) -> None:  # noqa: ANN001
        """Convenience method: update both location and price in one call."""
        self.location = self.decide_location(city, market)
        self.price = self.decide_price(city, market)

    def __repr__(self) -> str:
        return (
            f"Firm(id={self.firm_id!r}, loc={self.location}, "
            f"price={self.price:.3f}, share={self.market_share:.3f})"
        )
