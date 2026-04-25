"""
Base agent interface.

All intelligent agents (LLM, Q-learning, naive) extend :class:`BaseAgent`,
which itself extends :class:`~hotelling.core.firm.Firm`.  The extra layer
exposes a unified ``observe`` / ``act`` interface used by the simulation
engine.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

from hotelling.core.firm import Firm


class BaseAgent(Firm):
    """Abstract base class for all intelligent market actors.

    Adds an *observation* buffer and a unified ``act`` interface on top of
    :class:`~hotelling.core.firm.Firm`.

    Parameters
    ----------
    firm_id:
        Unique identifier inherited from :class:`Firm`.
    location:
        Initial *(x, y)* market position.
    price:
        Initial price.
    name:
        Optional human-readable label.
    """

    def __init__(
        self,
        firm_id: str,
        location: Tuple[float, float] = (0.5, 0.5),
        price: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(firm_id=firm_id, location=location, price=price, name=name)
        self._history: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Observation / action interface
    # ------------------------------------------------------------------

    def observe(self, state: Dict[str, Any]) -> None:
        """Receive a market state observation and store it.

        Parameters
        ----------
        state:
            Dictionary containing at least ``"firms"`` (list of dicts with
            firm locations/prices/shares) and ``"step"`` (int).
        """
        self._history.append(state)

    @abstractmethod
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decide and return new ``{"location": (x, y), "price": float}``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Firm interface implementation
    # ------------------------------------------------------------------

    def decide_location(self, city, market) -> Tuple[float, float]:  # noqa: ANN001
        state = self._build_state(city, market)
        action = self.act(state)
        return action.get("location", self.location)

    def decide_price(self, city, market) -> float:  # noqa: ANN001
        # price is resolved together with location in decide_location via act()
        # Store last action for reuse; subclasses may cache this differently.
        return self._last_price if hasattr(self, "_last_price") else self.price

    def update(self, city, market) -> None:  # noqa: ANN001
        state = self._build_state(city, market)
        action = self.act(state)
        self.location = action.get("location", self.location)
        self._last_price = action.get("price", self.price)
        self.price = self._last_price

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_state(self, city, market) -> Dict[str, Any]:  # noqa: ANN001
        return {
            "step": len(self._history),
            "self": {
                "firm_id": self.firm_id,
                "location": self.location,
                "price": self.price,
                "market_share": self.market_share,
            },
            "firms": [
                {
                    "firm_id": f.firm_id,
                    "location": f.location,
                    "price": f.price,
                    "market_share": f.market_share,
                }
                for f in market.firms
                if f.firm_id != self.firm_id
            ],
            "city": {"width": city.width, "height": city.height},
        }

    @property
    def history(self) -> list[Dict[str, Any]]:
        """Read-only view of the observation history."""
        return list(self._history)
