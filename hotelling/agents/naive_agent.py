"""
Naive (rule-based) agent for the Hotelling simulation.

:class:`NaiveAgent` implements simple heuristics that do not require
learning or external APIs:

* **"center"** – always move towards the city centre (principle of minimum
  differentiation).
* **"random_walk"** – take a small random step each period.
* **"stay"** – never move or change price.
"""

from __future__ import annotations

import numpy as np

from typing import Any, Dict, Optional, Tuple

from hotelling.agents.base_agent import BaseAgent


class NaiveAgent(BaseAgent):
    """Rule-based market actor.

    Parameters
    ----------
    firm_id:
        Unique identifier.
    location:
        Initial *(x, y)* position.
    price:
        Initial price.
    strategy:
        One of ``"center"``, ``"random_walk"``, or ``"stay"``.
    step_size:
        Maximum displacement per period (used by ``"random_walk"``).
    seed:
        Random seed for reproducibility.
    """

    STRATEGIES = ("center", "random_walk", "stay")

    def __init__(
        self,
        firm_id: str,
        location: Tuple[float, float] = (0.5, 0.5),
        price: float = 1.0,
        name: Optional[str] = None,
        strategy: str = "center",
        step_size: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(firm_id=firm_id, location=location, price=price, name=name)
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self.step_size = step_size
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        city = state["city"]
        cx, cy = city["width"] / 2, city["height"] / 2
        x, y = self.location

        if self.strategy == "center":
            # Move a step_size fraction towards the centre
            dx = (cx - x) * self.step_size
            dy = (cy - y) * self.step_size
            new_loc = (
                float(np.clip(x + dx, 0.0, city["width"])),
                float(np.clip(y + dy, 0.0, city["height"])),
            )
        elif self.strategy == "random_walk":
            delta = self._rng.uniform(-self.step_size, self.step_size, size=2)
            new_loc = (
                float(np.clip(x + delta[0], 0.0, city["width"])),
                float(np.clip(y + delta[1], 0.0, city["height"])),
            )
        else:  # "stay"
            new_loc = self.location

        return {"location": new_loc, "price": self.price}
