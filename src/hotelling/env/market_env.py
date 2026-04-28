"""PettingZoo ParallelEnv wrapper for the spatial Hotelling market.

Responsibility: implement the core simulation loop as a PettingZoo
ParallelEnv; manage per-period price/effort decisions, demand clearing,
and reward computation.

Public API: HotellingMarketEnv

Key dependencies: pettingzoo, numpy, hotelling.core

References: ADR-003; docs/agent_simulation_technical_report.md §3.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class HotellingMarketEnv:
    """PettingZoo-compatible Hotelling market environment.

    Parameters
    ----------
    city : object  City object with boundary and firms attributes.
    firms : list  List of Firm objects.
    m : int  Number of discrete price levels.
    min_price : float | None  Lowest price level; defaults to min marginal cost.
    max_price : float | None  Highest price level; defaults to 2 * min_price.
    """

    def __init__(
        self,
        city: Any,
        firms: list,
        m: int = 15,
        min_price: float | None = None,
        max_price: float | None = None,
    ) -> None:
        self.city = city
        self.firms = firms
        self.m = m

        mc_min = min(getattr(f, "marginal_cost", 1.0) for f in firms)
        self._min_price = min_price if min_price is not None else mc_min
        self._max_price = max_price if max_price is not None else 2.0 * self._min_price

        self.price_grid: np.ndarray = np.linspace(
            self._min_price, self._max_price, self.m
        )
        self.possible_agents: list[str] = [str(f.id) for f in firms]
        self.agents: list[str] = list(self.possible_agents)

        mid = self.m // 2
        self._current_prices: dict[str, int] = {a: mid for a in self.possible_agents}

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        """Reset the environment and return initial observations and infos."""
        raise NotImplementedError

    def step(self, actions: dict[str, int]) -> tuple[dict, dict, dict, dict, dict]:
        """Advance one period; return obs, rewards, terminations, truncations, infos."""
        raise NotImplementedError

    def observation_space(self, agent: str) -> Any:
        """Return the observation space for agent."""
        raise NotImplementedError

    def action_space(self, agent: str) -> Any:
        """Return the action space for agent."""
        raise NotImplementedError
