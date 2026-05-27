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
        m_effort: int = 5,
        e_max: float = 10.0,
        k_neighbors: int = 1,
        transport_cost: float = 0.01,
    ) -> None:
        self.city = city
        self.firms = firms
        self.m = m
        self.m_effort = m_effort
        self.e_max = e_max
        self.k_neighbors = k_neighbors
        self.transport_cost = transport_cost

        # Price grid
        mc_min = min(getattr(f, "marginal_cost", 0.0) for f in firms)
        self._min_price = min_price if min_price is not None else mc_min
        self._max_price = max_price if max_price is not None else max(2.0 * self._min_price, 2.0)
        self.price_grid: np.ndarray = np.linspace(self._min_price, self._max_price, self.m)

        # Effort grid: [0, e_max] with m_effort levels
        self.effort_grid: np.ndarray = np.linspace(0.0, e_max, m_effort)

        # Joint action space size
        self._action_size = m * m_effort

        # Agent lists
        self.possible_agents: list[str] = [str(f.id) for f in firms]
        self.agents: list[str] = list(self.possible_agents)

        # Starting joint action: mid price × zero effort
        mid_price_idx = m // 2
        mid_effort_idx = 0
        mid_joint = mid_price_idx * m_effort + mid_effort_idx
        self._current_joint_actions: dict[str, int] = {
            a: mid_joint for a in self.possible_agents
        }

        # Precompute k-nearest-neighbor lists from firm locations (EPSG:3035 metres)
        self._neighbor_indices: dict[str, list[int]] = self._build_neighbor_indices()

    def _build_neighbor_indices(self) -> dict[str, list[int]]:
        """Return k nearest neighbor indices (into self.firms) for each agent.

        Uses Euclidean distance between firm.location tuples (EPSG:3035 metres).
        A firm is never its own neighbor (self excluded).
        If there are fewer than k other firms, all other firms are used.

        Returns
        -------
        dict mapping agent_id str → list of k int indices into self.firms
        """
        n = len(self.firms)
        if n == 0:
            return {}

        # Build (n, 2) array of firm locations in metres
        locations = np.array([f.location for f in self.firms], dtype=np.float64)  # (n, 2)

        result: dict[str, list[int]] = {}
        for i, firm in enumerate(self.firms):
            # Euclidean distances from firm i to all others
            diffs = locations - locations[i]  # (n, 2)
            dists = np.hypot(diffs[:, 0], diffs[:, 1])  # (n,)
            dists[i] = np.inf  # exclude self

            k_actual = min(self.k_neighbors, n - 1)
            if k_actual == 0:
                neighbors = []
            else:
                neighbors = np.argsort(dists)[:k_actual].tolist()
            result[str(firm.id)] = neighbors

        return result

    def _build_observation(self, agent_id: str) -> dict:
        """Build the observation dict for one agent from current joint actions.

        Returns
        -------
        dict with:
            "own_prev_action"       : int — own last joint action index
            "neighbor_prev_actions" : list of int — k neighbors' last joint action indices
        """
        own = self._current_joint_actions[agent_id]
        neighbor_idxs = self._neighbor_indices.get(agent_id, [])
        neighbor_firm_ids = [str(self.firms[idx].id) for idx in neighbor_idxs]
        neighbor_actions = [
            self._current_joint_actions.get(nid, self._action_size // 2)
            for nid in neighbor_firm_ids
        ]
        # Pad with mid action if fewer neighbors than k (e.g. singleton market)
        while len(neighbor_actions) < self.k_neighbors:
            neighbor_actions.append(self._action_size // 2)
        return {
            "own_prev_action": own,
            "neighbor_prev_actions": neighbor_actions,
        }

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        """Reset the market to initial prices and return observations.

        Parameters
        ----------
        seed : optional int, currently unused (present for PettingZoo compat)

        Returns
        -------
        tuple (observations, infos) where observations is dict[agent_id → obs_dict]
        and infos is an empty dict.
        """
        # Reset all agents to mid-price, zero-effort starting action
        mid_price_idx = self.m // 2
        mid_effort_idx = 0
        mid_joint = mid_price_idx * self.m_effort + mid_effort_idx
        self._current_joint_actions = {a: mid_joint for a in self.possible_agents}
        self.agents = list(self.possible_agents)

        observations = {aid: self._build_observation(aid) for aid in self.agents}
        infos: dict = {}
        return observations, infos

    def step(self, actions: dict[str, int]) -> tuple[dict, dict, dict, dict, dict]:
        """Advance one period: decode actions, compute demand + profits, return.

        Parameters
        ----------
        actions : dict[agent_id str → int] — joint action indices chosen this period

        Returns
        -------
        5-tuple: (observations, rewards, terminations, truncations, infos)
        All dicts keyed by agent_id string.

        Notes
        -----
        Efforts are decoded from the joint action index.
        market_clearing is called with decoded prices and efforts arrays,
        maintaining the same firm ordering as self.firms.
        """
        from hotelling.core.market import market_clearing  # lazy to avoid circular import

        N = len(self.firms)
        prices = np.zeros(N, dtype=np.float64)
        efforts = np.zeros(N, dtype=np.float64)

        for i, firm in enumerate(self.firms):
            aid = str(firm.id)
            joint_idx = int(actions.get(aid, self._current_joint_actions[aid]))
            price_idx = joint_idx // self.m_effort
            effort_idx = joint_idx % self.m_effort
            # Clamp indices to valid range
            price_idx = max(0, min(price_idx, self.m - 1))
            effort_idx = max(0, min(effort_idx, self.m_effort - 1))
            prices[i] = self.price_grid[price_idx]
            efforts[i] = self.effort_grid[effort_idx]
            # Update current joint action record
            self._current_joint_actions[aid] = price_idx * self.m_effort + effort_idx

        demands, profits = market_clearing(
            prices=prices,
            efforts=efforts,
            city=self.city,
            transport_cost=self.transport_cost,
        )

        observations = {aid: self._build_observation(aid) for aid in self.agents}
        rewards = {str(f.id): float(profits[i]) for i, f in enumerate(self.firms)}
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: False for aid in self.agents}
        infos = {
            str(f.id): {
                "demand": float(demands[i]),
                "price": float(prices[i]),
                "effort": float(efforts[i]),
            }
            for i, f in enumerate(self.firms)
        }
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent: str) -> Any:
        """Return the observation space for one agent.

        Returns a gymnasium.spaces.Dict if gymnasium is available,
        otherwise returns a plain dict describing the space.
        """
        try:
            import gymnasium
            from gymnasium.spaces import Dict, Discrete, MultiDiscrete

            joint_space = self.m * self.m_effort
            return Dict({
                "own_prev_action": Discrete(joint_space),
                "neighbor_prev_actions": MultiDiscrete([joint_space] * self.k_neighbors),
            })
        except ImportError:
            joint_space = self.m * self.m_effort
            return {
                "own_prev_action": {"type": "Discrete", "n": joint_space},
                "neighbor_prev_actions": {
                    "type": "MultiDiscrete",
                    "nvec": [joint_space] * self.k_neighbors,
                },
            }

    def action_space(self, agent: str) -> Any:
        """Return the action space for one agent.

        The action space is Discrete(m_price * m_effort): each integer encodes
        a joint (price_idx, effort_idx) pair via:
            joint_action_idx = price_idx * m_effort + effort_idx

        Returns a gymnasium.spaces.Discrete if gymnasium is available,
        otherwise returns a plain dict.
        """
        try:
            import gymnasium
            from gymnasium.spaces import Discrete

            return Discrete(self.m * self.m_effort)
        except ImportError:
            return {"type": "Discrete", "n": self.m * self.m_effort}
