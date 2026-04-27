"""Tabular Q-learning agent.

Responsibility: implement a tabular Q-learning market agent with configurable
exploration schedule and synchronous/asynchronous Q-update modes.

Public API: QLearningAgent

Key dependencies: numpy

References:
    Calvano et al. (2020 AER) §III;
    Asker-Fershtman-Pakes (2022) async update variant.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from hotelling.agents.base import Action, Observation, Transition


class QLearningAgent:
    """Tabular Q-learning market actor.

    Parameters
    ----------
    firm_id: str
    m: int - number of discrete price levels (action space)
    k: int - number of competitor prices in state (default 1)
    alpha: float - learning rate
    beta: float - exploration decay rate (Calvano calibration)
    delta: float - discount factor
    mu: float - logit scale
    update_mode: str - "sync" or "async" (Asker-Fershtman-Pakes 2022)
    seed: Optional[int]
    """

    def __init__(
        self,
        firm_id: str,
        m: int = 15,
        k: int = 1,
        alpha: float = 0.10,
        beta: float = 2e-5,
        delta: float = 0.95,
        mu: float = 0.25,
        update_mode: str = "sync",
        seed: Optional[int] = None,
    ) -> None:
        self.firm_id = firm_id
        self.m = m
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.mu = mu
        self.update_mode = update_mode
        self._rng = np.random.default_rng(seed)
        self._q_table: Optional[np.ndarray] = None
        self._t: int = 0  # step counter for exploration schedule

    def reset(self, info: Dict[str, Any]) -> None:
        """Initialize Q-table from info dict (n_firms, price_grid)."""
        raise NotImplementedError

    def act(self, observation: Observation) -> Action:
        """Choose price index via epsilon-greedy policy."""
        raise NotImplementedError

    def update(self, transition: Transition) -> None:
        """Apply Q-learning update: Q(s,a) += alpha[r + delta*max Q(s',.) - Q(s,a)]."""
        raise NotImplementedError

    @property
    def epsilon(self) -> float:
        """Current exploration probability: exp(-beta*t)."""
        return float(np.exp(-self.beta * self._t))
