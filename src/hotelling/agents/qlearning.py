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
        m_effort: int = 5,
        e_max: float = 10.0,
        k: int = 1,
        alpha: float = 0.10,
        beta_decay: float = 2e-5,
        delta: float = 0.95,
        mu: float = 0.25,
        update_mode: str = "sync",
        seed: Optional[int] = None,
    ) -> None:
        self.firm_id = firm_id
        self.m = m
        self.m_effort = m_effort
        self.e_max = e_max
        self.k = k
        self.alpha = alpha
        self.beta_decay = beta_decay
        self.delta = delta
        self.mu = mu
        self.update_mode = update_mode
        self._rng = np.random.default_rng(seed)
        self._t = 0

        # Derived
        self._action_size = m * m_effort
        self._state_size = (m * m_effort) ** k
        self._q_table: Optional[np.ndarray] = None

    def _encode_state(self, neighbor_actions: list) -> int:
        """Encode k neighbors' joint action indices into a single state integer.

        Uses mixed-radix encoding:
            state = sum( neighbor_actions[i] * action_size^i  for i in 0..k-1 )

        For k=1: state = neighbor_actions[0]  (state space = action_size)
        For k=2: state = a0 + a1 * action_size  (state space = action_size^2)

        Parameters
        ----------
        neighbor_actions : list of k integers, each in [0, action_size-1]

        Returns
        -------
        int in [0, state_size-1]
        """
        state = 0
        for i, a in enumerate(neighbor_actions):
            state += int(a) * (self._action_size ** i)
        return state

    def _decode_action(self, joint_action_idx: int) -> tuple[int, int]:
        """Decode a joint action index to (price_idx, effort_idx).

        Encoding convention: joint_action_idx = price_idx * m_effort + effort_idx

        Parameters
        ----------
        joint_action_idx : integer in [0, m_price * m_effort - 1]

        Returns
        -------
        tuple (price_idx, effort_idx) where price_idx in [0, m-1],
            effort_idx in [0, m_effort-1]
        """
        price_idx = joint_action_idx // self.m_effort
        effort_idx = joint_action_idx % self.m_effort
        return price_idx, effort_idx

    def reset(self, info: Dict[str, Any]) -> None:
        """Initialize Q-table to zeros and reset exploration counter.

        Parameters
        ----------
        info : dict (unused; present for AgentProtocol compatibility)
        """
        self._q_table = np.zeros((self._state_size, self._action_size), dtype=np.float64)
        self._t = 0

    def act(self, observation: Observation) -> Action:
        """Choose a joint (price, effort) action via epsilon-greedy policy.

        Parameters
        ----------
        observation : dict with key "neighbor_prev_actions": list of k ints,
                      each being a neighbor's previous joint action index.

        Returns
        -------
        int — joint action index in [0, m_price * m_effort - 1]
        """
        if self._q_table is None:
            raise RuntimeError("Call reset() before act().")

        neighbor_actions = observation.get("neighbor_prev_actions", [0] * self.k)
        state = self._encode_state(neighbor_actions)

        if self._rng.random() < self.epsilon:
            # Explore: uniform random joint action
            action = int(self._rng.integers(0, self._action_size))
        else:
            # Exploit: greedy action (break ties randomly)
            q_row = self._q_table[state]
            max_q = q_row.max()
            best = np.flatnonzero(q_row == max_q)
            action = int(self._rng.choice(best))

        self._t += 1
        return action

    def update(self, transition: Transition) -> None:
        """Apply one Q-learning update step.

        Q(s, a) += alpha * [ r + delta * max_a' Q(s', a') - Q(s, a) ]

        Parameters
        ----------
        transition : dict with keys:
            "observation"      : dict (same structure as act() input)
            "action"           : int  (joint action index chosen)
            "reward"           : float (per-period profit)
            "next_observation" : dict (same structure as act() input)
        """
        if self._q_table is None:
            raise RuntimeError("Call reset() before update().")

        obs = transition["observation"]
        next_obs = transition["next_observation"]
        action = int(transition["action"])
        reward = float(transition["reward"])

        state = self._encode_state(obs.get("neighbor_prev_actions", [0] * self.k))
        next_state = self._encode_state(next_obs.get("neighbor_prev_actions", [0] * self.k))

        current_q = self._q_table[state, action]
        best_next = float(self._q_table[next_state].max())

        td_error = reward + self.delta * best_next - current_q
        self._q_table[state, action] += self.alpha * td_error

    @property
    def epsilon(self) -> float:
        """Current exploration probability: exp(-beta*t)."""
        return float(np.exp(-self.beta_decay * self._t))
