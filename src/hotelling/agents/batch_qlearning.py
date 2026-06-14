"""Vectorized batch Q-learning.

Holds N INDEPENDENT per-store Q-tables in a single (N, state_size, action_size)
ndarray, stacked along axis 0 purely for vectorized indexing/updates — this is
NOT a shared Q-table. Store i reads and writes only _q[i]; its TD update uses
only its own state, action, and reward. Equivalent to N separate QLearningAgent
instances (ADR-004, per-store independent Q-tables); vectorized for speed at
N~494 stores. See ADR-004.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BatchQLearningAgent:
    """Tabular Q-learning for N symmetric agents with vectorized updates."""

    def __init__(
        self,
        n_agents: int,
        m: int,
        m_effort: int,
        k: int,
        alpha: float,
        beta_decay: float,
        delta: float,
        seed: int | None = None,
        max_qtable_gib: float = 8.0,
        state_mode: str = "neighbors",
        state_size: int | None = None,
    ) -> None:
        self.n = n_agents
        self.m = m
        self.m_effort = m_effort
        self.k = k
        self.alpha = alpha
        self.beta_decay = beta_decay
        self.delta = delta
        self.state_mode = state_mode
        self.action_size = m * m_effort
        self.state_size = (
            int(state_size) if state_size is not None else self.action_size ** k
        )

        n_qvals = n_agents * self.state_size * self.action_size
        gib = n_qvals * 8 / (1024**3)
        logger.info(
            "Q-table: %d agents x %d states x %d actions = %.3g values "
            "(%.2f GiB, k=%d, action_size=%d)",
            n_agents,
            self.state_size,
            self.action_size,
            n_qvals,
            gib,
            k,
            self.action_size,
        )
        if gib > max_qtable_gib:
            raise MemoryError(
                f"Q-table would need {gib:.1f} GiB > max_qtable_gib={max_qtable_gib} "
                f"(k={k}, action_size={self.action_size}, states={self.state_size}). "
                f"Reduce k_neighbors, freeze effort (m_effort=1), or raise "
                f"max_qtable_gib explicitly."
            )

        self._rng = np.random.default_rng(seed)
        self._t = np.zeros(n_agents, dtype=np.int64)
        # Axis 0 indexes stores: _q[i] is store i's OWN independent Q-table (ADR-004).
        # Stacked into one ndarray only for vectorized act()/update(); no parameter sharing.
        self._q = np.zeros(
            (n_agents, self.state_size, self.action_size),
            dtype=np.float64,
        )

    def reset(self) -> None:
        """Reset Q-table and exploration counters."""
        self._q[:] = 0.0
        self._t[:] = 0

    def _encode_states(self, signal: np.ndarray) -> np.ndarray:
        """Encode state signal → flat state indices (N,)."""
        if self.state_mode == "local_summary":
            s = np.asarray(signal, dtype=np.int64)
            assert s.ndim == 1 and s.shape[0] == self.n, (
                "local_summary signal must be (N,)"
            )
            return s
        neighbor_actions = signal
        if self.k == 1:
            return neighbor_actions[:, 0].astype(np.int64)
        multipliers = self.action_size ** np.arange(self.k, dtype=np.int64)
        return (neighbor_actions.astype(np.int64) * multipliers[None, :]).sum(axis=1)

    def act(self, neighbor_actions: np.ndarray) -> np.ndarray:
        """Choose joint action indices for all agents.

        Parameters
        ----------
        neighbor_actions : (N, k) int array of neighbors' joint action indices

        Returns
        -------
        (N,) int array of chosen joint action indices
        """
        states = self._encode_states(neighbor_actions)
        epsilons = np.exp(-self.beta_decay * self._t)

        q_rows = self._q[np.arange(self.n), states]
        noise = self._rng.random((self.n, self.action_size)) * 1e-10
        greedy_actions = np.argmax(q_rows + noise, axis=1)

        random_actions = self._rng.integers(0, self.action_size, size=self.n)
        explore_mask = self._rng.random(self.n) < epsilons
        actions = np.where(explore_mask, random_actions, greedy_actions).astype(np.int64)

        self._t += 1
        return actions

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        """Vectorized TD update for all N agents."""
        idx_n = np.arange(self.n)
        current_q = self._q[idx_n, states, actions]
        best_next = self._q[idx_n, next_states].max(axis=1)
        td_error = rewards + self.delta * best_next - current_q
        self._q[idx_n, states, actions] += self.alpha * td_error

    @property
    def epsilon_mean(self) -> float:
        """Mean exploration probability across agents."""
        return float(np.exp(-self.beta_decay * self._t).mean())
