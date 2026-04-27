"""Per-store tabular Q-learning agent with CEO-set strategy envelope.

Responsibility: implement a tabular Q-learning store agent using a relative
action space (ADR-005) with an independent per-store Q-table (ADR-004),
operating inside the GroupEnvelope assigned by the chain CEO.

Public API: StoreQLearner

Key dependencies: numpy, hotelling.envelope

References:
    ADR-004 (per-store independent Q-tables);
    ADR-005 (relative action space);
    Calvano et al. (2020 AER) §III.
"""
from __future__ import annotations

import numpy as np

from hotelling.envelope import GroupEnvelope


class StoreQLearner:
    """Tabular Q-learning agent for a single physical store.

    Parameters
    ----------
    store_id : str
    group_labels : dict[str, str]
        Mapping from division name to category label for this store, e.g.
        ``{"DIVISION_COMPETITION": "HEAVY", "DIVISION_NEIGHBOURHOOD": "RICH"}``.
    alpha : float  Learning rate.
    gamma : float  Discount factor (fixed globally — see ADR-008).
    n_price_moves : int  Relative price action levels (default 7).
    n_effort_moves : int  Relative effort action levels (default 3).
    """

    def __init__(
        self,
        store_id: str,
        group_labels: dict[str, str],
        alpha: float,
        gamma: float,
        n_price_moves: int = 7,
        n_effort_moves: int = 3,
    ) -> None:
        self.store_id = store_id
        self.group_labels = group_labels
        self.alpha = alpha
        self.gamma = gamma
        self.n_price_moves = n_price_moves
        self.n_effort_moves = n_effort_moves

    def reset(self) -> None:
        """Initialise or reinitialise the Q-table to zeros."""
        raise NotImplementedError

    def act(self, state: dict, envelope: GroupEnvelope) -> tuple[int, int]:
        """Choose relative price and effort moves via ε-greedy policy."""
        raise NotImplementedError

    def update(
        self,
        state: dict,
        action: tuple[int, int],
        reward: float,
        next_state: dict,
    ) -> None:
        """Apply Bellman Q-update for the observed transition."""
        raise NotImplementedError

    def set_epsilon(self, epsilon: float) -> None:
        """Set the exploration rate (called by CEO at each epoch)."""
        raise NotImplementedError

    def get_qtable(self) -> np.ndarray:
        """Return the current Q-table array (shape: [n_states, n_actions])."""
        raise NotImplementedError

    def load_qtable(self, qtable: np.ndarray) -> None:
        """Replace the Q-table with an externally supplied array."""
        raise NotImplementedError
