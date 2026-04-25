"""
Q-learning agent for the Hotelling simulation.

:class:`QLearningAgent` discretises the location/price space into a grid of
states and uses tabular Q-learning to learn a pricing and location strategy
that maximises cumulative profit.

The state is the agent's own position (grid cell) and the positions/prices of
competitors as observed at the end of each period.  The action space consists
of *move direction* (stay, N, S, E, W) combined with a discrete *price delta*
(lower, keep, raise).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hotelling.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIRECTIONS: List[Tuple[float, float]] = [
    (0.0, 0.0),   # stay
    (0.0, 1.0),   # north
    (0.0, -1.0),  # south
    (1.0, 0.0),   # east
    (-1.0, 0.0),  # west
]
_PRICE_DELTAS: List[float] = [-0.05, 0.0, 0.05]
_N_ACTIONS = len(_DIRECTIONS) * len(_PRICE_DELTAS)  # 15


def _encode_action(dir_idx: int, price_idx: int) -> int:
    return dir_idx * len(_PRICE_DELTAS) + price_idx


def _decode_action(action: int) -> Tuple[int, int]:
    return divmod(action, len(_PRICE_DELTAS))


class QLearningAgent(BaseAgent):
    """Tabular Q-learning market actor.

    Parameters
    ----------
    firm_id:
        Unique identifier.
    location:
        Initial *(x, y)* position.
    price:
        Initial price.
    n_bins:
        Number of bins along each spatial axis for state discretisation.
    alpha:
        Learning rate.
    gamma:
        Discount factor.
    epsilon:
        Initial ε-greedy exploration rate.
    epsilon_decay:
        Multiplicative decay applied to ``epsilon`` each step.
    epsilon_min:
        Floor for the exploration rate.
    step_size:
        Physical distance moved per directional action.
    seed:
        Random seed.
    """

    def __init__(
        self,
        firm_id: str,
        location: Tuple[float, float] = (0.5, 0.5),
        price: float = 1.0,
        name: Optional[str] = None,
        n_bins: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        step_size: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(firm_id=firm_id, location=location, price=price, name=name)
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.step_size = step_size
        self._rng = np.random.default_rng(seed)
        # Q-table: state is a tuple of (x_bin, y_bin), action is int
        self._q: Dict[Tuple[int, int], np.ndarray] = {}
        self._last_state: Optional[Tuple[int, int]] = None
        self._last_action: Optional[int] = None

    # ------------------------------------------------------------------
    # State / action helpers
    # ------------------------------------------------------------------

    def _discretise(self, x: float, y: float, width: float, height: float) -> Tuple[int, int]:
        xb = int(np.clip(x / width * self.n_bins, 0, self.n_bins - 1))
        yb = int(np.clip(y / height * self.n_bins, 0, self.n_bins - 1))
        return xb, yb

    def _get_q(self, state: Tuple[int, int]) -> np.ndarray:
        if state not in self._q:
            self._q[state] = np.zeros(_N_ACTIONS)
        return self._q[state]

    def _choose_action(self, state: Tuple[int, int]) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, _N_ACTIONS))
        return int(np.argmax(self._get_q(state)))

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def observe(self, state: Dict[str, Any]) -> None:
        """Update Q-table using the reward received since the last action."""
        super().observe(state)
        reward = state.get("reward", self.profit)
        city = state["city"]
        x, y = self.location
        new_state = self._discretise(x, y, city["width"], city["height"])

        if self._last_state is not None and self._last_action is not None:
            old_q = self._get_q(self._last_state)
            new_q = self._get_q(new_state)
            old_q[self._last_action] += self.alpha * (
                reward + self.gamma * np.max(new_q) - old_q[self._last_action]
            )

        self._last_state = new_state
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        city = state["city"]
        x, y = self.location
        disc_state = self._discretise(x, y, city["width"], city["height"])
        action = self._choose_action(disc_state)
        self._last_action = action
        self._last_state = disc_state

        dir_idx, price_idx = _decode_action(action)
        dx, dy = _DIRECTIONS[dir_idx]
        price_delta = _PRICE_DELTAS[price_idx]

        new_x = float(np.clip(x + dx * self.step_size, 0.0, city["width"]))
        new_y = float(np.clip(y + dy * self.step_size, 0.0, city["height"]))
        new_price = float(np.clip(self.price + price_delta, 0.01, None))

        return {"location": (new_x, new_y), "price": new_price}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the Q-table to a JSON file."""
        serialisable = {str(k): v.tolist() for k, v in self._q.items()}
        with open(path, "w") as fh:
            json.dump({"q_table": serialisable, "epsilon": self.epsilon}, fh, indent=2)

    def load(self, path: str | Path) -> None:
        """Load a previously saved Q-table."""
        with open(path) as fh:
            data = json.load(fh)
        self._q = {
            tuple(int(v) for v in k.strip("()").split(",")): np.array(v)  # type: ignore[misc]
            for k, v in data["q_table"].items()
        }
        self.epsilon = data.get("epsilon", self.epsilon)
