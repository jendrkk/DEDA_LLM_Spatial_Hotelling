"""Random agent: uniformly samples price index and location.

Responsibility: provide a uniformly random baseline agent for benchmarking.

Public API: RandomAgent

Key dependencies: numpy
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from hotelling.agents.base import Action, Observation, Transition


class RandomAgent:
    """Uniformly random agent that serves as a baseline."""

    def __init__(self, m: int = 15, seed: Optional[int] = None) -> None:
        self.m = m
        self._rng = np.random.default_rng(seed)

    def reset(self, info: Dict[str, Any]) -> None:
        pass

    def act(self, observation: Observation) -> Action:
        return {"price_index": int(self._rng.integers(0, self.m))}

    def update(self, transition: Transition) -> None:
        pass
