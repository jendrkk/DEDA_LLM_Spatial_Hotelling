"""OPTIONAL: Stable-Baselines3 DQN wrapper for n>=5 firms.

Responsibility: provide a deep Q-network agent for large state spaces where
tabular Q-learning is infeasible.

Public API: DeepQAgent

Key dependencies: stable-baselines3 (optional import)

Note: This module is optional. stable-baselines3 must be installed separately.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from hotelling.agents.base import Action, Observation, Transition


class DeepQAgent:
    """SB3 DQN wrapper for large n scenarios. Requires stable-baselines3."""

    def __init__(
        self,
        firm_id: str,
        m: int = 15,
        policy: str = "MlpPolicy",
        **sb3_kwargs: Any,
    ) -> None:
        self.firm_id = firm_id
        self.m = m
        self._model: Any = None

    def reset(self, info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def act(self, observation: Observation) -> Action:
        raise NotImplementedError

    def update(self, transition: Transition) -> None:
        raise NotImplementedError
