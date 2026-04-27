"""Myopic best-response agent: computes Bertrand-Nash best response each period.

Responsibility: serve as the stage-game best-response baseline against which
Q-learning and LLM collusion can be measured.

Public API: MyopicAgent

Key dependencies: hotelling.agents.base

References: Calvano et al. (2020 AER) §III (baseline).
"""
from __future__ import annotations

from typing import Any, Dict

from hotelling.agents.base import Action, Observation, Transition


class MyopicAgent:
    """Stage-game best-response (Bertrand-Nash) baseline agent."""

    def __init__(
        self,
        firm_id: str,
        m: int = 15,
        mu: float = 0.25,
        transport_cost: float = 1.0,
    ) -> None:
        self.firm_id = firm_id
        self.m = m
        self.mu = mu
        self.transport_cost = transport_cost

    def reset(self, info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def act(self, observation: Observation) -> Action:
        """Return best-response price index given competitors' last prices."""
        raise NotImplementedError

    def update(self, transition: Transition) -> None:
        pass
