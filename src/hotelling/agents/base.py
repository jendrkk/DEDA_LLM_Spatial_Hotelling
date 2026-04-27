"""AgentProtocol: reset, act, update interface for all market agents.

Responsibility: define the minimal interface that all market agents must
satisfy, enabling duck-typed composition in the simulation engine.

Public API: AgentProtocol, Action, Observation, Transition

Key dependencies: typing
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

Action = Dict[str, Any]
Observation = Dict[str, Any]
Transition = Dict[str, Any]


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol that all market agents must satisfy."""

    def reset(self, info: Dict[str, Any]) -> None:
        """Reset agent state at the start of a new session."""
        ...

    def act(self, observation: Observation) -> Action:
        """Choose an action given the current market observation."""
        ...

    def update(self, transition: Transition) -> None:
        """Update agent internal state (e.g., Q-table) after a transition."""
        ...
