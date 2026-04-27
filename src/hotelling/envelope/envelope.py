"""Group and chain strategy envelope dataclasses.

Responsibility: runtime data structures representing the CEO-set strategy
envelopes that bound Q-learning store agents each epoch.

Public API: GroupEnvelope, ChainEnvelope

Key dependencies: dataclasses

References: docs/agent_simulation_technical_report.md §4; ADR-009.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GroupEnvelope:
    """Strategy parameters set by the CEO for one store group.

    Attributes
    ----------
    p_bar : float  Target price midpoint (€).
    delta_p : float  Price half-width (€).
    e_bar : float  Target effort midpoint [0, 1].
    delta_e : float  Effort half-width.
    epsilon : float  RL exploration rate for stores in this group.
    """

    p_bar: float
    delta_p: float
    e_bar: float
    delta_e: float
    epsilon: float


@dataclass
class ChainEnvelope:
    """Full envelope output for one chain at one CEO epoch.

    Attributes
    ----------
    groups : dict[str, GroupEnvelope]
        Composite group label → envelope; e.g. ``{"HEAVY_RICH": GroupEnvelope(...)}``.
    """

    groups: dict[str, GroupEnvelope] = field(default_factory=dict)

    def get_group(self, store_id: str) -> GroupEnvelope:
        """Return the GroupEnvelope for the group containing store_id."""
        raise NotImplementedError
