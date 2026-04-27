"""Entrant reassessment trigger system.

Responsibility: define the Trigger interface and baseline concrete triggers
(time-based, profit-drop, rival-event) that decide when the entrant LLM
should reassess its response function.

Public API: Trigger, TimeTrigger, ProfitDropTrigger, RivalEventTrigger

Key dependencies: abc

References: ADR-011; docs/agent_simulation_technical_report.md §7.3.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Trigger(ABC):
    """Abstract base for entrant reassessment triggers."""

    @abstractmethod
    def should_fire(self, entrant_state: dict, period: int) -> bool:
        """Return True if a reassessment should be triggered this period."""
        raise NotImplementedError


class TimeTrigger(Trigger):
    """Fires every ``period_interval`` periods unconditionally.

    Parameters
    ----------
    period_interval : int  Reassessment frequency (T_entrant in config).
    """

    def __init__(self, period_interval: int) -> None:
        self.period_interval = period_interval

    def should_fire(self, entrant_state: dict, period: int) -> bool:
        raise NotImplementedError


class ProfitDropTrigger(Trigger):
    """Fires when rolling profit falls more than threshold_pct relative to prior window.

    Parameters
    ----------
    threshold_pct : float  Relative profit drop fraction that triggers reassessment.
    window : int  Rolling window length in periods.
    """

    def __init__(self, threshold_pct: float, window: int) -> None:
        self.threshold_pct = threshold_pct
        self.window = window

    def should_fire(self, entrant_state: dict, period: int) -> bool:
        raise NotImplementedError


class RivalEventTrigger(Trigger):
    """Fires when a nearby rival's price changes beyond threshold or a rival opens/closes.

    Parameters
    ----------
    price_change_threshold : float  Rival price-change magnitude (€) that triggers.
    radius : float  Search radius (metres, EPSG:25833) for rival events.
    """

    def __init__(self, price_change_threshold: float, radius: float) -> None:
        self.price_change_threshold = price_change_threshold
        self.radius = radius

    def should_fire(self, entrant_state: dict, period: int) -> bool:
        raise NotImplementedError
