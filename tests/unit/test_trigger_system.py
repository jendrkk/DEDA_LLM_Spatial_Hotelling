"""Unit tests for the entrant reassessment trigger system.

Verifies that TimeTrigger fires exactly at the configured interval, that
ProfitDropTrigger fires when rolling profit breaches the threshold, and that
RivalEventTrigger fires on a rival price change above the configured magnitude.
"""
from __future__ import annotations

import pytest

from hotelling.simulation.triggers import (  # noqa: F401
    ProfitDropTrigger,
    RivalEventTrigger,
    TimeTrigger,
)


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_time_trigger_fires_at_interval() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_time_trigger_does_not_fire_before_interval() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_profit_drop_trigger_fires_on_threshold_breach() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_rival_event_trigger_fires_on_price_change() -> None:
    pass
