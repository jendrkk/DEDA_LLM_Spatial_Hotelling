"""Unit tests for the entrant response function executor.

Verifies that the deterministic response function interpreter correctly
triggers price cuts on rival undercutting or profit distress, and that
all output prices are clipped to the current GroupEnvelope bounds.
"""
from __future__ import annotations

import pytest

from hotelling.agents.entrant_llm import EntrantLLM  # noqa: F401


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_rival_undercut_response_triggers_price_cut() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_rival_undercut_response_does_not_trigger_below_threshold() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_profit_distress_response_triggers_at_threshold() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_response_function_clips_to_envelope() -> None:
    pass
