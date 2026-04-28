"""Unit tests for StoreQLearner relative action space behaviour (ADR-005).

Verifies that relative price moves are clipped correctly at envelope boundaries,
that the Q-table shape matches the expected state-action dimensions, and that
epsilon=0 produces a fully greedy policy with no exploration.
"""
from __future__ import annotations

import pytest

from hotelling.agents.store_rl import StoreQLearner  # noqa: F401


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_relative_action_clips_to_upper_envelope_boundary() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_relative_action_clips_to_lower_envelope_boundary() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_qtable_shape_matches_state_action_spec() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_epsilon_greedy_respects_epsilon_zero() -> None:
    pass
