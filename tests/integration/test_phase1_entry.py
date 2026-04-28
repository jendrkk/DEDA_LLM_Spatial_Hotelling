"""Integration tests for Phase 1 entry: entrant LLM decision and Q-table init.

Verifies that the entry decision produces a valid chain type from {D, S, B},
that the chosen location index is within the candidate site list, and that the
two main Q-table initialisation strategies (BLANK, INHERIT_ALGORITHM) produce
the expected initial Q-table state.
"""
from __future__ import annotations

import pytest

from hotelling.agents.entrant_llm import EntrantLLM  # noqa: F401
from hotelling.simulation.phases import Phase1Entry  # noqa: F401


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_entry_decision_produces_valid_chain_type() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_entry_decision_location_in_candidate_list() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_qtable_init_blank_produces_zero_table() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_qtable_init_inherit_algorithm_copies_valid_table() -> None:
    pass
