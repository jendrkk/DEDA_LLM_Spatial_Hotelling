"""Integration tests for Phase 0 burn-in: incumbent Q-table convergence.

Verifies that running Phase0BurnIn populates all store Q-tables (no zero-filled
tables after burn-in) and that the convergence criterion terminates the phase
before T_burnin is exhausted when agents have converged.
"""
from __future__ import annotations

import pytest

from hotelling.simulation.phases import Phase0BurnIn  # noqa: F401


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_burnin_populates_qtables() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_burnin_convergence_criterion_triggers() -> None:
    pass
