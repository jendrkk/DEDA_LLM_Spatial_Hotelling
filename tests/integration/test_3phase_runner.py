"""Integration tests for the full three-phase simulation runner.

Verifies end-to-end execution: that a full run completes with a mock LLM
backend, that the CEO is called at the correct T_CEO frequency during Phase 2,
and that Parquet output files are written after Phase 2 completes.
"""
from __future__ import annotations

import pytest

from hotelling.simulation.phases import Phase0BurnIn, Phase1Entry, Phase2StrategicGame  # noqa: F401


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_full_run_completes_with_mock_llm() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_phase2_ceo_called_at_correct_frequency() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_parquet_output_written_after_phase2() -> None:
    pass
