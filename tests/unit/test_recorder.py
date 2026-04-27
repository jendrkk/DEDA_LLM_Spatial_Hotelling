"""Unit tests for hotelling.simulation – SimulationRecorder."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hotelling.simulation.recorder import SimulationRecorder


class TestSimulationRecorder:
    def test_init_creates_run_dir(self, tmp_path: Path):
        run_dir = tmp_path / "run_001"
        rec = SimulationRecorder(run_dir=run_dir)
        assert run_dir.exists()

    def test_auto_run_id(self, tmp_path: Path):
        rec = SimulationRecorder(run_dir=tmp_path)
        assert len(rec.run_id) > 0

    def test_custom_run_id(self, tmp_path: Path):
        rec = SimulationRecorder(run_dir=tmp_path, run_id="my-run-123")
        assert rec.run_id == "my-run-123"

    def test_record_step_buffered(self, tmp_path: Path):
        rec = SimulationRecorder(run_dir=tmp_path)
        rec.record_step(period=0, agent_id="firm_0", price=1.5, demand=0.5, profit=0.25)
        assert len(rec._buffer) == 1

    def test_flush_writes_parquet(self, tmp_path: Path):
        rec = SimulationRecorder(run_dir=tmp_path, run_id="flush-test")
        rec.record_step(period=0, agent_id="firm_0", price=1.5, demand=0.5, profit=0.25)
        rec.record_step(period=0, agent_id="firm_1", price=1.6, demand=0.5, profit=0.30)
        out = rec.flush()
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == 2
        assert set(df.columns) >= {"run_id", "period", "agent_id", "price", "demand", "profit"}

    def test_flush_clears_buffer(self, tmp_path: Path):
        rec = SimulationRecorder(run_dir=tmp_path)
        rec.record_step(period=0, agent_id="A", price=1.0, demand=0.5, profit=0.5)
        rec.flush()
        assert len(rec._buffer) == 0

    def test_flush_empty_returns_path(self, tmp_path: Path):
        rec = SimulationRecorder(run_dir=tmp_path, run_id="empty")
        out = rec.flush()
        assert isinstance(out, Path)

    def test_extra_columns_stored(self, tmp_path: Path):
        rec = SimulationRecorder(run_dir=tmp_path, run_id="extra-cols")
        rec.record_step(
            period=1, agent_id="A", price=1.2, demand=0.4, profit=0.08, price_index=3
        )
        out = rec.flush()
        df = pd.read_parquet(out)
        assert "price_index" in df.columns

    def test_close_flushes(self, tmp_path: Path):
        rec = SimulationRecorder(run_dir=tmp_path, run_id="close-test")
        rec.record_step(period=0, agent_id="A", price=1.0, demand=0.5, profit=0.5)
        rec.close()
        out = tmp_path / "close-test.parquet"
        assert out.exists()
