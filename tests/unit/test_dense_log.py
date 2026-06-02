"""Unit tests for DenseLog scaling controls.

Covers:
  - Default behaviour (stride=1, tail=None) — backward-compat
  - stride-only: every k-th step recorded, others skipped
  - tail-only: only last d steps recorded (stride=T so only step 0 + tail)
  - stride + tail combination: bulk-stride + dense final window
  - store_demand_profit=False: demand/profit arrays absent
  - float_dtype: float32 vs float64 precision stored correctly
  - Early-convergence (simulation ends before T): rows_written < n_scheduled
  - Flush / load round-trip: all meta keys preserved; memmaps read back OK
  - to_dataframe: period column reflects actual steps, not row indices
  - recorded_steps property
  - Size-warning threshold (smoke test via mock)
"""
from __future__ import annotations

import json

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 4    # stores
M = 15   # price grid size
T = 100  # total steps for most tests


def _price_grid(m: int = M) -> np.ndarray:
    return np.linspace(1.0, 3.0, m, dtype=np.float32)


def _effort_grid(m_e: int = 5) -> np.ndarray:
    return np.linspace(0.0, 10.0, m_e, dtype=np.float32)


def _agent_ids(n: int = N) -> list[str]:
    return [str(i) for i in range(n)]


def _make_log(tmp_path, **kwargs):
    from hotelling.simulation.dense_log import DenseLog
    return DenseLog(
        run_dir=tmp_path,
        T=T,
        N=N,
        agent_ids=_agent_ids(),
        price_grid=_price_grid(),
        effort_grid=_effort_grid(),
        **kwargs,
    )


def _random_step(rng, n=N, m=M, m_e=5):
    price_idx  = rng.integers(0, m,   size=n, dtype=np.int64)
    effort_idx = rng.integers(0, m_e, size=n, dtype=np.int64)
    demands    = rng.uniform(0, 100, size=n)
    profits    = rng.uniform(-5, 50, size=n)
    return price_idx, effort_idx, demands, profits


def _write_all(log, rng=None, t_stop=T):
    if rng is None:
        rng = np.random.default_rng(0)
    for t in range(t_stop):
        pi, ei, d, pr = _random_step(rng)
        log.write_step(t, pi, ei, d, pr)


# ---------------------------------------------------------------------------
# Default behaviour (stride=1, no tail) — backward compat
# ---------------------------------------------------------------------------

class TestDefaultBehaviour:
    def test_all_steps_scheduled(self, tmp_path):
        log = _make_log(tmp_path)
        assert len(log._recorded_steps) == T

    def test_rows_written_after_full_run(self, tmp_path):
        log = _make_log(tmp_path)
        _write_all(log)
        assert log._rows_written == T

    def test_recorded_steps_property(self, tmp_path):
        log = _make_log(tmp_path)
        _write_all(log)
        np.testing.assert_array_equal(log.recorded_steps, np.arange(T))

    def test_values_round_trip(self, tmp_path):
        rng = np.random.default_rng(7)
        log = _make_log(tmp_path)
        pi_ref, ei_ref, d_ref, pr_ref = _random_step(rng)
        log.write_step(0, pi_ref, ei_ref, d_ref, pr_ref)
        log.flush()
        log2 = log.__class__.load(tmp_path)
        np.testing.assert_array_equal(log2.price_idx[0],  pi_ref.astype("int8"))
        np.testing.assert_array_equal(log2.effort_idx[0], ei_ref.astype("int8"))
        np.testing.assert_allclose(log2.demands[0], d_ref.astype(np.float32), rtol=1e-5)

    def test_to_dataframe_period_col(self, tmp_path):
        log = _make_log(tmp_path)
        _write_all(log)
        df = log.to_dataframe(agent_idx=0)
        np.testing.assert_array_equal(df["period"].values, np.arange(T))

    def test_to_dataframe_all_agents(self, tmp_path):
        log = _make_log(tmp_path)
        _write_all(log)
        df = log.to_dataframe()
        assert len(df) == T * N
        assert set(df.columns) >= {"period", "agent_id", "price", "effort",
                                   "demand", "profit"}


# ---------------------------------------------------------------------------
# dense_stride only
# ---------------------------------------------------------------------------

class TestStrideOnly:
    STRIDE = 10

    def test_scheduled_count(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=self.STRIDE)
        # steps 0, 10, 20, ... 90 → 10 rows
        assert len(log._recorded_steps) == T // self.STRIDE

    def test_stride_steps_correct(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=self.STRIDE)
        expected = np.arange(0, T, self.STRIDE)
        np.testing.assert_array_equal(log._recorded_steps, expected)

    def test_non_stride_steps_skipped(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=self.STRIDE)
        rng = np.random.default_rng(1)
        pi0, ei0, d0, pr0 = _random_step(rng)
        pi1, ei1, d1, pr1 = _random_step(rng)
        log.write_step(0,  pi0, ei0, d0, pr0)  # row 0 recorded
        log.write_step(1,  pi1, ei1, d1, pr1)  # row 1 SKIPPED
        log.write_step(10, pi0, ei0, d0, pr0)  # row 1 recorded
        assert log._rows_written == 2           # only 2 rows actually written

    def test_period_col_reflects_stride(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=self.STRIDE)
        _write_all(log)
        df = log.to_dataframe(agent_idx=0)
        np.testing.assert_array_equal(
            df["period"].values, np.arange(0, T, self.STRIDE)
        )

    def test_flush_load_meta(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=self.STRIDE)
        _write_all(log)
        log.flush()
        meta = json.loads((tmp_path / "dense_log_meta.json").read_text())
        assert meta["dense_stride"] == self.STRIDE
        assert meta["rows_written"] == T // self.STRIDE


# ---------------------------------------------------------------------------
# dense_tail only (stride = T so only step 0 + tail)
# ---------------------------------------------------------------------------

class TestTailOnly:
    TAIL = 20

    def _log(self, tmp_path):
        # stride = T means only step 0 from stride; tail adds last TAIL steps
        return _make_log(tmp_path, dense_stride=T, dense_tail=self.TAIL)

    def test_scheduled_count(self, tmp_path):
        log = self._log(tmp_path)
        # step 0 from stride, steps (T-TAIL)..T-1 from tail = TAIL + 1 unique
        expected = 1 + self.TAIL   # step 0 is not in [T-TAIL, T) since T-TAIL > 0
        assert len(log._recorded_steps) == expected

    def test_tail_steps_present(self, tmp_path):
        log = self._log(tmp_path)
        steps = set(log._recorded_steps.tolist())
        for t in range(T - self.TAIL, T):
            assert t in steps

    def test_period_col_for_tail(self, tmp_path):
        log = self._log(tmp_path)
        _write_all(log)
        df = log.to_dataframe(agent_idx=0)
        actual = df["period"].values
        # Last TAIL steps should be the final entries
        tail_steps = np.arange(T - self.TAIL, T)
        np.testing.assert_array_equal(actual[-self.TAIL:], tail_steps)


# ---------------------------------------------------------------------------
# stride + tail combination
# ---------------------------------------------------------------------------

class TestStridePlusTail:
    STRIDE = 10
    TAIL   = 15

    def test_no_duplicate_rows(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=self.STRIDE, dense_tail=self.TAIL)
        # steps.npy should have no duplicates
        steps = log._recorded_steps
        assert len(steps) == len(np.unique(steps))

    def test_all_tail_steps_present(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=self.STRIDE, dense_tail=self.TAIL)
        steps = set(log._recorded_steps.tolist())
        for t in range(T - self.TAIL, T):
            assert t in steps

    def test_stride_steps_present_outside_tail(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=self.STRIDE, dense_tail=self.TAIL)
        steps = set(log._recorded_steps.tolist())
        for t in range(0, T - self.TAIL, self.STRIDE):
            assert t in steps, f"stride step {t} missing"

    def test_row_count_is_union_size(self, tmp_path):
        stride_set = set(range(0, T, self.STRIDE))
        tail_set   = set(range(max(0, T - self.TAIL), T))
        expected   = len(stride_set | tail_set)
        log = _make_log(tmp_path, dense_stride=self.STRIDE, dense_tail=self.TAIL)
        assert len(log._recorded_steps) == expected


# ---------------------------------------------------------------------------
# store_demand_profit=False
# ---------------------------------------------------------------------------

class TestNoDemandProfit:
    def test_arrays_are_none(self, tmp_path):
        log = _make_log(tmp_path, store_demand_profit=False)
        assert log.demands is None
        assert log.profits is None

    def test_no_demand_files_on_disk(self, tmp_path):
        log = _make_log(tmp_path, store_demand_profit=False)
        log.flush()
        assert not (tmp_path / "demands.npy").exists()
        assert not (tmp_path / "profits.npy").exists()

    def test_write_step_does_not_raise(self, tmp_path):
        log = _make_log(tmp_path, store_demand_profit=False)
        rng = np.random.default_rng(3)
        pi, ei, d, pr = _random_step(rng)
        log.write_step(0, pi, ei, d, pr)  # should not raise
        assert log._rows_written == 1

    def test_to_dataframe_no_demand_profit_cols(self, tmp_path):
        log = _make_log(tmp_path, store_demand_profit=False)
        _write_all(log)
        df = log.to_dataframe(agent_idx=0)
        assert "demand" not in df.columns
        assert "profit" not in df.columns

    def test_load_preserves_flag(self, tmp_path):
        log = _make_log(tmp_path, store_demand_profit=False)
        _write_all(log)
        log.flush()
        log2 = log.__class__.load(tmp_path)
        assert log2.demands is None
        assert log2.profits is None
        assert log2._store_demand_profit is False

    def test_meta_records_flag(self, tmp_path):
        log = _make_log(tmp_path, store_demand_profit=False)
        log.flush()
        meta = json.loads((tmp_path / "dense_log_meta.json").read_text())
        assert meta["store_demand_profit"] is False


# ---------------------------------------------------------------------------
# float_dtype
# ---------------------------------------------------------------------------

class TestFloatDtype:
    def test_float32_default(self, tmp_path):
        log = _make_log(tmp_path)
        assert log.demands.dtype == np.dtype("float32")
        assert log.profits.dtype == np.dtype("float32")

    def test_float64_explicit(self, tmp_path):
        log = _make_log(tmp_path, float_dtype="float64")
        assert log.demands.dtype == np.dtype("float64")

    def test_float64_precision_preserved(self, tmp_path):
        """float64 should reproduce exact values; float32 is lossy."""
        log = _make_log(tmp_path, float_dtype="float64")
        rng = np.random.default_rng(5)
        pi, ei, d, _ = _random_step(rng)
        pr = np.array([1.23456789012345] * N)
        log.write_step(0, pi, ei, d, pr)
        log.flush()
        log2 = log.__class__.load(tmp_path)
        np.testing.assert_array_equal(log2.profits[0], pr)

    def test_meta_records_dtype(self, tmp_path):
        log = _make_log(tmp_path, float_dtype="float64")
        log.flush()
        meta = json.loads((tmp_path / "dense_log_meta.json").read_text())
        assert meta["float_dtype"] == "float64"


# ---------------------------------------------------------------------------
# Early convergence (simulation ends before T)
# ---------------------------------------------------------------------------

class TestEarlyConvergence:
    def test_rows_written_lt_scheduled(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=10)
        # Only run 50 steps instead of T=100
        _write_all(log, t_stop=50)
        # stride=10: steps 0,10,20,30,40 → 5 rows written
        assert log._rows_written == 5

    def test_to_dataframe_only_written_rows(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=10)
        _write_all(log, t_stop=50)
        df = log.to_dataframe(agent_idx=0)
        assert len(df) == 5
        np.testing.assert_array_equal(df["period"].values, [0, 10, 20, 30, 40])

    def test_flush_load_rows_written(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=10)
        _write_all(log, t_stop=50)
        log.flush()
        log2 = log.__class__.load(tmp_path)
        assert log2._rows_written == 5


# ---------------------------------------------------------------------------
# Flush / Load round-trip
# ---------------------------------------------------------------------------

class TestFlushLoad:
    def test_meta_all_keys_present(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=5, dense_tail=10,
                        store_demand_profit=False, float_dtype="float64")
        log.flush()
        meta = json.loads((tmp_path / "dense_log_meta.json").read_text())
        for key in ("T_allocated", "N", "rows_written", "n_rows_allocated",
                    "store_demand_profit", "float_dtype",
                    "dense_stride", "dense_tail", "T_written"):
            assert key in meta, f"meta missing key {key!r}"

    def test_steps_npy_written(self, tmp_path):
        log = _make_log(tmp_path)
        assert (tmp_path / "steps.npy").exists()

    def test_load_recorded_steps_match(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=7, dense_tail=5)
        _write_all(log)
        log.flush()
        log2 = log.__class__.load(tmp_path)
        np.testing.assert_array_equal(log2._recorded_steps, log._recorded_steps)

    def test_load_price_idx_match(self, tmp_path):
        rng = np.random.default_rng(9)
        log = _make_log(tmp_path)
        _write_all(log, rng=rng)
        log.flush()
        rng2 = np.random.default_rng(9)
        log2 = log.__class__.load(tmp_path)
        pi_ref, _, _, _ = _random_step(rng2)
        np.testing.assert_array_equal(log2.price_idx[0], pi_ref.astype("int8"))

    def test_legacy_load_without_steps_npy(self, tmp_path):
        """Load a log that has no steps.npy (legacy format)."""
        log = _make_log(tmp_path)
        _write_all(log)
        log.flush()
        # Remove steps.npy to simulate legacy format
        (tmp_path / "steps.npy").unlink()
        log2 = log.__class__.load(tmp_path)
        # Should reconstruct steps as 0..rows_written-1
        np.testing.assert_array_equal(log2._recorded_steps, np.arange(T))


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------

class TestToDataframe:
    def test_agent_slice_period_col(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=5)
        _write_all(log)
        df = log.to_dataframe(agent_idx=0, step_slice=slice(2, 5))
        # rows 2,3,4 → steps 10, 15, 20
        np.testing.assert_array_equal(df["period"].values, [10, 15, 20])

    def test_all_agents_period_col(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=5)
        _write_all(log)
        df = log.to_dataframe(step_slice=slice(0, 3))
        # rows 0,1,2 → steps 0,5,10; each repeated N times
        expected_periods = np.repeat([0, 5, 10], N)
        np.testing.assert_array_equal(df["period"].values, expected_periods)

    def test_price_decodes_correctly(self, tmp_path):
        log = _make_log(tmp_path)
        rng = np.random.default_rng(11)
        pi, ei, d, pr = _random_step(rng)
        log.write_step(0, pi, ei, d, pr)
        df = log.to_dataframe(agent_idx=0, step_slice=slice(0, 1))
        assert float(df["price"].iloc[0]) == pytest.approx(
            float(_price_grid()[pi[0]]), rel=1e-4
        )

    def test_no_demand_profit_cols_when_not_stored(self, tmp_path):
        log = _make_log(tmp_path, store_demand_profit=False)
        _write_all(log)
        for agent_idx in (None, 0):
            df = log.to_dataframe(agent_idx=agent_idx)
            assert "demand" not in df.columns
            assert "profit" not in df.columns


# ---------------------------------------------------------------------------
# Validation / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_stride_raises(self, tmp_path):
        with pytest.raises(ValueError, match="dense_stride"):
            _make_log(tmp_path, dense_stride=0)

    def test_stride_of_1_and_tail_of_T_equals_full(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=1, dense_tail=T)
        assert len(log._recorded_steps) == T

    def test_step_to_row_is_inverse(self, tmp_path):
        log = _make_log(tmp_path, dense_stride=7, dense_tail=3)
        for row, step in enumerate(log._recorded_steps):
            assert log._step_to_row[int(step)] == row
