"""Memory-mapped dense per-step simulation log (T × N)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class DenseLog:
    """Memory-mapped per-step simulation log: (T, N) arrays for all quantities.

    Disk layout (run_dir/):
        price_idx.npy    — (T, N) int8
        effort_idx.npy   — (T, N) int8
        demands.npy      — (T, N) float32
        profits.npy      — (T, N) float32
        agent_ids.npy    — (N,) str array
        price_grid.npy   — (m,) float32
        effort_grid.npy  — (m_effort,) float32
        dense_log_meta.json — T_allocated, T_written, N
    """

    def __init__(
        self,
        run_dir: Path,
        T: int,
        N: int,
        agent_ids: list[str],
        price_grid: np.ndarray,
        effort_grid: np.ndarray,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.T = T
        self.N = N
        self.agent_ids = agent_ids
        self.price_grid = np.asarray(price_grid, dtype=np.float32)
        self.effort_grid = np.asarray(effort_grid, dtype=np.float32)
        self._t_written = 0
        self._flush_every = 10_000

        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.price_idx = np.memmap(
            self.run_dir / "price_idx.npy", dtype="int8", mode="w+", shape=(T, N)
        )
        self.effort_idx = np.memmap(
            self.run_dir / "effort_idx.npy", dtype="int8", mode="w+", shape=(T, N)
        )
        self.demands = np.memmap(
            self.run_dir / "demands.npy", dtype="float32", mode="w+", shape=(T, N)
        )
        self.profits = np.memmap(
            self.run_dir / "profits.npy", dtype="float32", mode="w+", shape=(T, N)
        )

        np.save(self.run_dir / "agent_ids.npy", np.array(agent_ids, dtype=str))
        np.save(self.run_dir / "price_grid.npy", self.price_grid)
        np.save(self.run_dir / "effort_grid.npy", self.effort_grid)

    def write_step(
        self,
        t: int,
        price_idxs: np.ndarray,
        effort_idxs: np.ndarray,
        demands: np.ndarray,
        profits: np.ndarray,
    ) -> None:
        """Write one simulation step to row t."""
        self.price_idx[t] = price_idxs.astype("int8")
        self.effort_idx[t] = effort_idxs.astype("int8")
        self.demands[t] = demands.astype("float32")
        self.profits[t] = profits.astype("float32")
        self._t_written = t + 1

        if t > 0 and t % self._flush_every == 0:
            self.flush()

    def flush(self) -> None:
        """Flush all memmap arrays and write updated metadata."""
        self.price_idx.flush()
        self.effort_idx.flush()
        self.demands.flush()
        self.profits.flush()
        meta = {
            "T_allocated": self.T,
            "T_written": self._t_written,
            "N": self.N,
        }
        with (self.run_dir / "dense_log_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, run_dir: Path) -> DenseLog:
        """Load an existing DenseLog from a run directory (read-only)."""
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "dense_log_meta.json").read_text())
        T = meta["T_allocated"]
        N = meta["N"]
        agent_ids = list(np.load(run_dir / "agent_ids.npy"))
        price_grid = np.load(run_dir / "price_grid.npy")
        effort_grid = np.load(run_dir / "effort_grid.npy")

        obj = object.__new__(cls)
        obj.run_dir = run_dir
        obj.T = T
        obj.N = N
        obj.agent_ids = agent_ids
        obj.price_grid = price_grid
        obj.effort_grid = effort_grid
        obj._t_written = meta["T_written"]
        obj._flush_every = 10_000
        obj.price_idx = np.memmap(
            run_dir / "price_idx.npy", dtype="int8", mode="r", shape=(T, N)
        )
        obj.effort_idx = np.memmap(
            run_dir / "effort_idx.npy", dtype="int8", mode="r", shape=(T, N)
        )
        obj.demands = np.memmap(
            run_dir / "demands.npy", dtype="float32", mode="r", shape=(T, N)
        )
        obj.profits = np.memmap(
            run_dir / "profits.npy", dtype="float32", mode="r", shape=(T, N)
        )
        return obj

    def to_dataframe(
        self,
        agent_idx: int | None = None,
        step_slice: slice | None = None,
    ) -> pd.DataFrame:
        """Load a slice of the log as a pandas DataFrame."""
        import pandas as pd

        sl = step_slice or slice(None, self._t_written)
        if agent_idx is not None:
            pidx = self.price_idx[sl, agent_idx]
            eidx = self.effort_idx[sl, agent_idx]
            periods = np.arange(self._t_written)[sl]
            return pd.DataFrame(
                {
                    "period": periods,
                    "agent_id": self.agent_ids[agent_idx],
                    "price_idx": pidx,
                    "effort_idx": eidx,
                    "price": self.price_grid[pidx],
                    "effort": self.effort_grid[eidx],
                    "demand": self.demands[sl, agent_idx],
                    "profit": self.profits[sl, agent_idx],
                }
            )

        T_sl = self.price_idx[sl].shape[0]
        periods = np.repeat(np.arange(T_sl), self.N)
        agent_col = np.tile(self.agent_ids, T_sl)
        pidx_flat = self.price_idx[sl].ravel()
        eidx_flat = self.effort_idx[sl].ravel()
        return pd.DataFrame(
            {
                "period": periods,
                "agent_id": agent_col,
                "price_idx": pidx_flat,
                "effort_idx": eidx_flat,
                "price": self.price_grid[pidx_flat],
                "effort": self.effort_grid[eidx_flat],
                "demand": self.demands[sl].ravel(),
                "profit": self.profits[sl].ravel(),
            }
        )
