"""Per-step Parquet + MLflow simulation recorder.

Responsibility: buffer per-step observations in memory and flush to long-format
Parquet files.  Optionally logs hyperparameters and scalar metrics to MLflow.
Uses the DataCollector pattern: one row per (run_id, period, agent_id).

Public API: SimulationRecorder

Key dependencies: pathlib, uuid, pyarrow, mlflow (optional)

References:
    DataCollector pattern (Mesa framework);
    MLflow (Zaharia et al. 2018) https://mlflow.org/.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

_AGENT_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("period", pa.int32()),
    ("agent_id", pa.string()),
    ("price", pa.float32()),
    ("effort", pa.float32()),
    ("demand", pa.float32()),
    ("profit", pa.float32()),
    ("price_idx", pa.int8()),
    ("effort_idx", pa.int8()),
])


class SimulationRecorder:
    """Records simulation data to Parquet files and optionally to MLflow.

    Parameters
    ----------
    run_dir : directory for this run's output files (created if absent)
    run_id : unique run identifier; auto-generated UUID if not provided
    mlflow_tracking_uri : MLflow tracking server URI; disabled if None
    flush_every : buffer size threshold before streaming a batch to disk
    """

    def __init__(
        self,
        run_dir: Path,
        run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        flush_every: int = 50_000,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id or str(uuid.uuid4())
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.flush_every = flush_every
        self._buffer: List[Dict[str, Any]] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self._mlflow_run: Optional[Any] = None

        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def record_step(
        self,
        period: int,
        agent_id: str,
        price: float,
        demand: float,
        profit: float,
        effort: float = float("nan"),
        price_idx: int = 0,
        effort_idx: int = 0,
        **kwargs: Any,
    ) -> None:
        """Append one row to the in-memory buffer.

        Parameters
        ----------
        period : simulation step index
        agent_id : identifier of the acting agent/firm
        price : price charged this period
        demand : market share / demand received
        profit : profit earned this period
        effort : effort level this period
        price_idx : discrete price grid index
        effort_idx : discrete effort grid index
        **kwargs : ignored (backward compatibility for extra column names)
        """
        row: Dict[str, Any] = {
            "run_id": self.run_id,
            "period": int(period),
            "agent_id": agent_id,
            "price": float(price),
            "effort": float(kwargs.get("effort", effort)),
            "demand": float(demand),
            "profit": float(profit),
            "price_idx": int(kwargs.get("price_idx", kwargs.get("price_index", price_idx))),
            "effort_idx": int(kwargs.get("effort_idx", effort_idx)),
        }
        self._buffer.append(row)
        self._auto_flush()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _auto_flush(self) -> None:
        if len(self._buffer) >= self.flush_every:
            self._write_buffer()

    def _write_buffer(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=_AGENT_SCHEMA)
        if self._writer is None:
            out_path = self.run_dir / "agents.parquet"
            self._writer = pq.ParquetWriter(out_path, _AGENT_SCHEMA)
        self._writer.write_table(table)
        self._buffer.clear()

    def flush(self) -> Path:
        """Write remaining buffered rows and close the Parquet writer.

        The output file is named ``agents.parquet`` inside run_dir.
        The writer is opened lazily on the first write so empty runs
        do not leave a zero-byte file.

        Returns
        -------
        Path to the Parquet file (may not exist if no rows were recorded)
        """
        self._write_buffer()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self.run_dir / "agents.parquet"

    # ------------------------------------------------------------------
    # MLflow integration
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to MLflow (no-op if MLflow not configured).

        Parameters
        ----------
        params : dict of parameter name -> value
        """
        if self.mlflow_tracking_uri is None:
            return
        raise NotImplementedError

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log scalar metrics to MLflow (no-op if MLflow not configured).

        Parameters
        ----------
        metrics : dict of metric name -> float value
        step : optional step index for time-series metrics
        """
        if self.mlflow_tracking_uri is None:
            return
        raise NotImplementedError

    def close(self) -> None:
        """Flush remaining buffer and close any open MLflow run."""
        self.flush()
        if self._mlflow_run is not None:
            raise NotImplementedError
