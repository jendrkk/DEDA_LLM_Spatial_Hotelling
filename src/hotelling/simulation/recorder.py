"""Per-step Parquet + MLflow simulation recorder.

Responsibility: buffer per-step observations in memory and flush to long-format
Parquet files.  Optionally logs hyperparameters and scalar metrics to MLflow.
Uses the DataCollector pattern: one row per (run_id, period, agent_id).

Public API: SimulationRecorder

Key dependencies: pathlib, uuid, pandas, pyarrow, mlflow (optional)

References:
    DataCollector pattern (Mesa framework);
    MLflow (Zaharia et al. 2018) https://mlflow.org/.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class SimulationRecorder:
    """Records simulation data to Parquet files and optionally to MLflow.

    Parameters
    ----------
    run_dir : directory for this run's output files (created if absent)
    run_id : unique run identifier; auto-generated UUID if not provided
    mlflow_tracking_uri : MLflow tracking server URI; disabled if None
    """

    def __init__(
        self,
        run_dir: Path,
        run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id or str(uuid.uuid4())
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self._buffer: List[Dict[str, Any]] = []
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
        **kwargs : additional columns (e.g. location_x, location_y, price_index)
        """
        row: Dict[str, Any] = {
            "run_id": self.run_id,
            "period": period,
            "agent_id": agent_id,
            "price": price,
            "demand": demand,
            "profit": profit,
            **kwargs,
        }
        self._buffer.append(row)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> Path:
        """Write buffered rows to a Parquet file and clear the buffer.

        The output file is named ``<run_id>.parquet`` inside run_dir.

        Returns
        -------
        Path to the written Parquet file
        """
        if not self._buffer:
            return self.run_dir / f"{self.run_id}.parquet"

        df = pd.DataFrame(self._buffer)
        out_path = self.run_dir / f"{self.run_id}.parquet"
        df.to_parquet(out_path, index=False)
        self._buffer = []
        return out_path

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
