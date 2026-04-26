"""DuckDB query layer over simulation results Parquet files.

Responsibility: provide a convenient pandas-friendly query interface over the
long-format Parquet files produced by SimulationRecorder.  Supports both
on-disk (glob over a directory) and in-memory DuckDB modes.

Public API: ResultsDB

Key dependencies: duckdb, pandas, pathlib

References:
    DuckDB (Raasveldt & Mühleisen 2019) https://duckdb.org/.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


class ResultsDB:
    """DuckDB-backed query interface for simulation results.

    Parameters
    ----------
    parquet_dir : directory containing ``*.parquet`` run files
    in_memory : if True, load all parquet into an in-memory DuckDB instance
        instead of querying files directly (faster for small datasets)
    """

    def __init__(self, parquet_dir: Path, in_memory: bool = False) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.in_memory = in_memory
        self._conn: Optional[Any] = None  # type: ignore[name-defined]

    def connect(self) -> None:
        """Open DuckDB connection and (optionally) load all parquet files.

        Raises ImportError if duckdb is not installed.
        """
        raise NotImplementedError

    def query(self, sql: str) -> pd.DataFrame:
        """Execute an arbitrary SQL query and return the result as a DataFrame.

        Parameters
        ----------
        sql : SQL string; reference the results table as ``runs``

        Returns
        -------
        pd.DataFrame
        """
        raise NotImplementedError

    def list_runs(self) -> List[str]:
        """Return list of unique run_ids present in the database.

        Returns
        -------
        list of run_id strings
        """
        raise NotImplementedError

    def get_run(self, run_id: str) -> pd.DataFrame:
        """Return all rows for a given run_id.

        Parameters
        ----------
        run_id : UUID string

        Returns
        -------
        pd.DataFrame with columns: run_id, period, agent_id, price, demand, profit, ...
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            raise NotImplementedError
