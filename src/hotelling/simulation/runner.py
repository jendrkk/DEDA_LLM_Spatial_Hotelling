"""Hydra-driven batch and sweep runner.

Responsibility: provide functions to run a single simulation session from a
Hydra config dict, and to run a full parameter sweep in parallel using
multiprocessing.Pool.

Public API: run_single_session, run_sweep

Key dependencies: multiprocessing, pathlib, hotelling.simulation.engine,
    hydra-core (optional)

References:
    Hydra (Yadan 2019) https://hydra.cc/;
    Calvano et al. (2020 AER) §III - batch training protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def run_single_session(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single simulation session from a Hydra config dict.

    Builds the City, Firm list, agents, and SimulationEngine from the config,
    runs the engine, and returns session metrics.

    Parameters
    ----------
    config : Hydra config dict (usually OmegaConf DictConfig converted to dict)

    Returns
    -------
    dict with keys: run_id, seed, n_steps, profit_gain, final_prices, elapsed_s
    """
    raise NotImplementedError


def run_sweep(
    config_dir: Path,
    sweep_config_name: str,
    n_jobs: int = -1,
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run a parameter sweep defined in configs/sweep/.

    Loads the sweep config, expands the grid, and runs each configuration
    in parallel using multiprocessing.Pool (n_jobs=-1 = all CPUs).

    Parameters
    ----------
    config_dir : directory containing Hydra config files
    sweep_config_name : name of the sweep YAML (without .yaml extension)
    n_jobs : number of parallel workers; -1 uses os.cpu_count()
    output_dir : directory to write per-run Parquet files

    Returns
    -------
    list of result dicts from all sessions, one per parameter combination
    """
    raise NotImplementedError
