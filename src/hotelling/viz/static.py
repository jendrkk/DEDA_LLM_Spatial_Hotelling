"""Static matplotlib visualizations.

Responsibility: generate publication-quality static figures for the seminar
report and notebooks.  All functions return matplotlib Figure objects and
optionally save to a file.

Public API: plot_price_timeseries, plot_price_trajectories_by_chain,
    plot_irf, plot_profit_heatmap, plot_dose_response, plot_spatial_voronoi

Key dependencies: matplotlib, numpy, pandas, scipy (for Voronoi)

References:
    Calvano et al. (2020 AER) Fig. 1-4 - style reference;
    Matplotlib documentation https://matplotlib.org/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd


def plot_price_timeseries(
    results_df: pd.DataFrame,
    agent_ids: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[Path] = None,
) -> Any:
    """Plot price trajectories over training steps.

    Parameters
    ----------
    results_df : DataFrame with columns period, agent_id, price
    agent_ids : subset of agents to plot; None = all
    figsize : matplotlib figure size (width, height) in inches
    save_path : if provided, save figure to this path (PNG/PDF)

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def plot_price_trajectories_by_chain(
    run_dir: str | Path,
    *,
    ax=None,
    show_benchmarks: bool = True,
):
    """Plot mean price over simulation steps: one line for the total market
    mean and one per chain type (discount/standard/bio), read from
    <run_dir>/aggregate.parquet (columns: step, mean_price,
    mean_price_discount, mean_price_standard, mean_price_bio).

    If show_benchmarks and <run_dir>/metadata.json contains a
    chain_price_table, draw dashed horizontal Nash and monopoly reference
    lines for the GLOBAL benchmark (and, if present, faint per-chain Nash
    lines). Returns the matplotlib Axes.
    """
    import json
    import logging

    import matplotlib.pyplot as plt

    logger = logging.getLogger(__name__)
    run_path = Path(run_dir)
    agg_path = run_path / "aggregate.parquet"
    agg = pd.read_parquet(agg_path)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    ax.plot(
        agg["step"],
        agg["mean_price"],
        color="black",
        lw=2,
        label="Total mean",
    )

    chain_cols = {
        "discount": ("mean_price_discount", "tab:green", "Discount"),
        "standard": ("mean_price_standard", "tab:blue", "Standard"),
        "bio": ("mean_price_bio", "tab:red", "Bio"),
    }
    has_chain_cols = all(col in agg.columns for _, col, _ in chain_cols.values())
    if has_chain_cols:
        for _ct, (col, color, label) in chain_cols.items():
            ax.plot(agg["step"], agg[col], color=color, lw=1.3, label=label)
    else:
        msg = (
            f"Run {run_path.name} predates per-chain price logging; "
            "plotting total mean only (missing mean_price_discount/standard/bio)."
        )
        logger.info(msg)
        print(msg)

    if show_benchmarks:
        meta_path = run_path / "metadata.json"
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            cpt = meta.get("chain_price_table") or {}
            global_row = cpt.get("global")
            if global_row:
                ax.axhline(
                    global_row["nash"],
                    color="grey",
                    ls="--",
                    lw=1.2,
                    label="Nash (global)",
                )
                ax.axhline(
                    global_row["mono"],
                    color="lightcoral",
                    ls="--",
                    lw=1.2,
                    label="Mono (global)",
                )
            for ct, color in (
                ("discount", "tab:green"),
                ("standard", "tab:blue"),
                ("bio", "tab:red"),
            ):
                row = cpt.get(ct)
                if row and "nash" in row:
                    ax.axhline(
                        row["nash"],
                        color=color,
                        ls=":",
                        lw=0.8,
                        alpha=0.45,
                        label=f"Nash ({ct})",
                    )

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Mean price (EUR)")
    ax.set_title(f"Price trajectories by chain — {run_path.name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    if created_fig:
        fig.tight_layout()

    return ax


def plot_irf(
    irf_df: pd.DataFrame,
    deviator: str,
    figsize: Tuple[int, int] = (8, 4),
    save_path: Optional[Path] = None,
) -> Any:
    """Plot impulse-response function in Calvano (2020) Fig. 4 style.

    Parameters
    ----------
    irf_df : DataFrame returned by hotelling.analysis.irf.impulse_response
    deviator : agent_id of the deviating firm (highlighted in red)
    figsize : figure size
    save_path : optional output path

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def plot_profit_heatmap(
    alpha_values: np.ndarray,
    beta_values: np.ndarray,
    profit_gains: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
) -> Any:
    """Plot Delta(alpha, beta) heatmap for sweep results.

    Parameters
    ----------
    alpha_values : 1-D array of alpha (learning rate) values on x-axis
    beta_values : 1-D array of beta (exploration decay) values on y-axis
    profit_gains : 2-D array of shape (len(beta), len(alpha)) with Delta values
    figsize : figure size
    save_path : optional output path

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def plot_dose_response(
    transport_costs: np.ndarray,
    profit_gains: np.ndarray,
    figsize: Tuple[int, int] = (8, 4),
    save_path: Optional[Path] = None,
) -> Any:
    """Plot profit gain Delta as a function of transport cost t.

    Parameters
    ----------
    transport_costs : 1-D array of transport cost values
    profit_gains : 1-D array of mean Delta values (same length)
    figsize : figure size
    save_path : optional output path

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError


def plot_spatial_voronoi(
    city: Any,
    firms: Any,
    prices: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[Path] = None,
) -> Any:
    """Plot Voronoi market-area map with firm locations and optional price coloring.

    Parameters
    ----------
    city : City instance (for boundary and optional population_grid)
    firms : list of Firm instances
    prices : optional shape (N,) array of prices for color mapping
    figsize : figure size
    save_path : optional output path

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError
