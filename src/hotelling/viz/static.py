"""Static matplotlib visualizations.

Responsibility: generate publication-quality static figures for the seminar
report and notebooks.  All functions return matplotlib Figure objects and
optionally save to a file.

Public API: plot_price_timeseries, plot_irf, plot_profit_heatmap,
    plot_dose_response, plot_spatial_voronoi

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
