"""
Static matplotlib visualisations for the Hotelling simulation.

All functions accept keyword arguments that are forwarded to the underlying
matplotlib calls so callers can fine-tune figure aesthetics without
modifying this module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Matplotlib is an optional dependency; functions raise ImportError at
# call-time if it is absent.


def _require_matplotlib():  # noqa: ANN201
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it with: pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# Snapshot – single-step spatial view
# ---------------------------------------------------------------------------


def plot_snapshot(
    market,  # noqa: ANN001
    title: str = "Market Snapshot",
    figsize: tuple = (7, 7),
    consumer_alpha: float = 0.15,
    show_consumers: bool = True,
    ax=None,  # noqa: ANN001
    **kwargs: Any,
):  # noqa: ANN201
    """Plot firm positions and optionally consumer locations.

    Parameters
    ----------
    market:
        A :class:`~hotelling.core.market.Market` instance.
    title:
        Figure title.
    figsize:
        Figure size in inches.
    consumer_alpha:
        Alpha for consumer scatter points.
    show_consumers:
        Whether to plot consumer positions.
    ax:
        Optional existing ``matplotlib.axes.Axes``.
    """
    plt = _require_matplotlib()
    import numpy as np  # noqa: PLC0415

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    city = market.city

    if show_consumers:
        locs = city.consumer_locations
        ax.scatter(locs[:, 0], locs[:, 1], s=5, alpha=consumer_alpha, color="grey", label="Consumers")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, firm in enumerate(market.firms):
        color = colors[i % len(colors)]
        ax.scatter(*firm.location, s=200, zorder=5, color=color, label=firm.name, **kwargs)
        ax.annotate(
            f"{firm.name}\np={firm.price:.2f}\n{firm.market_share:.1%}",
            xy=firm.location,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    ax.set_xlim(0, city.width)
    ax.set_ylim(0, city.height)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    if fig is not None:
        plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Time-series plots
# ---------------------------------------------------------------------------


def plot_market_shares(
    agg: Dict[str, Any],
    figsize: tuple = (9, 4),
    ax=None,  # noqa: ANN001
    **kwargs: Any,
):  # noqa: ANN201
    """Plot market-share time series for all firms.

    Parameters
    ----------
    agg:
        Aggregated results dict as returned by
        :func:`~hotelling.simulation.metrics.aggregate_results`.
    """
    plt = _require_matplotlib()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    steps = agg["steps"]
    for firm_id, series in agg["firms"].items():
        ax.plot(steps, series["market_share"], label=firm_id, **kwargs)

    ax.set_xlabel("Step")
    ax.set_ylabel("Market share")
    ax.set_title("Market Share Evolution")
    ax.legend()
    if fig is not None:
        plt.tight_layout()
    return ax


def plot_price_evolution(
    agg: Dict[str, Any],
    figsize: tuple = (9, 4),
    ax=None,  # noqa: ANN001
    **kwargs: Any,
):  # noqa: ANN201
    """Plot price time series for all firms."""
    plt = _require_matplotlib()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    steps = agg["steps"]
    for firm_id, series in agg["firms"].items():
        ax.plot(steps, series["price"], label=firm_id, **kwargs)

    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.set_title("Price Evolution")
    ax.legend()
    if fig is not None:
        plt.tight_layout()
    return ax


def plot_firm_trajectories(
    agg: Dict[str, Any],
    city_width: float = 1.0,
    city_height: float = 1.0,
    figsize: tuple = (7, 7),
    ax=None,  # noqa: ANN001
    **kwargs: Any,
):  # noqa: ANN201
    """Plot spatial trajectories of all firms over the simulation."""
    plt = _require_matplotlib()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (firm_id, series) in enumerate(agg["firms"].items()):
        color = colors[i % len(colors)]
        xs, ys = series["x"], series["y"]
        ax.plot(xs, ys, "-o", color=color, markersize=3, label=firm_id, **kwargs)
        # Mark start and end
        ax.scatter(xs[0], ys[0], color=color, s=80, marker="^", zorder=5)
        ax.scatter(xs[-1], ys[-1], color=color, s=80, marker="s", zorder=5)

    ax.set_xlim(0, city_width)
    ax.set_ylim(0, city_height)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Firm Trajectories (▲ start, ■ end)")
    ax.legend()
    ax.set_aspect("equal")
    if fig is not None:
        plt.tight_layout()
    return ax
