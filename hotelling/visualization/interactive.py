"""
Interactive visualisations using Plotly.

Functions here mirror those in :mod:`hotelling.visualization.plots` but
produce interactive HTML/widget figures suitable for use inside Jupyter
notebooks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _require_plotly():  # noqa: ANN201
    try:
        import plotly.graph_objects as go  # noqa: PLC0415

        return go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive visualisations. "
            "Install it with: pip install plotly"
        ) from exc


def interactive_snapshot(market, title: str = "Market Snapshot", **kwargs: Any):  # noqa: ANN001, ANN201
    """Return a Plotly figure with an interactive market snapshot.

    Parameters
    ----------
    market:
        A :class:`~hotelling.core.market.Market` instance.
    title:
        Figure title.
    """
    go = _require_plotly()

    city = market.city
    locs = city.consumer_locations
    traces = []

    # Consumers
    traces.append(
        go.Scatter(
            x=locs[:, 0].tolist(),
            y=locs[:, 1].tolist(),
            mode="markers",
            marker={"size": 3, "color": "lightgrey", "opacity": 0.4},
            name="Consumers",
            hoverinfo="skip",
        )
    )

    # Firms
    for firm in market.firms:
        traces.append(
            go.Scatter(
                x=[firm.location[0]],
                y=[firm.location[1]],
                mode="markers+text",
                marker={"size": 14},
                name=firm.name,
                text=[f"{firm.name}<br>p={firm.price:.2f}<br>{firm.market_share:.1%}"],
                textposition="top center",
            )
        )

    layout = go.Layout(
        title=title,
        xaxis={"range": [0, city.width], "title": "x"},
        yaxis={"range": [0, city.height], "title": "y", "scaleanchor": "x"},
        **kwargs,
    )
    return go.Figure(data=traces, layout=layout)


def interactive_market_shares(agg: Dict[str, Any], title: str = "Market Share Evolution", **kwargs: Any):  # noqa: ANN201
    """Return a Plotly figure with interactive market-share time series."""
    go = _require_plotly()
    traces = []
    steps = agg["steps"]
    for firm_id, series in agg["firms"].items():
        traces.append(
            go.Scatter(x=steps, y=series["market_share"], mode="lines", name=firm_id)
        )
    layout = go.Layout(title=title, xaxis={"title": "Step"}, yaxis={"title": "Market share"}, **kwargs)
    return go.Figure(data=traces, layout=layout)


def interactive_price_evolution(agg: Dict[str, Any], title: str = "Price Evolution", **kwargs: Any):  # noqa: ANN201
    """Return a Plotly figure with interactive price time series."""
    go = _require_plotly()
    traces = []
    steps = agg["steps"]
    for firm_id, series in agg["firms"].items():
        traces.append(go.Scatter(x=steps, y=series["price"], mode="lines", name=firm_id))
    layout = go.Layout(title=title, xaxis={"title": "Step"}, yaxis={"title": "Price"}, **kwargs)
    return go.Figure(data=traces, layout=layout)
