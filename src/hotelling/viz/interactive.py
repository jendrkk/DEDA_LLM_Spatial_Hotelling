"""Interactive visualizations: folium, plotly, pydeck.

Responsibility: generate interactive HTML maps and charts for Jupyter
notebooks and the marimo exploration app.

Public API: folium_choropleth, plotly_price_facets, pydeck_entry_heatmap

Key dependencies: folium, plotly, pydeck (all optional), pandas

References:
    folium https://python-visualization.github.io/folium/;
    plotly https://plotly.com/python/;
    pydeck https://deckgl.readthedocs.io/.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def folium_choropleth(
    city: Any,
    metric_series: pd.Series,
    title: str = "Market Share",
) -> Any:
    """Render an interactive folium choropleth map of a spatial metric.

    Parameters
    ----------
    city : City instance with GeoDataFrame boundary (or GeoDataFrame directly)
    metric_series : pandas Series indexed by grid cell or LOR unit id
    title : map title / legend label

    Returns
    -------
    folium.Map
    """
    raise NotImplementedError


def plotly_price_facets(
    results_df: pd.DataFrame,
    facet_col: str = "agent_id",
    title: str = "Price Evolution",
) -> Any:
    """Render faceted plotly chart of price evolution over simulation steps.

    Parameters
    ----------
    results_df : DataFrame with columns: period, agent_id, price
    facet_col : column to use for faceting (one subplot per value)
    title : chart title

    Returns
    -------
    plotly.graph_objects.Figure
    """
    raise NotImplementedError


def pydeck_entry_heatmap(
    entry_df: pd.DataFrame,
    map_style: str = "mapbox://styles/mapbox/dark-v10",
) -> Any:
    """Render a pydeck 3-D heatmap of LLM entry location choices.

    Parameters
    ----------
    entry_df : DataFrame with columns: lon (float), lat (float), weight (float)
    map_style : Mapbox style URL

    Returns
    -------
    pydeck.Deck
    """
    raise NotImplementedError
