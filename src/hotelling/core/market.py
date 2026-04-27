"""Logit demand and market clearing.

Responsibility: compute logit market shares and firm profits given prices.

Public API: logit_demand, profit, market_clearing

Key dependencies: numpy, hotelling.core.city, hotelling.core.firm

References:
    Calvano et al. (2020 AER) §II.A;
    Anderson, de Palma, Thisse (1992) - spatial logit extension.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from hotelling.core.city import City
from hotelling.core.firm import Firm


def logit_demand(
    prices: np.ndarray,
    locations: np.ndarray,
    consumer_locations: np.ndarray,
    qualities: np.ndarray,
    transport_cost: float,
    mu: float,
) -> np.ndarray:
    """Compute logit market shares for N firms at given prices.

    Parameters
    ----------
    prices : shape (N,) - prices for each firm
    locations : shape (N, 2) - firm locations
    consumer_locations : shape (M, 2) - consumer locations
    qualities : shape (N,) - quality levels
    transport_cost : float - transport cost per unit distance
    mu : float - logit scale (taste heterogeneity)

    Returns
    -------
    np.ndarray shape (N,) - aggregate market shares summing to <= 1
    """
    raise NotImplementedError


def profit(
    price: float,
    demand: float,
    marginal_cost: float,
) -> float:
    """Compute firm profit = (p - c) * demand."""
    raise NotImplementedError


def market_clearing(
    prices: np.ndarray,
    city: City,
    transport_cost: float = 1.0,
    mu: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute equilibrium demands and profits for all firms.

    Parameters
    ----------
    prices : shape (N,) array of prices for each firm in city.firms
    city : City instance with firms and population_grid
    transport_cost : transport cost parameter t
    mu : logit scale mu

    Returns
    -------
    demands : np.ndarray shape (N,) market shares
    profits : np.ndarray shape (N,) per-firm profits
    """
    raise NotImplementedError
