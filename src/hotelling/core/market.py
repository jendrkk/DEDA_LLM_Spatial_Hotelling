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
    prices: np.ndarray,          # (N,)
    efforts: np.ndarray,         # (N,)
    dist2_km2: np.ndarray,       # (M, N) — precomputed squared distances in km²
    cell_pop: np.ndarray,        # (M,)   — ω_i
    lambda_phi: np.ndarray,      # (M,)   — λ·φ_i
    pi_H: np.ndarray,            # (M,)   — S_{r(i)}, H-type share per cell
    pi_H_lambda_phi: np.ndarray, # (M,) — π_H for λ·φ_i
    alpha: np.ndarray,           # (2,)   — [α_L, α_H]
    quality: np.ndarray,         # (N,)   — q_{θ_c(j)}
    beta: float,
    transport_cost: float,
    mu: float,
    a0: float = 0.0,
) -> np.ndarray:
    """Compute logit market shares for N firms at given prices.

    Parameters
    ----------
    prices : shape (N,) - prices for each firm
    efforts : shape (N,) - store efforts
    dist2_km2 : shape (M, N) - precomputed squared distances in km²
    cell_pop : shape (M,) - cell population
    lambda_phi : shape (M,) - prime-location population adjustment
    pi_H : shape (M,) - S_{r(i)}, H-type share per cell
    pi_H_lambda_phi : shape (M,) - π_H for λ·φ_i
    alpha : shape (2,) - [α_L, α_H] - type-specific WTP for quality
    quality : shape (N,) - chain-type quality intercepts
    beta : float - homogeneous marginal utility of store effort
    transport_cost : float - transport cost in €/km²
    mu : float - logit scale (taste heterogeneity)
    a0 : float - outside option utility (default 0.0)

    Returns
    -------
    np.ndarray shape (N,) - market shares for each firm
    """
    
    N = len(prices)
    M = len(cell_pop)
    
    # V_{hij}
    
    V = (
        alpha[:, None, None] * quality[None, None, :] +
        beta * efforts[None, None, :] -
        prices[None, :, None] -
        transport_cost * dist2_km2[None, :, :]
    )
    
    # Append outside option utility
    V = np.concatenate([V, np.full((2, len(cell_pop), 1), a0)], axis=2)
    V = V/mu
    V -= V.max(axis=2, keepdims=True) # Normalize to avoid overflow
    exp_V = np.exp(V)
    prob = exp_V[:, :, :N] / exp_V.sum(axis=2, keepdims=True)  # (2, M, N)
    
    pi = np.stack([1-pi_H, pi_H], axis=0) # (2, M)
    pi_lambda_phi = np.stack([1-pi_H_lambda_phi, pi_H_lambda_phi], axis=0) # (2, M)

    weights = cell_pop[None,:] * pi + lambda_phi[None,:] * pi_lambda_phi # (2, M)

    return np.einsum("hi, hij -> j", weights, prob) # (N,)

def profit(
    price: np.ndarray | float,
    demand: np.ndarray | float,
    marginal_cost: np.ndarray | float,
    kappa0: float,
    effort: np.ndarray | float,
    size: np.ndarray | float,
    rent: np.ndarray | float = 0.0,
) -> float:
    """Compute firm profit = (p - c) * demand - 0.5 * kappa0 * effort**2 - rent * size."""
    
    return (price - marginal_cost) * demand - 0.5 * kappa0 * effort**2 - rent * size

def market_clearing(
    prices: np.ndarray,   # (N,) 
    efforts: np.ndarray,  # (N,)
    city: City,
    transport_cost: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute equilibrium demands and profits for all firms.

    Parameters
    ----------
    prices : shape (N,) array of prices for each firm in city.firms
    efforts : shape (N,) array of store efforts for each firm in city.firms
    city : City instance with firms and population_grid
    transport_cost : transport cost parameter t
    Returns
    -------
    demands : np.ndarray shape (N,) market demands
    profits : np.ndarray shape (N,) per-firm profits
    """
    
    firms = city.firms
    
    qualities      = np.array([firm.quality for firm in firms])        # (N,)
    marginal_costs = np.array([firm.marginal_cost for firm in firms])  # (N,)
    kappa0         = np.array([firm.kappa0 for firm in firms])         # (N,)
    sizes          = np.array([firm.size for firm in firms])           # (N,)
    rents          = np.array([firm.rent for firm in firms])           # (N,)
    
    demands = logit_demand(
        prices = prices, efforts=efforts,
        dist2_km2 = city.dist2_km2, cell_pop = city.cell_pop,
        lambda_phi = city.lambda_phi, pi_H = city.pi_H, pi_H_lambda_phi = city.pi_H_lambda_phi,
        alpha = city.alpha, quality = qualities, beta = city.beta, transport_cost = transport_cost, mu = city.mu, a0 = city.a0
    )
    
    profits = profit(
        price = prices, demand = demands, marginal_cost = marginal_costs, kappa0 = kappa0, effort = efforts, size = sizes, rent = rents
    )
    
    return demands, profits # (N,), (N,)
