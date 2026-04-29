"""Equilibrium solvers: Bertrand-Nash, joint monopoly, Tabuchi 2-D benchmark.

Responsibility: compute theoretical equilibrium benchmarks for the spatial
Hotelling model.

Public API: bertrand_nash, joint_monopoly, tabuchi_2d_benchmark

Key dependencies: numpy, scipy.optimize, hotelling.core.city

References:
    Calvano et al. (2020 AER);
    Tabuchi (1994) JUE;
    Bertrand (1883).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from hotelling.core.city import City
from hotelling.core.market import market_clearing


def bertrand_nash(
    city: City,
    transport_cost: float = 1.0,
    mu: float = 0.25,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find Bertrand-Nash equilibrium prices by iterating best responses.

    Returns
    -------
    prices : np.ndarray shape (N,) equilibrium prices
    efforts : np.ndarray shape (N,) equilibrium efforts
    """
    
    firms = city.firms
    
    total_pop = (city.cell_pop + city.lambda_phi).sum()
    
    N = len(firms)
    costs = np.array([firm.marginal_cost for firm in firms])
    kappa0 = np.array([firm.kappa0 for firm in firms])
    beta = city.beta
    
    # Initialize prices and efforts
    prices = costs.copy()
    efforts = np.zeros(N)
    
    # Iterate best responses
    for _ in range(max_iter):
        # Compute demands and profits
        demands, profits = market_clearing(
            prices = prices, efforts = efforts,
            city = city, transport_cost = transport_cost, mu = mu
        )
        
        # Closed-from best response update
        shares = demands / total_pop
        new_prices = costs + mu / np.clip(1-shares, 1e-9, None)
        new_efforts = beta * demands / kappa0
        
        if np.max(np.abs(new_prices - prices)) < tol and np.max(np.abs(new_efforts - efforts)) < tol:
            prices, efforts = new_prices, new_efforts
            break
        
        prices, efforts = new_prices, new_efforts
    
    return prices, efforts # (N,), (N,)


def joint_monopoly(
    city: City,
    transport_cost: float = 1.0,
    mu: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find joint-monopoly (cartel) prices maximising total profit.

    Returns
    -------
    prices : np.ndarray shape (N,) equilibrium prices
    efforts : np.ndarray shape (N,) equilibrium efforts
    """
    from scipy.optimize import minimize
    
    firms  = city.firms
    N      = len(firms)
    costs  = np.array([firm.marginal_cost for firm in firms])
    kappa0 = np.array([firm.kappa0 for firm in firms])
    
    def neg_total_profit(x):
        prices, efforts = x[:N], x[N:]
        demands, profits = market_clearing(
            prices = prices, efforts = efforts,
            city = city, transport_cost = transport_cost, mu = mu
        )
        return -profits.sum()
    
    x0     = np.concatenate([costs + 2*mu, np.zeros(N)])      # initial guess: 2*mu above marginal cost, 0 effort
    bounds = [(c, c * 10) for c in costs] + [(0, None)] * N   # bounds: prices ≥ cost, efforts ≥ 0
    
    res = minimize(
        neg_total_profit, x0, bounds = bounds, method = 'L-BFGS-B',
        options = {
            'tol': 1e-10,
            'maxiter': 2000
        }
    )
    
    prices = res.x[:N]
    efforts = res.x[N:]
    
    return prices, efforts # (N,), (N,)

def tabuchi_2d_benchmark(
    n: int = 2,
    t: float = 0.5,
    mu: float = 0.25,
) -> Tuple[float, float]:
    """Return (equilibrium_price, equilibrium_profit) for Tabuchi (1994) symmetric 2-D case.

    Returns
    -------
    (price, profit) tuple for the symmetric case
    """
    # Symmetric logit duopoly markup: p* - c = μ/(1 - 1/N) = Nμ/(N-1)
    # Spatial term: expected quadratic transport cost for uniform consumers
    # For N equidistant firms on unit square ≈ t/(4N) (Tabuchi 1994 approx.)
    markup   = n * mu / (n - 1)          # logit markup
    avg_dist = t / (4 * n)              # spatial differentiation term (approx)
    price    = markup + avg_dist
    profit   = markup / n               # symmetric share = 1/N
    return price, profit
