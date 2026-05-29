"""Logit demand and market clearing.

Responsibility: compute logit market shares and firm profits given prices.

Public API: logit_demand, profit, market_clearing

Key dependencies: numpy, numba, hotelling.core.city, hotelling.core.firm

References:
    Calvano et al. (2020 AER) §II.A;
    Anderson, de Palma, Thisse (1992) - spatial logit extension.
"""
from __future__ import annotations

from typing import Tuple

import numba as nb
import numpy as np

from hotelling.core.city import City


@nb.njit(parallel=True, fastmath=True, cache=True)
def _logit_demand_jit(
    prices: np.ndarray,
    efforts: np.ndarray,
    dist2_km2: np.ndarray,
    cell_pop: np.ndarray,
    lambda_phi: np.ndarray,
    pi_H: np.ndarray,
    pi_H_lambda_phi: np.ndarray,
    alpha_L: float,
    alpha_H: float,
    qualities: np.ndarray,
    beta: float,
    transport_cost: float,
    mu: float,
    a0: float,
    transport_exponent: float,
) -> np.ndarray:
    M = dist2_km2.shape[0]
    N = dist2_km2.shape[1]
    # Per-cell contributions (parallel-safe: one row per cell, no shared writes)
    cell_contrib = np.zeros((M, N))
    inv_mu = 1.0 / mu
    a0_scaled = a0 * inv_mu

    for i in nb.prange(M):
        w_H_res = cell_pop[i] * pi_H[i]
        w_L_res = cell_pop[i] * (1.0 - pi_H[i])
        w_H_phi = lambda_phi[i] * pi_H_lambda_phi[i]
        w_L_phi = lambda_phi[i] * (1.0 - pi_H_lambda_phi[i])

        for h in range(2):
            alpha_h = alpha_L if h == 0 else alpha_H
            w_h = (w_L_res + w_L_phi) if h == 0 else (w_H_res + w_H_phi)

            v_max = a0_scaled
            for j in range(N):
                v_j = (
                    alpha_h * qualities[j]
                    + beta * efforts[j]
                    - prices[j]
                    - transport_cost * dist2_km2[i, j] ** transport_exponent
                ) * inv_mu
                if v_j > v_max:
                    v_max = v_j

            exp_sum = np.exp(a0_scaled - v_max)
            exp_v = np.empty(N)
            for j in range(N):
                v_j = (
                    alpha_h * qualities[j]
                    + beta * efforts[j]
                    - prices[j]
                    - transport_cost * dist2_km2[i, j] ** transport_exponent
                ) * inv_mu
                exp_v[j] = np.exp(v_j - v_max)
                exp_sum += exp_v[j]

            inv_exp_sum = 1.0 / exp_sum
            for j in range(N):
                cell_contrib[i, j] += w_h * exp_v[j] * inv_exp_sum

    return cell_contrib.sum(axis=0)


def logit_demand(
    prices: np.ndarray,
    efforts: np.ndarray,
    dist2_km2: np.ndarray,
    cell_pop: np.ndarray,
    lambda_phi: np.ndarray,
    pi_H: np.ndarray,
    pi_H_lambda_phi: np.ndarray,
    alpha: np.ndarray,
    quality: np.ndarray,
    beta: float,
    transport_cost: float,
    mu: float,
    a0: float = 0.0,
    transport_exponent: float = 1.0,
) -> np.ndarray:
    """Compute logit market shares for N firms at given prices."""
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    efforts = np.ascontiguousarray(efforts, dtype=np.float64)
    dist2_km2 = np.ascontiguousarray(dist2_km2, dtype=np.float64)
    cell_pop = np.ascontiguousarray(cell_pop, dtype=np.float64)
    lambda_phi = np.ascontiguousarray(lambda_phi, dtype=np.float64)
    pi_H = np.ascontiguousarray(pi_H, dtype=np.float64)
    pi_H_lambda_phi = np.ascontiguousarray(pi_H_lambda_phi, dtype=np.float64)
    quality = np.ascontiguousarray(quality, dtype=np.float64)

    assert dist2_km2.shape == (len(cell_pop), len(prices))

    return _logit_demand_jit(
        prices,
        efforts,
        dist2_km2,
        cell_pop,
        lambda_phi,
        pi_H,
        pi_H_lambda_phi,
        float(alpha[0]),
        float(alpha[1]),
        quality,
        float(beta),
        float(transport_cost),
        float(mu),
        float(a0),
        float(transport_exponent),
    )


def profit(
    price: np.ndarray | float,
    demand: np.ndarray | float,
    marginal_cost: np.ndarray | float,
    kappa0: float,
    effort: np.ndarray | float,
    size: np.ndarray | float,
    rent: np.ndarray | float = 0.0,
) -> float | np.ndarray:
    """Compute firm profit = (p - c) * demand - 0.5 * kappa0 * effort**2 - rent * size."""
    return (price - marginal_cost) * demand - 0.5 * kappa0 * effort**2 - rent * size


def market_clearing(
    prices: np.ndarray,
    efforts: np.ndarray,
    city: City,
    transport_cost: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute equilibrium demands and profits for all firms."""
    firms = city.firms

    qualities = np.ascontiguousarray(
        [firm.quality for firm in firms], dtype=np.float64
    )
    marginal_costs = np.ascontiguousarray(
        [firm.marginal_cost for firm in firms], dtype=np.float64
    )
    kappa0 = np.ascontiguousarray([firm.kappa0 for firm in firms], dtype=np.float64)
    sizes = np.ascontiguousarray([firm.size for firm in firms], dtype=np.float64)
    rents = np.ascontiguousarray([firm.rent for firm in firms], dtype=np.float64)

    prices = np.ascontiguousarray(prices, dtype=np.float64)
    efforts = np.ascontiguousarray(efforts, dtype=np.float64)

    demands = logit_demand(
        prices=prices,
        efforts=efforts,
        dist2_km2=city.dist2_km2,
        cell_pop=city.cell_pop,
        lambda_phi=city.lambda_phi,
        pi_H=city.pi_H,
        pi_H_lambda_phi=city.pi_H_lambda_phi,
        alpha=city.alpha,
        quality=qualities,
        beta=city.beta,
        transport_cost=transport_cost,
        mu=city.mu,
        a0=city.a0,
        transport_exponent=getattr(city, "transport_exponent", 1.0),
    )

    profits = profit(
        price=prices,
        demand=demands,
        marginal_cost=marginal_costs,
        kappa0=kappa0,
        effort=efforts,
        size=sizes,
        rent=rents,
    )

    return demands, profits
