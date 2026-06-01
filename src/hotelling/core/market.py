"""Logit demand and market clearing.

Responsibility: compute logit market shares and firm profits given prices.

Public API: logit_demand, profit, market_clearing, cell_choice_mass, cell_metrics

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


@nb.njit(parallel=True, fastmath=True, cache=True)
def _cell_choice_mass_jit(
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Spatial decomposition of logit demand — parallel numba kernel.

    Mirrors :func:`_logit_demand_jit` exactly in utility computation and
    log-sum-exp stabilisation, but **returns per-cell allocations** rather
    than the column sum.  Designed for choropleth visualisation.

    Parameters
    ----------
    prices, efforts, dist2_km2, cell_pop, lambda_phi, pi_H,
    pi_H_lambda_phi, alpha_L, alpha_H, qualities, beta, transport_cost,
    mu, a0, transport_exponent :
        Identical to the corresponding arguments of ``_logit_demand_jit``.

    Returns
    -------
    inside_mass : ndarray of shape (M, N), float64
        ``inside_mass[i, j]`` is the expected number of consumers in cell *i*
        choosing store *j*  (summed over both income types H and L).
        Column sums equal the output of ``_logit_demand_jit``.
    outside_mass : ndarray of shape (M,), float64
        ``outside_mass[i]`` is the expected number of consumers in cell *i*
        choosing the outside option.

    Notes
    -----
    Two income types are mixed at each cell following
    Anderson, de Palma & Thisse (1992), Ch. 3, with Calvano (2020 §II.A)
    calibration.  The composite weights are::

        w_H[i] = cell_pop[i] * pi_H[i]      + lambda_phi[i] * pi_H_lambda_phi[i]
        w_L[i] = cell_pop[i] * (1-pi_H[i])  + lambda_phi[i] * (1-pi_H_lambda_phi[i])

    Each type's probability allocation follows the standard log-sum-exp
    formula with the outside option utility ``a0 / mu`` stabilised by the
    per-cell row maximum ``v_max``.
    """
    M = dist2_km2.shape[0]
    N = dist2_km2.shape[1]
    inside_mass = np.zeros((M, N))
    outside_mass = np.zeros(M)
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

            # -- log-sum-exp stabiliser (same as _logit_demand_jit) --
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

            outside_exp = np.exp(a0_scaled - v_max)
            exp_sum = outside_exp
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
            outside_mass[i] += w_h * outside_exp * inv_exp_sum
            for j in range(N):
                inside_mass[i, j] += w_h * exp_v[j] * inv_exp_sum

    return inside_mass, outside_mass


def cell_choice_mass(
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Spatial decomposition of logit demand for choropleth visualisation.

    Computes the same utility model as :func:`logit_demand` but returns the
    full ``(M, N)`` allocation matrix instead of only the column sums, plus
    the per-cell outside-option mass.

    Parameters
    ----------
    prices : ndarray of shape (N,)
        Posted prices of *N* stores.
    efforts : ndarray of shape (N,)
        Service-effort levels of *N* stores.
    dist2_km2 : ndarray of shape (M, N)
        Squared network distances (km²) from each of *M* cells to each store.
    cell_pop : ndarray of shape (M,)
        Residential consumer mass per cell.
    lambda_phi : ndarray of shape (M,)
        Footfall (non-resident) consumer mass per cell.
    pi_H : ndarray of shape (M,)
        Fraction of high-income residents per cell.
    pi_H_lambda_phi : ndarray of shape (M,)
        Fraction of high-income footfall consumers per cell.
    alpha : ndarray of shape (2,)
        Income-type quality sensitivities ``[alpha_L, alpha_H]``.
    quality : ndarray of shape (N,)
        Exogenous quality attributes of each store.
    beta : float
        Effort sensitivity parameter.
    transport_cost : float
        Transport disutility coefficient  (Calvano 2020 §II.A).
    mu : float
        Logit scale parameter (taste heterogeneity); default 0.25.
    a0 : float, optional
        Scaled outside-option utility; default 0.0.
    transport_exponent : float, optional
        Exponent applied to ``dist2_km2`` before scaling by
        ``transport_cost``; default 1.0 (linear, ADR-020).

    Returns
    -------
    inside_mass : ndarray of shape (M, N), float64
        Expected consumers in cell *i* choosing store *j*.
        ``inside_mass.sum(axis=0)`` reproduces :func:`logit_demand` exactly.
    outside_mass : ndarray of shape (M,), float64
        Expected consumers in cell *i* choosing the outside option.

    See Also
    --------
    logit_demand : Aggregated column sums (hot path).
    cell_metrics : Thin helper for common choropleth scalars.

    References
    ----------
    Anderson, de Palma & Thisse (1992) *Discrete Choice Theory of Product
    Differentiation*, Ch. 3.
    Calvano, E. et al. (2020) *Artificial Intelligence, Algorithmic Pricing,
    and Collusion*, AER §II.A.
    """
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    efforts = np.ascontiguousarray(efforts, dtype=np.float64)
    dist2_km2 = np.ascontiguousarray(dist2_km2, dtype=np.float64)
    cell_pop = np.ascontiguousarray(cell_pop, dtype=np.float64)
    lambda_phi = np.ascontiguousarray(lambda_phi, dtype=np.float64)
    pi_H = np.ascontiguousarray(pi_H, dtype=np.float64)
    pi_H_lambda_phi = np.ascontiguousarray(pi_H_lambda_phi, dtype=np.float64)
    quality = np.ascontiguousarray(quality, dtype=np.float64)

    assert dist2_km2.shape == (len(cell_pop), len(prices))

    return _cell_choice_mass_jit(
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


def cell_metrics(
    prices: np.ndarray,
    efforts: np.ndarray,
    city: City,
    transport_cost: float,
    metric: str = "expected_price",
) -> np.ndarray:
    """Return a (M,) per-cell scalar array for spatial choropleths.

    Pulls all firm attributes (quality, etc.) from ``city.firms`` in the same
    way as :func:`market_clearing`, then delegates to
    :func:`cell_choice_mass`.

    Parameters
    ----------
    prices : ndarray of shape (N,)
        Posted prices of *N* stores (must match ``len(city.firms)``).
    efforts : ndarray of shape (N,)
        Service-effort levels of *N* stores.
    city : City
        Spatial market container.  ``city.firms`` must be populated.
    transport_cost : float
        Transport disutility coefficient.
    metric : {"expected_price", "served_demand", "dominant_chain",
              "consumer_surplus"}, optional
        Which scalar to compute per cell.  Default is ``"expected_price"``.

        ``expected_price``
            Demand-weighted average price received by residents of cell *i*:
            ``sum_j inside[i,j] * prices[j] / sum_j inside[i,j]``.
            Returns ``NaN`` for cells with no inside mass.
        ``served_demand``
            Total expected consumers in cell *i* who choose any store:
            ``sum_j inside[i,j]``.
        ``dominant_chain``
            Index of the store attracting the largest share in cell *i*
            (``argmax_j inside[i,j]``).  Caller maps integer indices to chain
            labels via ``[f.chain for f in city.firms]``.
        ``consumer_surplus``
            Per-cell logsum (inclusive value) in price units, weighted by
            consumer mass and averaged across income types, following
            Anderson, de Palma & Thisse (1992) Ch. 3.

    Returns
    -------
    result : ndarray of shape (M,)
        Per-cell metric values; dtype float64 for all metrics.

    Raises
    ------
    ValueError
        If *metric* is not one of the four recognised strings.

    References
    ----------
    Anderson, de Palma & Thisse (1992) *Discrete Choice Theory of Product
    Differentiation*, Ch. 3.
    Calvano, E. et al. (2020) *Artificial Intelligence, Algorithmic Pricing,
    and Collusion*, AER §II.A.
    """
    firms = city.firms
    qualities = np.ascontiguousarray(
        [f.quality for f in firms], dtype=np.float64
    )
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    efforts = np.ascontiguousarray(efforts, dtype=np.float64)
    transport_exponent = getattr(city, "transport_exponent", 1.0)

    inside, outside = cell_choice_mass(
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
        transport_exponent=transport_exponent,
    )

    if metric == "expected_price":
        row_sum = inside.sum(axis=1)
        with np.errstate(invalid="ignore"):
            result = (inside @ prices) / row_sum
        return np.where(row_sum > 0.0, result, np.nan)

    elif metric == "served_demand":
        return inside.sum(axis=1)

    elif metric == "dominant_chain":
        return inside.argmax(axis=1).astype(np.float64)

    elif metric == "consumer_surplus":
        # Per-cell inclusive value (logsum) in price units, averaged per consumer.
        # For type h at cell i:  CS_h(i) = mu * log(sum_j exp(v_jh/mu) + exp(a0/mu))
        # Weighted average over types:
        #   CS(i) = [w_H(i)*CS_H(i) + w_L(i)*CS_L(i)] / [cell_pop(i)+lambda_phi(i)]
        dist2 = np.ascontiguousarray(city.dist2_km2, dtype=np.float64)
        inv_mu = 1.0 / city.mu
        a0_scaled = city.a0 * inv_mu
        td = transport_cost * (dist2 ** transport_exponent)  # (M, N)

        w_H = city.cell_pop * city.pi_H + city.lambda_phi * city.pi_H_lambda_phi
        w_L = city.cell_pop * (1.0 - city.pi_H) + city.lambda_phi * (1.0 - city.pi_H_lambda_phi)
        total_w = city.cell_pop + city.lambda_phi

        result = np.zeros(len(city.cell_pop))
        for alpha_h, w_h in (
            (float(city.alpha[0]), w_L),
            (float(city.alpha[1]), w_H),
        ):
            # Utilities (M, N), scaled by inv_mu
            v = (
                alpha_h * qualities[np.newaxis, :]
                + city.beta * efforts[np.newaxis, :]
                - prices[np.newaxis, :]
                - td
            ) * inv_mu
            # Append outside option as an (M,1) column
            v_all = np.concatenate(
                [v, np.full((v.shape[0], 1), a0_scaled)], axis=1
            )
            v_max = v_all.max(axis=1, keepdims=True)
            logsum = np.log(np.exp(v_all - v_max).sum(axis=1)) + v_max[:, 0]
            result += w_h * (city.mu * logsum)

        with np.errstate(invalid="ignore"):
            return np.where(total_w > 0.0, result / total_w, np.nan)

    else:
        raise ValueError(
            f"Unknown metric {metric!r}. "
            "Choose from: expected_price, served_demand, dominant_chain, "
            "consumer_surplus"
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
