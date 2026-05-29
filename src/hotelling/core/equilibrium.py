"""Equilibrium solvers: Bertrand-Nash, joint monopoly, Tabuchi 2-D benchmark.

Responsibility: compute theoretical equilibrium benchmarks for the spatial
Hotelling model.

Public API: bertrand_nash, joint_monopoly, tabuchi_2d_benchmark

Key dependencies: numpy, scipy.optimize, numba, hotelling.core.city

References:
    Calvano et al. (2020 AER);
    Tabuchi (1994) JUE;
    Bertrand (1883).
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numba as nb
import numpy as np

from hotelling.core.city import City
from hotelling.core.market import market_clearing


def _param_signature(city: City, transport_cost: float) -> str:
    """Stable short hash of all economic parameters that move the equilibrium."""
    parts = [
        f"tc={transport_cost!r}",
        f"mu={city.mu!r}",
        f"a0={city.a0!r}",
        f"beta={city.beta!r}",
        f"alpha={np.asarray(city.alpha).tolist()!r}",
        f"N={len(city.firms)}",
        f"q={[round(f.quality,6) for f in city.firms]!r}",
        f"c={[round(f.marginal_cost,6) for f in city.firms]!r}",
        f"kappa={[round(f.kappa0,6) for f in city.firms]!r}",
        f"pop_sum={float(np.asarray(city.cell_pop).sum()):.6f}",
        f"lphi_sum={float(np.asarray(city.lambda_phi).sum()):.6f}",
        f"dist_sum={float(np.asarray(city.dist2_km2).sum()):.6f}",
    ]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


@nb.njit(parallel=True, fastmath=True, cache=True)
def _demand_and_elasticity_jit(prices, efforts, dist2, cell_pop, lambda_phi,
                               pi_H, pi_H_lphi, alpha_L, alpha_H, qualities,
                               beta, transport_cost, mu, a0, transport_exponent):
    M = dist2.shape[0]; N = dist2.shape[1]
    Dc = np.zeros((M, N)); Ec = np.zeros((M, N))
    inv_mu = 1.0 / mu; a0s = a0 * inv_mu
    for i in nb.prange(M):
        w_H = cell_pop[i] * pi_H[i] + lambda_phi[i] * pi_H_lphi[i]
        w_L = cell_pop[i] * (1.0 - pi_H[i]) + lambda_phi[i] * (1.0 - pi_H_lphi[i])
        for h in range(2):
            alpha_h = alpha_L if h == 0 else alpha_H
            w_h = w_L if h == 0 else w_H
            vmax = a0s
            for j in range(N):
                vj = (alpha_h * qualities[j] + beta * efforts[j] - prices[j]
                      - transport_cost * dist2[i, j] ** transport_exponent) * inv_mu
                if vj > vmax: vmax = vj
            denom = np.exp(a0s - vmax); ev = np.empty(N)
            for j in range(N):
                vj = (alpha_h * qualities[j] + beta * efforts[j] - prices[j]
                      - transport_cost * dist2[i, j] ** transport_exponent) * inv_mu
                ev[j] = np.exp(vj - vmax); denom += ev[j]
            inv_d = 1.0 / denom
            for j in range(N):
                s = ev[j] * inv_d
                Dc[i, j] += w_h * s
                Ec[i, j] += w_h * s * (1.0 - s)
    return Dc.sum(axis=0), Ec.sum(axis=0)


@nb.njit(parallel=True, fastmath=True, cache=True)
def _monopoly_demand_grad_jit(prices, efforts, costs, dist2, cell_pop, lambda_phi,
                              pi_H, pi_H_lphi, alpha_L, alpha_H, qualities,
                              beta, transport_cost, mu, a0, transport_exponent):
    M = dist2.shape[0]; N = dist2.shape[1]
    Dc = np.zeros((M, N)); Gc = np.zeros((M, N))
    inv_mu = 1.0 / mu; a0s = a0 * inv_mu
    for i in nb.prange(M):
        w_H = cell_pop[i] * pi_H[i] + lambda_phi[i] * pi_H_lphi[i]
        w_L = cell_pop[i] * (1.0 - pi_H[i]) + lambda_phi[i] * (1.0 - pi_H_lphi[i])
        for h in range(2):
            alpha_h = alpha_L if h == 0 else alpha_H
            w_h = w_L if h == 0 else w_H
            vmax = a0s
            for j in range(N):
                vj = (alpha_h * qualities[j] + beta * efforts[j] - prices[j]
                      - transport_cost * dist2[i, j] ** transport_exponent) * inv_mu
                if vj > vmax: vmax = vj
            denom = np.exp(a0s - vmax); ev = np.empty(N)
            for j in range(N):
                vj = (alpha_h * qualities[j] + beta * efforts[j] - prices[j]
                      - transport_cost * dist2[i, j] ** transport_exponent) * inv_mu
                ev[j] = np.exp(vj - vmax); denom += ev[j]
            inv_d = 1.0 / denom
            m_hi = 0.0
            for j in range(N):
                m_hi += (prices[j] - costs[j]) * (ev[j] * inv_d)
            for j in range(N):
                s = ev[j] * inv_d
                Dc[i, j] += w_h * s
                Gc[i, j] += w_h * inv_mu * s * (m_hi - (prices[j] - costs[j]))
    return Dc.sum(axis=0), Gc.sum(axis=0)


def _load_benchmark_cache(
    cache_path: Path,
    prefix: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not cache_path.exists():
        return None
    data = np.load(cache_path)
    prices_key = f"{prefix}_prices"
    efforts_key = f"{prefix}_efforts"
    if prices_key in data and efforts_key in data:
        return data[prices_key], data[efforts_key]
    return None


def _save_benchmark_cache(
    cache_path: Path,
    prefix: str,
    prices: np.ndarray,
    efforts: np.ndarray,
) -> None:
    merged: Dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as data:
            merged = {key: data[key] for key in data.files}
    merged[f"{prefix}_prices"] = prices
    merged[f"{prefix}_efforts"] = efforts
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **merged)


def bertrand_nash(
    city: City,
    transport_cost: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 500,
    *,
    cache_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find Bertrand-Nash equilibrium prices by iterating best responses.

    Uses the elasticity-correct single-product logit FOC with heterogeneous
    consumers:  p_j - c_j = mu * D_j / E_j  where D_j = sum_i w_ij s_ij and
    E_j = sum_i w_ij s_ij (1 - s_ij).

    Returns
    -------
    prices : np.ndarray shape (N,) equilibrium prices
    efforts : np.ndarray shape (N,) equilibrium efforts
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        _sig = _param_signature(city, transport_cost)
        cached = _load_benchmark_cache(cache_path, f"nash_{_sig}")
        if cached is not None:
            return cached

    firms = city.firms
    N = len(firms)
    costs   = np.array([f.marginal_cost for f in firms], dtype=np.float64)
    kappa0  = np.array([f.kappa0        for f in firms], dtype=np.float64)
    quals   = np.array([f.quality       for f in firms], dtype=np.float64)
    beta    = city.beta
    prices  = costs.copy()
    efforts = np.zeros(N)
    converged = False

    for _ in range(max_iter):
        D, E = _demand_and_elasticity_jit(
            prices, efforts, city.dist2_km2, city.cell_pop, city.lambda_phi,
            city.pi_H, city.pi_H_lambda_phi,
            float(city.alpha[0]), float(city.alpha[1]),
            quals, float(beta), float(transport_cost), float(city.mu), float(city.a0),
            float(getattr(city, "transport_exponent", 1.0)))
        new_prices  = costs + city.mu * D / np.clip(E, 1e-12, None)
        new_efforts = beta * D / kappa0
        converged = (np.max(np.abs(new_prices - prices)) < tol
                     and np.max(np.abs(new_efforts - efforts)) < tol)
        prices, efforts = new_prices, new_efforts
        if converged:
            break

    if not converged:
        warnings.warn(f"Bertrand-Nash not converged after {max_iter} iters")

    if cache_path is not None:
        _save_benchmark_cache(cache_path, f"nash_{_sig}", prices, efforts)

    return prices, efforts


def joint_monopoly(
    city: City,
    transport_cost: float = 1.0,
    *,
    cache_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find joint-monopoly (cartel) prices maximising total profit.

    Optimizes all N prices jointly using the analytic gradient of total profit
    from the spatial logit model.

    Returns
    -------
    prices : np.ndarray shape (N,) equilibrium prices
    efforts : np.ndarray shape (N,) equilibrium efforts (zero at benchmark)
    """
    from scipy.optimize import minimize

    if cache_path is not None:
        cache_path = Path(cache_path)
        _sig = _param_signature(city, transport_cost)
        cached = _load_benchmark_cache(cache_path, f"mono_{_sig}")
        if cached is not None:
            return cached

    firms = city.firms
    N = len(firms)
    costs   = np.array([f.marginal_cost for f in firms], dtype=np.float64)
    quals   = np.array([f.quality       for f in firms], dtype=np.float64)
    efforts = np.zeros(N)

    def neg_obj_and_grad(p: np.ndarray) -> Tuple[float, np.ndarray]:
        D, G = _monopoly_demand_grad_jit(
            p, efforts, costs, city.dist2_km2, city.cell_pop, city.lambda_phi,
            city.pi_H, city.pi_H_lambda_phi,
            float(city.alpha[0]), float(city.alpha[1]),
            quals, float(city.beta), float(transport_cost), float(city.mu), float(city.a0),
            float(getattr(city, "transport_exponent", 1.0)))
        profit_val = float(np.sum((p - costs) * D))
        grad = D + G
        return -profit_val, -grad

    x0 = costs + 3.0 * city.mu
    bounds = [(float(c), float(c) + 50.0 * city.mu) for c in costs]
    res = minimize(neg_obj_and_grad, x0, jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"ftol": 1e-9, "gtol": 1e-6, "maxiter": 500})

    if not res.success:
        warnings.warn(f"Joint-monopoly optimizer did not converge: {res.message}",
                      RuntimeWarning)

    prices  = res.x.astype(np.float64)
    efforts = np.zeros(N)

    if cache_path is not None:
        _save_benchmark_cache(cache_path, f"mono_{_sig}", prices, efforts)

    return prices, efforts


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
    if n == 1:
        warnings.warn(
            "Tabuchi (1994) symmetric 2-D benchmark is undefined for n=1 "
            "(monopoly markup unbounded in pure logit)",
            RuntimeWarning,
        )
        return np.inf, np.inf

    markup = n * mu / (n - 1)
    avg_dist = t / (4 * n)
    price = markup + avg_dist
    profit = markup / n
    return price, profit
