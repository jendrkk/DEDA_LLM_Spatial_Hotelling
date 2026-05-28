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

import dataclasses
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from hotelling.core.city import City
from hotelling.core.market import market_clearing


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

    Returns
    -------
    prices : np.ndarray shape (N,) equilibrium prices
    efforts : np.ndarray shape (N,) equilibrium efforts
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        cached = _load_benchmark_cache(cache_path, "nash")
        if cached is not None:
            return cached

    firms = city.firms

    total_pop = (city.cell_pop + city.lambda_phi).sum()

    N = len(firms)
    costs = np.array([firm.marginal_cost for firm in firms])
    kappa0 = np.array([firm.kappa0 for firm in firms])
    beta = city.beta

    prices = costs.copy()
    efforts = np.zeros(N)
    converged = False

    for _ in range(max_iter):
        demands, profits = market_clearing(
            prices=prices,
            efforts=efforts,
            city=city,
            transport_cost=transport_cost,
        )

        shares = demands / total_pop
        new_prices = costs + city.mu / np.clip(1 - shares, 1e-9, None)
        new_efforts = beta * demands / kappa0

        converged = (
            np.max(np.abs(new_prices - prices)) < tol
            and np.max(np.abs(new_efforts - efforts)) < tol
        )

        prices, efforts = new_prices, new_efforts

        if converged:
            break

    if not converged:
        warnings.warn(
            f"Bertrand-Nash equilibrium not found after max_iter {max_iter} iterations"
        )

    if cache_path is not None:
        _save_benchmark_cache(cache_path, "nash", prices, efforts)

    return prices, efforts


def joint_monopoly(
    city: City,
    transport_cost: float = 1.0,
    *,
    cache_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find joint-monopoly (cartel) prices maximising total profit.

    Uses a K-firm spatial subsample (K <= 30) as a proxy for large N, then
    broadcasts the mean optimized price/effort to all N firms.  Suitable for
    scalar benchmark quantities (e.g. Calvano Δ).

    Returns
    -------
    prices : np.ndarray shape (N,) equilibrium prices
    efforts : np.ndarray shape (N,) equilibrium efforts
    """
    from scipy.optimize import minimize

    if cache_path is not None:
        cache_path = Path(cache_path)
        cached = _load_benchmark_cache(cache_path, "mono")
        if cached is not None:
            return cached

    firms = city.firms
    N = len(firms)
    costs = np.array([firm.marginal_cost for firm in firms])
    kappa0 = np.array([firm.kappa0 for firm in firms])

    K = min(30, N)
    idx = np.round(np.linspace(0, N - 1, K)).astype(int)
    proxy_firms = [city.firms[i] for i in idx]
    proxy_dist2 = city.dist2_km2[:, idx]
    proxy_city = dataclasses.replace(
        city,
        firms=proxy_firms,
        dist2_km2=proxy_dist2,
    )

    proxy_costs = costs[idx]
    proxy_kappa0 = kappa0[idx]

    def neg_total_profit(x: np.ndarray) -> float:
        prices_k, efforts_k = x[:K], x[K:]
        _, profits = market_clearing(
            prices=prices_k,
            efforts=efforts_k,
            city=proxy_city,
            transport_cost=transport_cost,
        )
        return -float(profits.sum())

    x0 = np.concatenate([proxy_costs + 3 * city.mu, np.zeros(K)])
    bounds = [(c, c + 20 * city.mu) for c in proxy_costs] + [(0, 10.0)] * K

    res = minimize(
        neg_total_profit,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={
            "ftol": 1e-6,
            "gtol": 1e-5,
            "maxiter": 300,
            "maxfun": 600,
        },
    )

    if not res.success:
        warnings.warn(
            f"Joint-monopoly optimizer did not converge: {res.message}",
            RuntimeWarning,
        )

    optimized_prices = res.x[:K]
    optimized_efforts = res.x[K:]

    prices = np.full(N, float(optimized_prices.mean()))
    efforts = np.full(N, float(optimized_efforts.mean()))

    if cache_path is not None:
        _save_benchmark_cache(cache_path, "mono", prices, efforts)

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
