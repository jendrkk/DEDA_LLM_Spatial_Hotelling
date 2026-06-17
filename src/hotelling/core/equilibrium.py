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
from hotelling.core.market import catchment_demand, market_clearing


def _param_signature(city: City, transport_cost: float) -> str:
    """Stable short hash of all economic parameters that move the equilibrium.

    Includes catchment geometry stats when the sparse path is active so the
    benchmark cache is invalidated when ``catchment_minutes / k_min / k_max``
    change (different truncation → different equilibrium prices).
    """
    if city.dist2_km2 is not None:
        dist_str = f"dist_sum={float(np.asarray(city.dist2_km2).sum()):.6f}"
    elif city.catch_indptr is not None:
        nnz = int(city.catch_indptr[-1])
        if city.catch_tt is not None and nnz > 0:
            tt_sum = float(city.catch_tt.sum())
            tt_max = float(city.catch_tt.max())
        else:
            tt_sum, tt_max = 0.0, 0.0
        dist_str = f"catch_nnz={nnz},tt_sum={tt_sum:.3f},tt_max={tt_max:.3f}"
    else:
        dist_str = "dist_sum=unknown"

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
        dist_str,
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


# ---------------------------------------------------------------------------
# Catchment (sparse CSR) equilibrium kernels
# ---------------------------------------------------------------------------

@nb.njit(parallel=True, fastmath=True, cache=False)
def _catchment_demand_and_elasticity_jit(
    g,          # (N,) float64  beta*efforts - prices
    A,          # (2, N) float64 alpha_h * quality_j
    a0_scaled,  # float
    inv_mu,     # float
    w_L,        # (M,) float64
    w_H,        # (M,) float64
    indptr,     # (M+1,) int64
    indices,    # (NNZ,) int32
    catch_C,    # (NNZ,) float64
    N,          # int
):
    """Catchment equivalent of :func:`_demand_and_elasticity_jit`.

    Returns ``(D, E)`` where::

        D[j] = sum_i sum_h w_h(i) * s_ij^h
        E[j] = sum_i sum_h w_h(i) * s_ij^h * (1 - s_ij^h)

    Used inside the Bertrand-Nash iteration:  p_j - c_j = mu * D_j / E_j.
    Thread-local partial sums replace the (M, N) cell-contribution arrays.
    """
    M = len(indptr) - 1
    n_threads = nb.get_num_threads()
    partial_D = np.zeros((n_threads, N))
    partial_E = np.zeros((n_threads, N))

    for i in nb.prange(M):
        tid = nb.get_thread_id()
        start = indptr[i]
        end   = indptr[i + 1]
        k = int(end - start)
        if k == 0:
            continue

        ev = np.empty(k)

        for h in range(2):
            w_h = w_L[i] if h == 0 else w_H[i]
            if w_h == 0.0:
                continue

            vmax = a0_scaled
            for q in range(k):
                p = start + q
                j = indices[p]
                v = (A[h, j] + g[j] + catch_C[p]) * inv_mu
                if v > vmax:
                    vmax = v

            denom = np.exp(a0_scaled - vmax)
            for q in range(k):
                p = start + q
                j = indices[p]
                v = (A[h, j] + g[j] + catch_C[p]) * inv_mu
                ev[q] = np.exp(v - vmax)
                denom += ev[q]

            inv_d = 1.0 / denom
            for q in range(k):
                j = indices[start + q]
                s = ev[q] * inv_d
                partial_D[tid, j] += w_h * s
                partial_E[tid, j] += w_h * s * (1.0 - s)

    D = np.zeros(N)
    E = np.zeros(N)
    for t in range(n_threads):
        for j in range(N):
            D[j] += partial_D[t, j]
            E[j] += partial_E[t, j]
    return D, E


@nb.njit(parallel=True, fastmath=True, cache=False)
def _catchment_monopoly_grad_jit(
    g,          # (N,) float64  beta*efforts - prices
    margins,    # (N,) float64  prices - marginal_costs
    A,          # (2, N) float64
    a0_scaled,  # float
    inv_mu,     # float
    w_L,        # (M,) float64
    w_H,        # (M,) float64
    indptr,     # (M+1,) int64
    indices,    # (NNZ,) int32
    catch_C,    # (NNZ,) float64
    N,          # int
):
    """Catchment equivalent of :func:`_monopoly_demand_grad_jit`.

    Returns ``(D, G)`` where ``D[j]`` is demand and ``G[j]`` is the gradient
    contribution ``sum_i sum_h w_h(i) * inv_mu * s_ij^h * (m_h(i) - margin_j)``
    with ``m_h(i) = sum_j margin_j * s_ij^h`` the weighted average margin at
    cell *i* for type *h*.  Used inside the joint-monopoly gradient.
    """
    M = len(indptr) - 1
    n_threads = nb.get_num_threads()
    partial_D = np.zeros((n_threads, N))
    partial_G = np.zeros((n_threads, N))

    for i in nb.prange(M):
        tid = nb.get_thread_id()
        start = indptr[i]
        end   = indptr[i + 1]
        k = int(end - start)
        if k == 0:
            continue

        ev = np.empty(k)

        for h in range(2):
            w_h = w_L[i] if h == 0 else w_H[i]
            if w_h == 0.0:
                continue

            # Pass 1 — vmax
            vmax = a0_scaled
            for q in range(k):
                p = start + q
                j = indices[p]
                v = (A[h, j] + g[j] + catch_C[p]) * inv_mu
                if v > vmax:
                    vmax = v

            # Pass 2 — exp values, denom, and weighted average margin
            denom = np.exp(a0_scaled - vmax)
            for q in range(k):
                p = start + q
                j = indices[p]
                v = (A[h, j] + g[j] + catch_C[p]) * inv_mu
                ev[q] = np.exp(v - vmax)
                denom += ev[q]

            inv_d  = 1.0 / denom
            m_hi   = 0.0
            for q in range(k):
                j     = indices[start + q]
                m_hi += margins[j] * ev[q] * inv_d

            # Accumulate D and G
            for q in range(k):
                j = indices[start + q]
                s = ev[q] * inv_d
                partial_D[tid, j] += w_h * s
                partial_G[tid, j] += w_h * inv_mu * s * (m_hi - margins[j])

    D = np.zeros(N)
    G = np.zeros(N)
    for t in range(n_threads):
        for j in range(N):
            D[j] += partial_D[t, j]
            G[j] += partial_G[t, j]
    return D, G


def _ensure_catchment_eq(city: City, transport_cost: float) -> None:
    """Ensure City has catchment precompute arrays, building them if absent."""
    from hotelling.core.market import _ensure_catchment_precompute
    _ensure_catchment_precompute(city, transport_cost)


def _catchment_g(city: City, prices: np.ndarray, efforts: np.ndarray) -> np.ndarray:
    """Compute g = beta*efforts - prices for the catchment kernel."""
    return city.beta * np.ascontiguousarray(efforts, dtype=np.float64) - \
           np.ascontiguousarray(prices, dtype=np.float64)


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
    use_catchment = (city.catch_indptr is not None)
    if city.dist2_km2 is None and not use_catchment:
        raise ValueError(
            "bertrand_nash requires either city.dist2_km2 (dense matrix) or "
            "city.catch_indptr (sparse catchment).  Set dense_distances=True or "
            "catchment_minutes in load_berlin_city."
        )

    if cache_path is not None:
        cache_path = Path(cache_path)
        _sig = _param_signature(city, transport_cost)
        cached = _load_benchmark_cache(cache_path, f"nash_{_sig}")
        if cached is not None:
            return cached

    firms  = city.firms
    N      = len(firms)
    costs  = np.array([f.marginal_cost for f in firms], dtype=np.float64)
    kappa0 = np.array([f.kappa0        for f in firms], dtype=np.float64)
    quals  = np.array([f.quality       for f in firms], dtype=np.float64)
    beta   = city.beta
    prices  = costs.copy()
    efforts = np.zeros(N)
    converged = False

    if use_catchment:
        _ensure_catchment_eq(city, transport_cost)
        inv_mu    = 1.0 / float(city.mu)
        a0_scaled = float(city.a0) * inv_mu
        indptr    = city.catch_indptr.astype(np.int64,  copy=False)
        indices   = city.catch_indices.astype(np.int32, copy=False)
        catch_C   = np.ascontiguousarray(city.catch_C,     dtype=np.float64)
        A         = np.ascontiguousarray(city.A_quality,   dtype=np.float64)
        w_L       = np.ascontiguousarray(city.w_L,         dtype=np.float64)
        w_H       = np.ascontiguousarray(city.w_H,         dtype=np.float64)

        for _ in range(max_iter):
            g = _catchment_g(city, prices, efforts)
            D, E = _catchment_demand_and_elasticity_jit(
                g, A, a0_scaled, inv_mu, w_L, w_H, indptr, indices, catch_C, N
            )
            new_prices  = costs + city.mu * D / np.clip(E, 1e-12, None)
            new_efforts = beta * D / kappa0
            converged = (np.max(np.abs(new_prices - prices)) < tol
                         and np.max(np.abs(new_efforts - efforts)) < tol)
            prices, efforts = new_prices, new_efforts
            if converged:
                break
    else:
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
    effort_fixed: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find joint-monopoly (cartel) prices maximising total profit.

    Optimizes all N prices jointly using the analytic gradient of total profit
    from the spatial logit model, holding effort fixed.

    Parameters
    ----------
    effort_fixed : (N,) array or None
        Effort vector at which to evaluate demand while optimising prices.
        ``None`` (default) freezes effort at 0 — correct for price-only runs.
        For ``--with-effort`` runs, pass the Bertrand-Nash equilibrium efforts
        so the monopoly benchmark shares the Nash effort footing (otherwise the
        Calvano Δ compares effort-active prices to an effort-free benchmark).

    Returns
    -------
    prices : np.ndarray (N,)
    efforts : np.ndarray (N,)  — equal to ``effort_fixed`` (or zeros if None)
    """
    from scipy.optimize import minimize

    use_catchment = (city.catch_indptr is not None)
    if city.dist2_km2 is None and not use_catchment:
        raise ValueError(
            "joint_monopoly requires either city.dist2_km2 (dense matrix) or "
            "city.catch_indptr (sparse catchment).  Set dense_distances=True or "
            "catchment_minutes in load_berlin_city."
        )

    firms   = city.firms
    N       = len(firms)
    costs   = np.array([f.marginal_cost for f in firms], dtype=np.float64)
    quals   = np.array([f.quality       for f in firms], dtype=np.float64)

    if effort_fixed is None:
        efforts = np.zeros(N)
        _mono_prefix_base = "mono"
    else:
        efforts = np.ascontiguousarray(effort_fixed, dtype=np.float64)
        if efforts.shape != (N,):
            raise ValueError(f"effort_fixed shape {efforts.shape} != (N={N},)")
        _eh = hashlib.sha1(np.round(efforts, 6).tobytes()).hexdigest()[:6]
        _mono_prefix_base = f"monoEfix{_eh}"

    if cache_path is not None:
        cache_path = Path(cache_path)
        _sig = _param_signature(city, transport_cost)
        _prefix = f"{_mono_prefix_base}_{_sig}"
        cached = _load_benchmark_cache(cache_path, _prefix)
        if cached is not None:
            return cached

    if use_catchment:
        _ensure_catchment_eq(city, transport_cost)
        inv_mu    = 1.0 / float(city.mu)
        a0_scaled = float(city.a0) * inv_mu
        indptr    = city.catch_indptr.astype(np.int64,  copy=False)
        indices   = city.catch_indices.astype(np.int32, copy=False)
        catch_C   = np.ascontiguousarray(city.catch_C,   dtype=np.float64)
        A         = np.ascontiguousarray(city.A_quality, dtype=np.float64)
        w_L       = np.ascontiguousarray(city.w_L,       dtype=np.float64)
        w_H       = np.ascontiguousarray(city.w_H,       dtype=np.float64)

        def neg_obj_and_grad(p: np.ndarray) -> Tuple[float, np.ndarray]:
            margins = p - costs
            g = city.beta * efforts - p
            D, G = _catchment_monopoly_grad_jit(
                g, margins, A, a0_scaled, inv_mu, w_L, w_H,
                indptr, indices, catch_C, N
            )
            profit_val = float(np.sum(margins * D))
            grad = D + G
            return -profit_val, -grad
    else:
        def neg_obj_and_grad(p: np.ndarray) -> Tuple[float, np.ndarray]:
            D, G = _monopoly_demand_grad_jit(
                p, efforts, costs, city.dist2_km2, city.cell_pop, city.lambda_phi,
                city.pi_H, city.pi_H_lambda_phi,
                float(city.alpha[0]), float(city.alpha[1]),
                quals, float(city.beta), float(transport_cost), float(city.mu),
                float(city.a0), float(getattr(city, "transport_exponent", 1.0)))
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

    if cache_path is not None:
        _save_benchmark_cache(cache_path, _prefix, prices, efforts)

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
