"""Logit demand and market clearing.

Responsibility: compute logit market shares and firm profits given prices.

Public API: logit_demand, catchment_demand, profit, market_clearing,
            market_clearing_arrays, precompute_firm_arrays, FirmArrays,
            cell_choice_mass, catchment_cell_mass, cell_metrics

Key dependencies: numpy, numba, hotelling.core.city, hotelling.core.firm

Kernel dispatch
---------------
Two demand paths are supported:

Dense path (``city.dist2_km2`` is not None):
    ``_logit_demand_jit`` — the original (M,N)-allocated parallel numba kernel.
    Used for small grids (inner-ring Berlin) and validation.

Sparse catchment path (``city.catch_indptr`` is not None):
    ``_catchment_demand_jit`` — thread-local partial-sum reduction over the
    per-cell CSR catchment.  No (M,N) allocation; period-invariant terms
    (``catch_C``, ``A_quality``, ``w_H/L``) are precomputed by
    ``loader.populate_catchment_precompute``.

    Optionally dispatches to ``_catchment_demand_expw_jit`` when
    ``city.precompute_expweights=True``: only N exp() calls per period
    (vs one per catchment entry in the stable kernel).

Both paths produce identical demand vectors up to the catchment truncation
error (which is controlled by ``catchment_minutes / k_min / k_max``).

References:
    Calvano et al. (2020 AER) §II.A;
    Anderson, de Palma, Thisse (1992) - spatial logit extension.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple

import numba as nb
import numpy as np

from hotelling.core.city import City


# ---------------------------------------------------------------------------
# Precomputed per-firm arrays (hot-path cache)
# ---------------------------------------------------------------------------

class FirmArrays(NamedTuple):
    """Precomputed contiguous float64 arrays for the market-clearing hot path.

    All arrays have shape (N,) and are pre-cast to float64 C-contiguous layout
    so that ``market_clearing_arrays`` can call ``logit_demand`` (numba JIT)
    without per-call allocation.

    Fields
    ------
    qualities      : vertical quality intercepts per store.
    marginal_costs : per-unit marginal costs.
    kappa0         : quadratic effort cost coefficients.
    sizes          : store floor areas in m² (used with the rent×size channel).
    rents          : per-m² rent costs.
    fixed_costs    : per-period lump-sum fixed costs (ADR-022).
    """

    qualities:      np.ndarray
    marginal_costs: np.ndarray
    kappa0:         np.ndarray
    sizes:          np.ndarray
    rents:          np.ndarray
    fixed_costs:    np.ndarray


def precompute_firm_arrays(firms: list) -> FirmArrays:
    """Build a :class:`FirmArrays` from a list of :class:`~hotelling.core.firm.Firm`.

    Called once per City by :func:`market_clearing` (via the module-level cache)
    and once per :class:`~hotelling.env.market_env.HotellingMarketEnv` instance
    (at ``__init__`` time).  Subsequent hot-path calls to
    :func:`market_clearing_arrays` pass the precomputed struct directly.

    Parameters
    ----------
    firms : list of Firm objects in canonical column order.

    Returns
    -------
    FirmArrays  namedtuple of six (N,) float64 arrays.
    """
    return FirmArrays(
        qualities=np.ascontiguousarray(
            [f.quality for f in firms], dtype=np.float64
        ),
        marginal_costs=np.ascontiguousarray(
            [f.marginal_cost for f in firms], dtype=np.float64
        ),
        kappa0=np.ascontiguousarray(
            [f.kappa0 for f in firms], dtype=np.float64
        ),
        sizes=np.ascontiguousarray(
            [f.size for f in firms], dtype=np.float64
        ),
        rents=np.ascontiguousarray(
            [f.rent for f in firms], dtype=np.float64
        ),
        fixed_costs=np.ascontiguousarray(
            [f.fixed_cost for f in firms], dtype=np.float64
        ),
    )


import weakref

# Module-level cache: id(city) → (weakref_to_city, FirmArrays).
#
# We store a weakref alongside the arrays so we can detect the case where
# Python's allocator reuses an old City's memory address for a new City with
# different Firm attributes.  On lookup:
#   - cache miss        → compute and store.
#   - ref() is city     → same live object, return cached FirmArrays.
#   - ref() is None     → original City was GC'd; new City at same address,
#                         compute fresh FirmArrays and overwrite entry.
#
# City is a mutable (non-frozen) dataclass and therefore unhashable, which
# precludes WeakKeyDictionary on Python ≥ 3.14.  This approach is both
# correct and compatible across all Python versions.
_FIRM_ARRAY_CACHE: dict[
    int, "tuple[weakref.ref[City], FirmArrays]"
] = {}


def _get_firm_arrays(city: City) -> FirmArrays:
    """Return cached FirmArrays for ``city``, computing on first call."""
    key = id(city)
    entry = _FIRM_ARRAY_CACHE.get(key)
    if entry is not None:
        city_ref, fa = entry
        if city_ref() is city:
            return fa
    # Miss or stale (original City was GC'd and address reused)
    fa = precompute_firm_arrays(city.firms)
    _FIRM_ARRAY_CACHE[key] = (weakref.ref(city), fa)
    return fa


# ---------------------------------------------------------------------------
# Catchment (sparse CSR) demand kernels
# ---------------------------------------------------------------------------

# cache=False: this parallel kernel uses nb.get_thread_id()/thread-local
# reduction, which Numba cannot disk-cache (it would warn and recompile
# regardless). Explicit cache=False silences the misleading warning; the
# one-time ~1-2s recompile per process is unchanged. See market.py kernels.
@nb.njit(parallel=True, fastmath=True, cache=False)
def _catchment_demand_jit(
    g,          # (N,) float64  beta*efforts - prices  [per-period]
    A,          # (2, N) float64  alpha_h * quality_j  [invariant]
    a0_scaled,  # float: a0 / mu                       [invariant]
    inv_mu,     # float: 1 / mu                        [invariant]
    w_L,        # (M,) float64  low-income cell weights [invariant]
    w_H,        # (M,) float64  high-income cell weights[invariant]
    indptr,     # (M+1,) int64  CSR row pointers        [invariant]
    indices,    # (NNZ,) int32  store column indices    [invariant]
    catch_C,    # (NNZ,) float64  -tc*tt^exp            [invariant]
    N,          # int: number of stores
):
    """Numerically-stable catchment demand kernel — thread-local reduction.

    Uses a two-pass log-sum-exp (per income type, per cell).  The only
    per-period inputs are ``g = beta*efforts - prices``; all other arrays
    are period-invariant and precomputed by
    :func:`~hotelling.spatial.loader.populate_catchment_precompute`.

    The thread-local ``partial[tid, j]`` accumulator replaces the dense
    ``(M, N)`` matrix used by ``_logit_demand_jit``, cutting peak memory
    from O(M*N) to O(n_threads*N).

    Parameters
    ----------
    g        : (N,) per-period combined price-effort vector.
    A        : (2, N) quality matrix, A[h, j] = alpha_h * quality_j.
    a0_scaled: outside-option utility, scaled by inv_mu.
    inv_mu   : reciprocal of the logit scale parameter.
    w_L, w_H : (M,) cell-level consumer mass by income type.
    indptr   : (M+1,) CSR row-pointer array.
    indices  : (NNZ,) store column indices.
    catch_C  : (NNZ,) per-entry transport disutility (already negated and
               scaled by transport_cost and transport_exponent).
    N        : number of stores.

    Returns
    -------
    (N,) float64 demand vector.
    """
    M = len(indptr) - 1
    n_threads = nb.get_num_threads()
    partial = np.zeros((n_threads, N))

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

            # Pass 1 — find vmax for log-sum-exp stabilisation
            vmax = a0_scaled
            for q in range(k):
                p = start + q
                j = indices[p]
                v = (A[h, j] + g[j] + catch_C[p]) * inv_mu
                if v > vmax:
                    vmax = v

            # Pass 2 — compute exp values and denominator
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
                partial[tid, j] += w_h * ev[q] * inv_d

    # Sequential reduction across threads — O(n_threads * N), negligible
    result = np.zeros(N)
    for t in range(n_threads):
        for j in range(N):
            result[j] += partial[t, j]
    return result


# cache=False: this parallel kernel uses nb.get_thread_id()/thread-local
# reduction, which Numba cannot disk-cache (it would warn and recompile
# regardless). Explicit cache=False silences the misleading warning; the
# one-time ~1-2s recompile per process is unchanged. See market.py kernels.
@nb.njit(parallel=True, fastmath=True, cache=False)
def _catchment_demand_expw_jit(
    g,       # (N,) float64  beta*efforts - prices  [per-period]
    Kexp_L,  # (NNZ,) precomputed exp((A[0,idx]+C)*inv_mu)
    Kexp_H,  # (NNZ,) precomputed exp((A[1,idx]+C)*inv_mu)
    a0_exp,  # float: exp(a0_scaled) — precomputed constant
    inv_mu,  # float
    w_L,     # (M,) float64
    w_H,     # (M,) float64
    indptr,  # (M+1,) int64
    indices, # (NNZ,) int32
    N,       # int: number of stores
):
    """Max-performance catchment demand kernel — N exp() calls per period.

    Relies on precomputed Kexp arrays built by
    :func:`~hotelling.spatial.loader.populate_catchment_precompute`.
    The cell loop is pure multiply-accumulate (no exp, no log-sum-exp):
    ``num_p = Kexp_h[p] * w_g[j]``, ``denom = a0_exp + sum(num_p)``.

    Requires ``city.precompute_expweights = True`` (set when max exponent
    < 700 guard passes at build time).  Falls back to :func:`_catchment_demand_jit`
    when the guard fails.

    Parameters
    ----------
    g       : (N,) per-period vector ``beta*efforts - prices``.
    Kexp_L  : (NNZ,) exp-weight array for low-income type.
    Kexp_H  : (NNZ,) exp-weight array for high-income type.
    a0_exp  : ``exp(a0 / mu)`` — constant outside-option factor.
    inv_mu  : ``1 / mu``.
    w_L, w_H: (M,) cell consumer-mass weights.
    indptr  : (M+1,) CSR row-pointer array.
    indices : (NNZ,) store column indices.
    N       : number of stores.

    Returns
    -------
    (N,) float64 demand vector.
    """
    M = len(indptr) - 1
    n_threads = nb.get_num_threads()
    partial = np.zeros((n_threads, N))

    # Per-period: N exp() calls, one per store — no exp inside the cell loop
    w_g = np.empty(N)
    for j in range(N):
        w_g[j] = np.exp(g[j] * inv_mu)

    for i in nb.prange(M):
        tid = nb.get_thread_id()
        start = indptr[i]
        end   = indptr[i + 1]
        k = int(end - start)
        if k == 0:
            continue

        for h in range(2):
            w_h = w_L[i] if h == 0 else w_H[i]
            if w_h == 0.0:
                continue

            Kexp_h = Kexp_L if h == 0 else Kexp_H

            # Pass 1: accumulate denominator (no exp in this loop)
            denom = a0_exp
            for q in range(k):
                p = start + q
                j = indices[p]
                denom += Kexp_h[p] * w_g[j]

            inv_d = 1.0 / denom

            # Pass 2: scatter contributions
            for q in range(k):
                p = start + q
                j = indices[p]
                partial[tid, j] += w_h * (Kexp_h[p] * w_g[j]) * inv_d

    result = np.zeros(N)
    for t in range(n_threads):
        for j in range(N):
            result[j] += partial[t, j]
    return result


@nb.njit(parallel=True, fastmath=True, cache=True)
def _catchment_cell_mass_jit(
    g,       # (N,) float64  [per-period]
    A,       # (2, N) float64 [invariant]
    a0_scaled,  # float
    inv_mu,     # float
    w_L,     # (M,) float64
    w_H,     # (M,) float64
    indptr,  # (M+1,) int64
    indices, # (NNZ,) int32
    catch_C, # (NNZ,) float64
    N,       # int: number of stores
):
    """Catchment variant of :func:`_cell_choice_mass_jit` — per-cell allocations.

    Returns a dense (M, N) inside-mass matrix and (M,) outside-mass vector,
    filling only the catchment entries of each row.  Non-catchment entries
    are exactly zero.  Safe to use with ``prange(M)`` because each cell *i*
    writes exclusively to row *i* of ``inside_mass``.

    Designed for choropleth visualisation on small-to-medium grids.  For the
    full Berlin grid consider the CSR-valued variant (not yet implemented).
    """
    M = len(indptr) - 1
    inside_mass  = np.zeros((M, N))
    outside_mass = np.zeros(M)

    for i in nb.prange(M):
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

            outside_ev = np.exp(a0_scaled - vmax)
            denom = outside_ev
            for q in range(k):
                p = start + q
                j = indices[p]
                v = (A[h, j] + g[j] + catch_C[p]) * inv_mu
                ev[q] = np.exp(v - vmax)
                denom += ev[q]

            inv_d = 1.0 / denom
            outside_mass[i] += w_h * outside_ev * inv_d
            for q in range(k):
                j = indices[start + q]
                inside_mass[i, j] += w_h * ev[q] * inv_d

    return inside_mass, outside_mass


# ---------------------------------------------------------------------------
# Python dispatch helpers for the catchment path
# ---------------------------------------------------------------------------

def _ensure_catchment_precompute(city: "City", transport_cost: float) -> None:
    """Lazily populate City's catchment precompute arrays if absent or stale.

    Called by ``catchment_demand`` and ``catchment_cell_mass`` before handing
    off to the numba kernels.  In a normal simulation flow
    ``loader.populate_catchment_precompute`` has already run and this is a
    no-op (the tc check is a single float comparison).
    """
    if (
        city.catch_C is None
        or city.A_quality is None
        or city.catch_C_transport_cost != transport_cost
    ):
        from hotelling.spatial.loader import populate_catchment_precompute
        populate_catchment_precompute(
            city,
            transport_cost=transport_cost,
            precompute_expweights=getattr(city, "precompute_expweights", False),
            low_precision_storage=getattr(city, "low_precision_storage", False),
        )


def catchment_demand(
    city: "City",
    prices: np.ndarray,
    efforts: np.ndarray,
    transport_cost: float | None = None,
) -> np.ndarray:
    """Compute logit market shares using the sparse catchment kernel.

    Dispatches to the expweights fast path when ``city.precompute_expweights``
    is True, otherwise falls back to the numerically-stable two-pass kernel.

    Parameters
    ----------
    city           : City with ``catch_indptr`` / ``catch_indices`` / ``catch_tt``
                     populated (sparse catchment path).
    prices         : (N,) float64 posted prices.
    efforts        : (N,) float64 effort levels.
    transport_cost : if None, uses ``city.catch_C_transport_cost`` (the value
                     baked in at precompute time).  Pass an explicit value to
                     force a recompute with a different tc.

    Returns
    -------
    (N,) float64 demand vector.
    """
    assert city.catch_indptr is not None, (
        "catchment_demand: city.catch_indptr is None — build CSR first."
    )
    tc = float(transport_cost) if transport_cost is not None else (
        city.catch_C_transport_cost if city.catch_C_transport_cost is not None else 0.0
    )
    _ensure_catchment_precompute(city, tc)

    prices  = np.ascontiguousarray(prices,  dtype=np.float64)
    efforts = np.ascontiguousarray(efforts, dtype=np.float64)
    N = len(city.firms)
    g = city.beta * efforts - prices          # (N,) per-period vector

    inv_mu    = 1.0 / float(city.mu)
    a0_scaled = float(city.a0) * inv_mu
    indptr    = city.catch_indptr.astype(np.int64,  copy=False)
    indices   = city.catch_indices.astype(np.int32, copy=False)
    catch_C   = np.ascontiguousarray(city.catch_C, dtype=np.float64)
    A         = np.ascontiguousarray(city.A_quality, dtype=np.float64)
    w_L       = np.ascontiguousarray(city.w_L, dtype=np.float64)
    w_H       = np.ascontiguousarray(city.w_H, dtype=np.float64)

    if city.precompute_expweights:
        a0_exp  = float(np.exp(a0_scaled))
        Kexp_L  = np.ascontiguousarray(city.catch_Kexp_L, dtype=np.float64)
        Kexp_H  = np.ascontiguousarray(city.catch_Kexp_H, dtype=np.float64)
        return _catchment_demand_expw_jit(
            g, Kexp_L, Kexp_H, a0_exp, inv_mu, w_L, w_H, indptr, indices, N
        )
    else:
        return _catchment_demand_jit(
            g, A, a0_scaled, inv_mu, w_L, w_H, indptr, indices, catch_C, N
        )


def catchment_cell_mass(
    city: "City",
    prices: np.ndarray,
    efforts: np.ndarray,
    transport_cost: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Catchment-path equivalent of :func:`cell_choice_mass`.

    Returns the dense ``(M, N)`` inside-mass matrix (non-catchment entries
    are zero) and the ``(M,)`` outside-mass vector.  Suitable for choropleth
    visualisation on the sparse path.

    Parameters
    ----------
    city           : City with sparse catchment populated.
    prices, efforts: (N,) float64.
    transport_cost : see :func:`catchment_demand`.

    Returns
    -------
    inside_mass : (M, N) float64
    outside_mass: (M,)  float64
    """
    assert city.catch_indptr is not None
    tc = float(transport_cost) if transport_cost is not None else (
        city.catch_C_transport_cost if city.catch_C_transport_cost is not None else 0.0
    )
    _ensure_catchment_precompute(city, tc)

    prices  = np.ascontiguousarray(prices,  dtype=np.float64)
    efforts = np.ascontiguousarray(efforts, dtype=np.float64)
    N       = len(city.firms)
    g       = city.beta * efforts - prices

    inv_mu    = 1.0 / float(city.mu)
    a0_scaled = float(city.a0) * inv_mu
    indptr    = city.catch_indptr.astype(np.int64,  copy=False)
    indices   = city.catch_indices.astype(np.int32, copy=False)
    catch_C   = np.ascontiguousarray(city.catch_C, dtype=np.float64)
    A         = np.ascontiguousarray(city.A_quality, dtype=np.float64)
    w_L       = np.ascontiguousarray(city.w_L, dtype=np.float64)
    w_H       = np.ascontiguousarray(city.w_H, dtype=np.float64)

    return _catchment_cell_mass_jit(
        g, A, a0_scaled, inv_mu, w_L, w_H, indptr, indices, catch_C, N
    )


# ---------------------------------------------------------------------------
# Dense (original) kernels
# ---------------------------------------------------------------------------

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
    prices  = np.ascontiguousarray(prices,  dtype=np.float64)
    efforts = np.ascontiguousarray(efforts, dtype=np.float64)
    transport_exponent = getattr(city, "transport_exponent", 1.0)

    # Dispatch to catchment variant on the sparse path
    if city.catch_indptr is not None:
        inside, outside = catchment_cell_mass(
            city=city, prices=prices, efforts=efforts,
            transport_cost=transport_cost,
        )
    else:
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
        if city.dist2_km2 is None:
            raise ValueError(
                "cell_metrics 'consumer_surplus' requires the dense distance matrix "
                "(city.dist2_km2).  On the sparse catchment path this metric is "
                "not yet supported."
            )
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
    fixed_cost: np.ndarray | float = 0.0,
) -> float | np.ndarray:
    """Compute firm profit per period.

    profit = (price - mc) * demand - 0.5 * kappa0 * effort² - rent * size - fixed_cost

    Parameters
    ----------
    price : posted price(s)
    demand : realised demand(s)
    marginal_cost : per-unit cost(s)
    kappa0 : quadratic effort cost coefficient (scalar or per-firm array)
    effort : effort level(s)
    size : store floor area in m² (used with the rent*size channel)
    rent : per-m² rent cost(s); default 0.0
    fixed_cost : per-period lump-sum fixed operating cost(s), size-independent;
        default 0.0.  As an additive constant this does NOT enter the price FOC —
        Bertrand-Nash / joint-monopoly prices and Calvano Δ are invariant to it.
        It shifts absolute profit levels and is relevant only at the entry/exit
        margin (Phase 1+).  See ADR-022.

    Returns
    -------
    float or ndarray — per-firm profit(s)
    """
    return (
        (price - marginal_cost) * demand
        - 0.5 * kappa0 * effort**2
        - rent * size
        - fixed_cost
    )


def market_clearing_arrays(
    prices: np.ndarray,
    efforts: np.ndarray,
    city: City,
    transport_cost: float,
    firm_arrays: FirmArrays,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute equilibrium demands and profits using precomputed firm arrays.

    This is the hot-path entry point for :class:`~hotelling.env.market_env.HotellingMarketEnv`
    and :class:`~hotelling.simulation.engine.BatchSimulationEngine`: the caller
    pre-builds ``firm_arrays`` once (in ``__init__``) and passes it in every
    period, avoiding all per-call Python list comprehensions.

    Parameters
    ----------
    prices : (N,) float64 — posted prices this period.
    efforts : (N,) float64 — service efforts this period.
    city : City — spatial container (provides cell_pop, dist2_km2, etc.).
    transport_cost : float — transport disutility coefficient.
    firm_arrays : FirmArrays — precomputed per-firm attribute arrays.
        Build once via :func:`precompute_firm_arrays` or obtain from the
        module-level cache via :func:`market_clearing`.

    Returns
    -------
    demands : (N,) float64 — logit market shares.
    profits : (N,) float64 — per-firm profits this period.

    See Also
    --------
    market_clearing : thin wrapper that handles caching automatically.
    precompute_firm_arrays : build the FirmArrays struct.
    """
    prices  = np.ascontiguousarray(prices,  dtype=np.float64)
    efforts = np.ascontiguousarray(efforts, dtype=np.float64)

    # Dispatch: sparse catchment path when CSR is populated, else dense.
    if city.catch_indptr is not None:
        demands = catchment_demand(city=city, prices=prices, efforts=efforts,
                                   transport_cost=transport_cost)
    else:
        demands = logit_demand(
            prices=prices,
            efforts=efforts,
            dist2_km2=city.dist2_km2,
            cell_pop=city.cell_pop,
            lambda_phi=city.lambda_phi,
            pi_H=city.pi_H,
            pi_H_lambda_phi=city.pi_H_lambda_phi,
            alpha=city.alpha,
            quality=firm_arrays.qualities,
            beta=city.beta,
            transport_cost=transport_cost,
            mu=city.mu,
            a0=city.a0,
            transport_exponent=getattr(city, "transport_exponent", 1.0),
        )

    profits = profit(
        price=prices,
        demand=demands,
        marginal_cost=firm_arrays.marginal_costs,
        kappa0=firm_arrays.kappa0,
        effort=efforts,
        size=firm_arrays.sizes,
        rent=firm_arrays.rents,
        fixed_cost=firm_arrays.fixed_costs,
    )

    return demands, profits


def market_clearing(
    prices: np.ndarray,
    efforts: np.ndarray,
    city: City,
    transport_cost: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute equilibrium demands and profits for all firms.

    Thin wrapper around :func:`market_clearing_arrays`.  Per-firm attribute
    arrays are built on the first call for each *city* object and cached for
    all subsequent calls (keyed by ``(id(city), len(city.firms))``).

    Parameters
    ----------
    prices : (N,) float64 — posted prices this period.
    efforts : (N,) float64 — service efforts this period.
    city : City — spatial container.
    transport_cost : float — transport disutility coefficient.

    Returns
    -------
    demands : (N,) float64 — logit market shares.
    profits : (N,) float64 — per-firm profits this period.
    """
    return market_clearing_arrays(
        prices, efforts, city, transport_cost, _get_firm_arrays(city)
    )
