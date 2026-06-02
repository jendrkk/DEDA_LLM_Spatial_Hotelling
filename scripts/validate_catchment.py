"""Catchment-kernel validation harness.

Loads a *small* synthetic city that has **both** a dense distance matrix and
a sparse CSR catchment, then checks that the catchment demand kernel matches
the dense reference within tolerance for 50 random price/effort vectors.

Additionally verifies that Bertrand-Nash and joint-monopoly prices agree
between the two paths within 1e-3.

Usage
-----
    conda activate py314
    python scripts/validate_catchment.py

Optional flags
--------------
    --tc FLOAT        transport_cost (default 0.01)
    --n-samples INT   number of random price/effort draws (default 50)
    --expweights      also validate the expweights fast path
    --seed INT        numpy random seed (default 42)

Exit codes
----------
0 — all checks passed
1 — at least one tolerance was exceeded
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic-city helpers (no GIS data required)
# ---------------------------------------------------------------------------

def _make_tiny_city(
    n_stores: int = 6,
    n_cells: int = 20,
    transport_cost: float = 0.01,
    catchment_minutes: float = 1000.0,
    k_min: int = 6,
    k_max: int = 6,
    seed: int = 42,
    precompute_expweights: bool = False,
) -> "City":
    """Build a self-consistent tiny City with both dense and sparse representations.

    Default catchment parameters (radius=1000 min, k_min=k_max=n_stores) ensure
    **every cell includes ALL stores**, making catchment demand numerically equal
    to the dense demand (up to floating-point rounding).  The tolerance checks
    (1e-4 / 1e-3) therefore measure kernel arithmetic accuracy, not truncation.

    Pass ``catchment_minutes=30`` to see the effect of deliberate truncation
    (in which case the "error" reflects the intentional subset selection and
    will be large for small cities where every store matters).
    """
    from hotelling.core.city import City
    from hotelling.core.firm import Firm
    from hotelling.spatial.loader import (
        build_catchment,
        populate_catchment_precompute,
    )

    rng = np.random.default_rng(seed)

    # Firms
    firms = [
        Firm(
            id=str(j),
            location=(float(j) * 100.0, 0.0),
            marginal_cost=0.0,
            quality=rng.uniform(0.0, 1.5),
            kappa0=1.0,
            size=600.0,
            rent=0.0,
            fixed_cost=0.0,
        )
        for j in range(n_stores)
    ]

    # Travel-time matrix — (n_cells, n_stores) minutes in [1, 60]
    tt_dense = rng.uniform(1.0, 60.0, size=(n_cells, n_stores))

    # Cell population and type arrays
    cell_pop          = rng.uniform(50.0, 500.0, size=n_cells)
    lambda_phi        = rng.uniform(0.0, 50.0, size=n_cells)
    pi_H              = rng.uniform(0.2, 0.8, size=n_cells)
    pi_H_lambda_phi   = pi_H.copy()

    # Build CSR from the same travel-time data
    import pandas as pd
    rows, cols = np.meshgrid(np.arange(n_cells), np.arange(n_stores), indexing="ij")
    cell_ids  = [f"cell_{i}" for i in range(n_cells)]
    store_ids = [str(j) for j in range(n_stores)]
    tt_long = pd.DataFrame({
        "from_id":     [cell_ids[i]  for i in rows.ravel()],
        "to_id":       [store_ids[j] for j in cols.ravel()],
        "travel_time": tt_dense.ravel(),
    })

    indptr, indices, catch_tt = build_catchment(
        tt_df=tt_long,
        cell_ids=cell_ids,
        store_ids=store_ids,
        transport_cost=transport_cost,
        transport_exponent=1.0,
        catchment_minutes=catchment_minutes,
        k_min=min(k_min, n_stores),
        k_max=min(k_max, n_stores),
    )

    city = City(
        boundary=(0.0, 0.0, 1000.0, 1000.0),
        population_grid=None,
        firms=firms,
        dist2_km2=tt_dense,          # keep dense for reference comparison
        cell_pop=cell_pop,
        lambda_phi=lambda_phi,
        pi_H=pi_H,
        pi_H_lambda_phi=pi_H_lambda_phi,
        alpha=np.array([0.5, 1.5]),
        beta=0.001,
        mu=0.25,
        a0=-1.0,
        transport_exponent=1.0,
        catch_indptr=indptr,
        catch_indices=indices,
        catch_tt=catch_tt,
    )

    populate_catchment_precompute(
        city,
        transport_cost=transport_cost,
        precompute_expweights=precompute_expweights,
    )

    return city


# ---------------------------------------------------------------------------
# Demand comparison helpers
# ---------------------------------------------------------------------------

def _dense_demand(city, prices, efforts, transport_cost):
    from hotelling.core.market import logit_demand
    return logit_demand(
        prices=prices,
        efforts=efforts,
        dist2_km2=city.dist2_km2,
        cell_pop=city.cell_pop,
        lambda_phi=city.lambda_phi,
        pi_H=city.pi_H,
        pi_H_lambda_phi=city.pi_H_lambda_phi,
        alpha=city.alpha,
        quality=np.array([f.quality for f in city.firms], dtype=np.float64),
        beta=city.beta,
        transport_cost=transport_cost,
        mu=city.mu,
        a0=city.a0,
        transport_exponent=getattr(city, "transport_exponent", 1.0),
    )


def _catchment_demand_stable(city, prices, efforts, transport_cost):
    """Force the stable kernel even if expweights are precomputed."""
    from hotelling.core.market import _catchment_demand_jit
    import numpy as np
    prices  = np.ascontiguousarray(prices,  dtype=np.float64)
    efforts = np.ascontiguousarray(efforts, dtype=np.float64)
    N = len(city.firms)
    g = city.beta * efforts - prices
    inv_mu    = 1.0 / float(city.mu)
    a0_scaled = float(city.a0) * inv_mu
    return _catchment_demand_jit(
        g,
        np.ascontiguousarray(city.A_quality, dtype=np.float64),
        a0_scaled, inv_mu,
        np.ascontiguousarray(city.w_L, dtype=np.float64),
        np.ascontiguousarray(city.w_H, dtype=np.float64),
        city.catch_indptr.astype(np.int64,  copy=False),
        city.catch_indices.astype(np.int32, copy=False),
        np.ascontiguousarray(city.catch_C, dtype=np.float64),
        N,
    )


def _catchment_demand_expw(city, prices, efforts):
    """Call the expweights kernel (requires city.precompute_expweights=True)."""
    from hotelling.core.market import _catchment_demand_expw_jit
    import numpy as np
    prices  = np.ascontiguousarray(prices,  dtype=np.float64)
    efforts = np.ascontiguousarray(efforts, dtype=np.float64)
    N = len(city.firms)
    g = city.beta * efforts - prices
    inv_mu    = 1.0 / float(city.mu)
    a0_scaled = float(city.a0) * inv_mu
    return _catchment_demand_expw_jit(
        g,
        np.ascontiguousarray(city.catch_Kexp_L, dtype=np.float64),
        np.ascontiguousarray(city.catch_Kexp_H, dtype=np.float64),
        float(np.exp(a0_scaled)),
        inv_mu,
        np.ascontiguousarray(city.w_L, dtype=np.float64),
        np.ascontiguousarray(city.w_H, dtype=np.float64),
        city.catch_indptr.astype(np.int64,  copy=False),
        city.catch_indices.astype(np.int32, copy=False),
        N,
    )


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

def validate(
    n_samples: int = 50,
    transport_cost: float = 0.01,
    with_expweights: bool = False,
    seed: int = 42,
    tol_stable: float = 1e-4,
    tol_expw:   float = 1e-3,
    tol_bench:  float = 1e-3,
) -> bool:
    """Run all validation checks.  Returns True if all pass."""
    all_ok = True

    print("=" * 70)
    print("Catchment kernel validation harness")
    print("=" * 70)

    # ── Build city ──────────────────────────────────────────────────────────
    city = _make_tiny_city(
        transport_cost=transport_cost,
        precompute_expweights=with_expweights,
        seed=seed,
    )
    N = len(city.firms)
    nnz = int(city.catch_indptr[-1])
    sizes = np.diff(city.catch_indptr)
    print(f"\nCity: {N} stores, {len(city.cell_pop)} cells")
    print(f"CSR: NNZ={nnz}, mean_catchment={sizes.mean():.1f}, "
          f"median={float(np.median(sizes)):.1f}, "
          f"min={sizes.min()}, max={sizes.max()}")
    print(f"expweights precomputed: {city.precompute_expweights}")

    # ── Warm up numba JIT ────────────────────────────────────────────────────
    _p0 = np.ones(N, dtype=np.float64)
    _e0 = np.zeros(N, dtype=np.float64)
    _dense_demand(city, _p0, _e0, transport_cost)
    _catchment_demand_stable(city, _p0, _e0, transport_cost)
    if with_expweights and city.precompute_expweights:
        _catchment_demand_expw(city, _p0, _e0)

    # ── Random price / effort vectors ────────────────────────────────────────
    rng = np.random.default_rng(seed + 100)
    price_range  = (0.5, 5.0)
    effort_range = (0.0, 10.0)

    err_stable = []
    err_expw   = []

    t_dense   = 0.0
    t_stable  = 0.0
    t_expw    = 0.0

    for _ in range(n_samples):
        prices  = rng.uniform(*price_range,  size=N)
        efforts = rng.uniform(*effort_range, size=N)

        t0 = time.perf_counter()
        d_dense  = _dense_demand(city, prices, efforts, transport_cost)
        t_dense += time.perf_counter() - t0

        t0 = time.perf_counter()
        d_stable = _catchment_demand_stable(city, prices, efforts, transport_cost)
        t_stable += time.perf_counter() - t0

        norm_ref = float(np.linalg.norm(d_dense))
        if norm_ref > 0:
            err_stable.append(float(np.linalg.norm(d_dense - d_stable)) / norm_ref)

        if with_expweights and city.precompute_expweights:
            t0 = time.perf_counter()
            d_expw = _catchment_demand_expw(city, prices, efforts)
            t_expw += time.perf_counter() - t0
            if norm_ref > 0:
                err_expw.append(float(np.linalg.norm(d_dense - d_expw)) / norm_ref)

    print(f"\n── Demand accuracy over {n_samples} random draws ──")
    max_stable = max(err_stable) if err_stable else 0.0
    print(f"  Stable path   max rel-err = {max_stable:.2e}  (tol={tol_stable:.0e})")
    if max_stable > tol_stable:
        print(f"  FAIL: stable path exceeds tolerance!")
        all_ok = False
    else:
        print(f"  PASS")

    if err_expw:
        max_expw = max(err_expw)
        print(f"  Expweights    max rel-err = {max_expw:.2e}  (tol={tol_expw:.0e})")
        if max_expw > tol_expw:
            print(f"  FAIL: expweights path exceeds tolerance!")
            all_ok = False
        else:
            print(f"  PASS")

    print(f"\n── Per-call timing ({n_samples} calls) ──")
    print(f"  Dense:      {1e3 * t_dense  / n_samples:.3f} ms/call")
    print(f"  Stable CSR: {1e3 * t_stable / n_samples:.3f} ms/call  "
          f"(ratio dense/csr = {t_dense / max(t_stable, 1e-12):.1f}×)")
    if t_expw > 0:
        print(f"  Expweights: {1e3 * t_expw / n_samples:.3f} ms/call  "
              f"(ratio dense/expw = {t_dense / max(t_expw, 1e-12):.1f}×)")

    # ── Equilibrium benchmarks ────────────────────────────────────────────────
    print(f"\n── Equilibrium benchmark agreement (tol={tol_bench:.0e}) ──")
    from hotelling.core.equilibrium import bertrand_nash, joint_monopoly

    # Dense path (city.dist2_km2 is not None)
    city_dense_only = city
    p_nash_dense, e_nash_dense = bertrand_nash(
        city_dense_only, transport_cost=transport_cost
    )

    # Catchment path: temporarily mask out dist2_km2
    import copy
    city_sparse = copy.copy(city)
    city_sparse.dist2_km2 = None
    p_nash_catch, e_nash_catch = bertrand_nash(
        city_sparse, transport_cost=transport_cost
    )

    nash_err = float(np.max(np.abs(p_nash_dense - p_nash_catch)))
    print(f"  Bertrand-Nash price max-abs-err = {nash_err:.2e}  (tol={tol_bench:.0e})")
    if nash_err > tol_bench:
        print("  FAIL: Bertrand-Nash prices disagree between dense and catchment!")
        all_ok = False
    else:
        print("  PASS")

    p_mono_dense, _ = joint_monopoly(
        city_dense_only, transport_cost=transport_cost
    )
    city_sparse2 = copy.copy(city)
    city_sparse2.dist2_km2 = None
    p_mono_catch, _ = joint_monopoly(
        city_sparse2, transport_cost=transport_cost
    )

    mono_err = float(np.max(np.abs(p_mono_dense - p_mono_catch)))
    print(f"  Joint-monopoly price max-abs-err = {mono_err:.2e}  (tol={tol_bench:.0e})")
    if mono_err > tol_bench:
        print("  FAIL: joint-monopoly prices disagree between dense and catchment!")
        all_ok = False
    else:
        print("  PASS")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED") + "\n")
    return all_ok


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tc",         type=float, default=0.01,  dest="transport_cost")
    p.add_argument("--n-samples",  type=int,   default=50)
    p.add_argument("--expweights", action="store_true")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ok = validate(
        n_samples=args.n_samples,
        transport_cost=args.transport_cost,
        with_expweights=args.expweights,
        seed=args.seed,
    )
    sys.exit(0 if ok else 1)
