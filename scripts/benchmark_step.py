#!/usr/bin/env python
"""Benchmark dense vs sparse catchment demand kernels (inner ring).

Correctness: 50 random (price, effort) draws; max relative L2 error vs dense.
Timing: 300 hot-path calls per kernel after JIT warmup.

Usage:
    python scripts/benchmark_step.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_baseline import _ENV_YAMLS, load_config  # noqa: E402
from hotelling.core.market import (  # noqa: E402
    catchment_demand,
    logit_demand,
    precompute_firm_arrays,
)
from hotelling.spatial.loader import load_berlin_city  # noqa: E402


def _loader_kwargs(env_cfg: dict, **overrides) -> dict:
    """Mirror run_single_session load_berlin_city kwargs; apply overrides last."""
    _catchment_minutes_raw = overrides.pop(
        "catchment_minutes",
        env_cfg.get("catchment_minutes", None),
    )
    _catchment_minutes = (
        float(_catchment_minutes_raw) if _catchment_minutes_raw is not None else None
    )
    _dense_distances = bool(
        overrides.pop("dense_distances", env_cfg.get("dense_distances", True))
    )
    kw: dict = dict(
        grid_path=env_cfg.get("grid_path", "data/processed/demand_grid.parquet"),
        stores_path=env_cfg.get("stores_path", "data/processed/supermarkets.parquet"),
        travel_times_path=env_cfg.get(
            "travel_times_path", "data/processed/travel_times.parquet"
        ),
        lambda_val=float(env_cfg["lambda_val"]),
        q_S=float(env_cfg.get("q_S", 0.8)),
        q_B=float(env_cfg.get("q_B", 1.5)),
        alpha_L=float(env_cfg.get("alpha_L", 0.5)),
        alpha_H=float(env_cfg.get("alpha_H", 1.5)),
        beta_effort=float(env_cfg.get("beta_effort", 0.001)),
        kappa0=float(env_cfg.get("kappa0", 1.0)),
        store_size=float(env_cfg.get("store_size", 600.0)),
        transport_cost=float(env_cfg.get("transport_cost", 0.01)),
        a0=float(env_cfg.get("a0", -1.0)),
        mu=float(env_cfg.get("mu", 0.25)),
        nan_fill_minutes=float(env_cfg.get("nan_fill_minutes", 120.0)),
        marginal_cost_D=float(env_cfg.get("marginal_cost_D", 0.0)),
        marginal_cost_S=float(env_cfg.get("marginal_cost_S", 0.0)),
        marginal_cost_B=float(env_cfg.get("marginal_cost_B", 0.0)),
        rent_scale=float(env_cfg.get("rent_scale", 0.0)),
        rent_normalization=str(env_cfg.get("rent_normalization", "mean_ratio")),
        dense_distances=_dense_distances,
        catchment_minutes=_catchment_minutes,
        catchment_k_min=int(env_cfg.get("catchment_k_min", 12)),
        catchment_k_max=int(env_cfg.get("catchment_k_max", 80)),
        precompute_expweights=bool(env_cfg.get("precompute_expweights", False)),
        low_precision_storage=bool(env_cfg.get("low_precision_storage", False)),
    )
    kw.update(overrides)
    return kw


def _dense_demand(city, firm_arrays, prices, efforts, transport_cost: float) -> np.ndarray:
    """Dense branch of market_clearing_arrays."""
    return logit_demand(
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


def _catchment_size_stats(city) -> None:
    indptr = city.catch_indptr
    if indptr is None:
        print("  (no catchment CSR on city)")
        return
    sizes = np.diff(indptr.astype(np.int64))
    print(
        f"  catchment stores/cell: min={sizes.min()}, max={sizes.max()}, "
        f"mean={sizes.mean():.2f}, median={np.median(sizes):.1f}, "
        f"empty_cells={(sizes == 0).sum()}"
    )


def main() -> None:
    config = load_config(env_yaml=_ENV_YAMLS["inner_ring"])
    env_cfg = config["env"]
    if float(env_cfg.get("lambda_val", 0)) == 1500.0:
        env_cfg["lambda_val"] = 429.2

    agent_cfg = config.get("agents", {})
    e_max = float(agent_cfg.get("e_max", 10.0))
    tc = float(env_cfg["transport_cost"])
    n_draws = 50
    n_timing = 300
    rtol_max = 5e-3

    print("Loading city_dense (dense matrix, no catchment CSR) …")
    city_dense, firms_dense = load_berlin_city(
        **_loader_kwargs(env_cfg, dense_distances=True, catchment_minutes=None)
    )
    firm_arrays = precompute_firm_arrays(firms_dense)
    N = len(firms_dense)

    print("Loading city_catch (dense + 20 min catchment, expweights) …")
    city_catch, _firms_catch = load_berlin_city(
        **_loader_kwargs(
            env_cfg,
            dense_distances=True,
            catchment_minutes=20.0,
            catchment_k_min=12,
            catchment_k_max=60,
            precompute_expweights=True,
        )
    )

    rng = np.random.default_rng(42)
    rel_errors: list[float] = []

    print(f"\nCorrectness ({n_draws} random price/effort draws, seed=42):")
    for _ in range(n_draws):
        prices = rng.uniform(0.6, 0.95, size=N).astype(np.float64)
        efforts = rng.uniform(0.0, e_max, size=N).astype(np.float64)
        d_dense = _dense_demand(city_dense, firm_arrays, prices, efforts, tc)
        d_catch = catchment_demand(
            city=city_catch, prices=prices, efforts=efforts, transport_cost=tc
        )
        denom = np.linalg.norm(d_dense)
        rel = float(np.linalg.norm(d_catch - d_dense) / denom) if denom > 0 else 0.0
        rel_errors.append(rel)

    rel_errors_arr = np.asarray(rel_errors)
    max_rel = float(rel_errors_arr.max())
    mean_rel = float(rel_errors_arr.mean())
    print(f"  max relative L2 error:  {max_rel:.6e}")
    print(f"  mean relative L2 error: {mean_rel:.6e}")

    ok_correct = max_rel < rtol_max
    if ok_correct:
        print(f"  CORRECTNESS: PASS (max < {rtol_max:.0e})")
    else:
        print(f"  CORRECTNESS: FAIL (max >= {rtol_max:.0e})")
        _catchment_size_stats(city_catch)
    assert ok_correct, f"max relative error {max_rel:.6e} >= {rtol_max}"

    # Timing — fixed vectors, JIT warmup excluded
    prices_fix = rng.uniform(0.6, 0.95, size=N).astype(np.float64)
    efforts_fix = rng.uniform(0.0, e_max, size=N).astype(np.float64)

    _dense_demand(city_dense, firm_arrays, prices_fix, efforts_fix, tc)
    catchment_demand(city=city_catch, prices=prices_fix, efforts=efforts_fix, transport_cost=tc)

    t0 = time.perf_counter()
    for _ in range(n_timing):
        _dense_demand(city_dense, firm_arrays, prices_fix, efforts_fix, tc)
    dense_ms = (time.perf_counter() - t0) / n_timing * 1000.0

    t0 = time.perf_counter()
    for _ in range(n_timing):
        catchment_demand(
            city=city_catch, prices=prices_fix, efforts=efforts_fix, transport_cost=tc
        )
    catch_ms = (time.perf_counter() - t0) / n_timing * 1000.0

    speedup = dense_ms / catch_ms if catch_ms > 0 else float("inf")
    kernel_label = "expweights" if city_catch.precompute_expweights else "stable"

    print(f"\nTiming ({n_timing} periods each, after JIT warmup):")
    print(f"  dense      : {dense_ms:.3f} ms/period")
    print(f"  catchment ({kernel_label}): {catch_ms:.3f} ms/period")
    print(f"  precompute_expweights: {city_catch.precompute_expweights}")
    print(f"  speedup    : {speedup:.2f}x")

    ok_speed = speedup > 5.0
    print(f"  SPEED: {'PASS' if ok_speed else 'FAIL'} (need > 5x)")
    assert ok_speed, f"speedup {speedup:.2f}x <= 5"

    import numba

    print(f"\nnumba threads: {numba.get_num_threads()}")
    print(
        "  Tune with NUMBA_NUM_THREADS before running "
        "(Apple Silicon: try performance-core count, e.g. NUMBA_NUM_THREADS=6)."
    )


if __name__ == "__main__":
    main()
