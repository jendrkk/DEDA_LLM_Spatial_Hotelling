#!/usr/bin/env python
"""Effort margin calibration guard for the Berlin spatial Hotelling model.

Verifies that the (beta_effort, kappa0, e_max) triple in
configs/env/berlin_inner_ring.yaml produces an **interior** single-firm effort
best response — i.e. the profit-maximising effort level is strictly between 0
and e_max, so the effort dimension of the Q-table is non-degenerate.

Algorithm
---------
1. Load City + Firms from the Berlin parquet files using the env config.
2. Choose a representative symmetric price:  midpoint of the auto price grid
   (average of Bertrand-Nash and joint-monopoly benchmarks for all stores equal).
3. For a sample of up to N_SAMPLE stores (default 30), sweep effort
   e ∈ linspace(0, e_max, N_SWEEP=200) for that store while fixing all other
   stores at effort=0.  Compute profit_j(e) = (p - c) * D_j(e) - 0.5*κ₀*e²
   using core.market.market_clearing.
4. Report argmax e* for each sampled store, the median e*, and a verdict:
   - INTERIOR   : median e* ∈ (0.05·e_max, 0.95·e_max)
   - CORNER-LOW  : median e* ≤ 0.05·e_max  → lower kappa0 or raise beta_effort
   - CORNER-HIGH : median e* ≥ 0.95·e_max  → raise kappa0 or lower beta_effort
5. Print the closed-form FOC target from the spatial logit first-order condition:
       e*_foc = (p - c) · (β/μ) · D_j · (1 - s_j_local) / κ₀
   approximated numerically using central-difference dD_j/de at the sweep argmax.

This script is diagnostic only: it prints results and exits 0 regardless of the
verdict.  The verdict should be checked manually before launching an effort-
activated training run.

Usage
-----
    conda activate py314
    python scripts/check_effort_calibration.py

    # Override lambda (if not yet set in config):
    python scripts/check_effort_calibration.py --lambda-val 1800.0

    # Adjust e_max for the sweep independently of the config:
    python scripts/check_effort_calibration.py --e-max 5.0

    # Sample more stores:
    python scripts/check_effort_calibration.py --n-sample 60

References
----------
Calvano et al. (2020 AER) §II.A — profit function and effort FOC.
ADR-014: marginal cost = 0 for all chain types.
ADR-017: kappa0 is chain-invariant.
ADR-021: effort activation and calibration guard design.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# ── Ensure repo src is on path ────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("check_effort_calibration")

# ── Constants ─────────────────────────────────────────────────────────────────
N_SWEEP   = 200   # effort grid points in the numerical sweep
N_SAMPLE  = 30    # maximum number of stores to sample
CORNER_LO = 0.05  # fraction of e_max below which result is CORNER-LOW
CORNER_HI = 0.95  # fraction of e_max above which result is CORNER-HIGH


# ---------------------------------------------------------------------------
# Config loading (mirrors load_config in run_baseline.py)
# ---------------------------------------------------------------------------

def _load_yaml(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        return yaml.safe_load(f) or {}


def load_env_and_agents_cfg() -> tuple[dict, dict]:
    env_cfg    = _load_yaml(_REPO_ROOT / "configs" / "env"    / "berlin_inner_ring.yaml")
    agents_cfg = _load_yaml(_REPO_ROOT / "configs" / "agents" / "qlearning_effort.yaml")
    env_cfg.setdefault("mu",  env_cfg.pop("logit_scale",    0.25))
    env_cfg.setdefault("a0",  env_cfg.pop("outside_option", -1.0))
    return env_cfg, agents_cfg


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_sweep(
    env_cfg: dict,
    agents_cfg: dict,
    e_max_override: float | None = None,
    n_sample: int = N_SAMPLE,
    n_sweep: int = N_SWEEP,
) -> None:
    """Load city, sweep effort for sampled stores, print calibration verdict."""
    from hotelling.spatial.loader import load_berlin_city
    from hotelling.core.market import market_clearing
    from hotelling.core.equilibrium import bertrand_nash, joint_monopoly

    # ── Extract parameters ────────────────────────────────────────────────────
    beta_effort = float(env_cfg.get("beta_effort", 0.001))
    kappa0      = float(env_cfg.get("kappa0",      1.0))
    transport_cost = float(env_cfg.get("transport_cost", 0.01))
    e_max       = float(e_max_override or agents_cfg.get("e_max", 10.0))
    mu          = float(env_cfg.get("mu",  0.25))

    print("\n" + "="*65)
    print("  EFFORT CALIBRATION CHECK")
    print("="*65)
    print(f"  beta_effort = {beta_effort}")
    print(f"  kappa0      = {kappa0}")
    print(f"  e_max       = {e_max}")
    print(f"  mu          = {mu}")
    print(f"  transport_cost = {transport_cost}")
    print()

    # ── Load City ────────────────────────────────────────────────────────────
    logger.info("Loading Berlin city from parquet files …")
    city, firms = load_berlin_city(
        grid_path           = _REPO_ROOT / env_cfg.get("grid_path",
                                "data/processed/demand_grid.parquet"),
        stores_path         = _REPO_ROOT / env_cfg.get("stores_path",
                                "data/processed/supermarkets.parquet"),
        travel_times_path   = _REPO_ROOT / env_cfg.get("travel_times_path",
                                "data/processed/travel_times.parquet"),
        lambda_val          = float(env_cfg["lambda_val"]),
        q_S                 = float(env_cfg.get("q_S",  0.8)),
        q_B                 = float(env_cfg.get("q_B",  1.5)),
        alpha_L             = float(env_cfg.get("alpha_L", 0.5)),
        alpha_H             = float(env_cfg.get("alpha_H", 1.5)),
        beta_effort         = beta_effort,
        kappa0              = kappa0,
        transport_cost      = transport_cost,
        a0                  = float(env_cfg.get("a0", -1.0)),
        mu                  = mu,
        nan_fill_minutes    = float(env_cfg.get("nan_fill_minutes", 120.0)),
        marginal_cost_D     = float(env_cfg.get("marginal_cost_D", 0.0)),
        marginal_cost_S     = float(env_cfg.get("marginal_cost_S", 0.0)),
        marginal_cost_B     = float(env_cfg.get("marginal_cost_B", 0.0)),
    )
    N = len(firms)
    logger.info("City loaded: %d stores.", N)

    # ── Symmetric representative price ───────────────────────────────────────
    # Use midpoint of Bertrand-Nash and joint-monopoly benchmarks.
    logger.info("Computing Bertrand-Nash and joint-monopoly benchmarks …")
    try:
        p_nash_arr, _ = bertrand_nash(city, transport_cost=transport_cost)
        p_mono_arr, _ = joint_monopoly(city, transport_cost=transport_cost)
        p_rep = float(0.5 * (p_nash_arr.mean() + p_mono_arr.mean()))
        logger.info(
            "Benchmarks: p_Nash=%.4f, p_Mono=%.4f → p_rep=%.4f",
            p_nash_arr.mean(), p_mono_arr.mean(), p_rep,
        )
    except Exception as exc:
        logger.warning("Benchmark computation failed (%s); using p_rep=1.0.", exc)
        p_rep = 1.0

    prices_base = np.full(N, p_rep, dtype=np.float64)
    efforts_zero = np.zeros(N, dtype=np.float64)
    marginal_costs = np.array([f.marginal_cost for f in firms], dtype=np.float64)

    # ── Store sample ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(N, size=min(n_sample, N), replace=False)
    sample_idx.sort()

    # ── Effort sweep per sampled store ───────────────────────────────────────
    effort_grid_sweep = np.linspace(0.0, e_max, n_sweep)
    star_efforts = np.empty(len(sample_idx))

    print(f"  {'Store':>5}  {'chain':>12}  {'e*':>8}  {'e*/e_max':>10}  {'profit@e*':>12}")
    print("  " + "-"*55)

    for idx_pos, j in enumerate(sample_idx):
        profits_sweep = np.empty(n_sweep)
        efforts_j = efforts_zero.copy()
        mc_j = marginal_costs[j]
        for k, e_j in enumerate(effort_grid_sweep):
            efforts_j[j] = e_j
            demands, profs = market_clearing(
                prices=prices_base,
                efforts=efforts_j,
                city=city,
                transport_cost=transport_cost,
            )
            # profit = (p - c) * D - 0.5 * kappa0 * e²   (rent=0 by ADR-015)
            profits_sweep[k] = (
                (prices_base[j] - mc_j) * demands[j]
                - 0.5 * kappa0 * e_j ** 2
            )
        efforts_j[j] = 0.0  # reset

        best_k = int(np.argmax(profits_sweep))
        e_star = effort_grid_sweep[best_k]
        star_efforts[idx_pos] = e_star
        chain_label = getattr(firms[j], "chain", None) or firms[j].id
        print(
            f"  {j:>5}  {chain_label:>12}  {e_star:>8.4f}  "
            f"{e_star/e_max:>10.3f}  {profits_sweep[best_k]:>12.4f}"
        )

    median_e_star = float(np.median(star_efforts))
    frac = median_e_star / e_max

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    print(f"  Median e* = {median_e_star:.4f}  ({frac:.3f} × e_max = {e_max})")
    print()
    if frac <= CORNER_LO:
        verdict = "CORNER-LOW"
        guidance = (
            "→ Effort is always zero. "
            "Raise beta_effort (increase demand sensitivity to effort) "
            "and/or lower kappa0 (reduce effort cost) in berlin_inner_ring.yaml."
        )
    elif frac >= CORNER_HI:
        verdict = "CORNER-HIGH"
        guidance = (
            "→ Effort is always maximal. "
            "Raise kappa0 (increase effort cost) "
            "and/or lower beta_effort in berlin_inner_ring.yaml. "
            "Alternatively lower e_max so the cost curve bends earlier."
        )
    else:
        verdict = "INTERIOR"
        guidance = (
            "→ Best-response effort is in the interior of [0, e_max]. "
            "The effort Q-table dimension is non-degenerate. "
            "Proceed with --with-effort runs."
        )

    width = 65
    print("  " + "─" * (width - 2))
    print(f"  VERDICT: {verdict}")
    print(f"  {guidance}")
    print("  " + "─" * (width - 2))

    # ── Closed-form FOC target ────────────────────────────────────────────────
    # FOC: (p - c) * dD_j/de = kappa0 * e
    # => e*_foc = (p - c) * dD_j/de(e=0) / kappa0
    # Approximate dD_j/de at e=0 via central difference on the median store.
    j_median = int(sample_idx[len(sample_idx) // 2])
    mc_jm = marginal_costs[j_median]
    h = e_max / n_sweep
    efforts_h = efforts_zero.copy()
    efforts_h[j_median] = h
    dem_hi, _ = market_clearing(prices_base, efforts_h, city, transport_cost)
    efforts_h[j_median] = 0.0
    dem_lo, _ = market_clearing(prices_base, efforts_h, city, transport_cost)
    dd_de = (dem_hi[j_median] - dem_lo[j_median]) / h
    e_star_foc = (p_rep - mc_jm) * dd_de / kappa0 if kappa0 > 0 else float("inf")

    # Logit-based analytic approximation: dD/de ≈ (β/μ) * D * (1 - s_j)
    dem_base, _ = market_clearing(prices_base, efforts_zero, city, transport_cost)
    total_pop = float((city.cell_pop + city.lambda_phi).sum())
    s_j = dem_base[j_median] / total_pop if total_pop > 0 else 0.0
    dd_de_analytic = (beta_effort / mu) * dem_base[j_median] * (1.0 - s_j)
    e_star_analytic = (p_rep - mc_jm) * dd_de_analytic / kappa0 if kappa0 > 0 else float("inf")

    print()
    print("  Closed-form FOC target (median store):")
    print(f"    e*_foc (numerical ∂D/∂e)  = {e_star_foc:.4f}")
    print(f"    e*_foc (logit analytic)   = {e_star_analytic:.4f}")
    print(f"    ∂D_j/∂e  (numerical)      = {dd_de:.6f}")
    print(f"    ∂D_j/∂e  (logit approx)   = {dd_de_analytic:.6f}")
    print()
    print("  Calibration guidance:")
    target_frac_lo, target_frac_hi = 0.3, 0.6
    foc_frac = e_star_analytic / e_max if e_max > 0 else float("nan")
    print(f"    e*_foc / e_max = {foc_frac:.3f}  (target: {target_frac_lo}–{target_frac_hi})")
    if foc_frac < target_frac_lo:
        print(
            f"    → e*_foc is below {target_frac_lo*100:.0f}% of e_max. "
            "Consider reducing e_max to ≈ 2 × e*_foc, or raise beta_effort / lower kappa0."
        )
    elif foc_frac > target_frac_hi:
        print(
            f"    → e*_foc is above {target_frac_hi*100:.0f}% of e_max. "
            "Consider raising e_max to ≈ 2 × e*_foc, or lower beta_effort / raise kappa0."
        )
    else:
        print(
            f"    → e*_foc sits comfortably in [{target_frac_lo*100:.0f}%, "
            f"{target_frac_hi*100:.0f}%] of e_max. Current e_max is well-calibrated."
        )
    print("="*65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check effort best-response is interior (not a corner solution).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lambda-val", type=float, default=None,
        help="Override lambda_val in env config (use calibrated value).",
    )
    parser.add_argument(
        "--e-max", type=float, default=None, dest="e_max",
        help="Override e_max for the sweep (does not modify the config file).",
    )
    parser.add_argument(
        "--n-sample", type=int, default=N_SAMPLE,
        help=f"Number of stores to sample for the sweep (default {N_SAMPLE}).",
    )
    parser.add_argument(
        "--n-sweep", type=int, default=N_SWEEP,
        help=f"Number of effort grid points in each store's sweep (default {N_SWEEP}).",
    )
    args = parser.parse_args()

    env_cfg, agents_cfg = load_env_and_agents_cfg()

    if args.lambda_val is not None:
        env_cfg["lambda_val"] = args.lambda_val
        logger.info("CLI override: lambda_val = %.4f", args.lambda_val)

    if float(env_cfg.get("lambda_val", 0)) <= 0:
        logger.warning(
            "lambda_val is 0 or not set. Run "
            "  python scripts/run_baseline.py --calibrate-only\n"
            "and set lambda_val in configs/env/berlin_inner_ring.yaml first."
        )

    run_sweep(
        env_cfg=env_cfg,
        agents_cfg=agents_cfg,
        e_max_override=args.e_max,
        n_sample=args.n_sample,
        n_sweep=args.n_sweep,
    )


if __name__ == "__main__":
    main()
