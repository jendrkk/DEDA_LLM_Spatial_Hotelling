#!/usr/bin/env python
"""Side-by-side comparison of the two calibration methods (ADR-026 vs ADR-032).

Runs both `calibrate_structural` (mom_2param) and `calibrate_foc_inversion`
on the same targets.yaml + env YAML inputs, then reports the parameter
deltas, moment-fit deltas, the FOC-spread diagnostic, and the realised
Bertrand-Nash chain-type price means under each calibrated parameter set.

This is a READ-ONLY diagnostic. It does NOT write any env YAML. To produce
an actual calibrated YAML, use `scripts/calibrate_structural.py --method
{mom_2param,foc_inversion}`.

Usage
-----
    conda activate py314

    # Standard run (uses targets.yaml :: foc_inversion defaults):
    python scripts/diagnose_calibration_methods.py

    # Override s_B/s_D from CLI (sensitivity sweep):
    python scripts/diagnose_calibration_methods.py --s-b-over-s-d 0.60

    # Reduce solver budget for fast smoke test:
    python scripts/diagnose_calibration_methods.py --max-nfev 8

    # Save JSON artefact:
    python scripts/diagnose_calibration_methods.py \\
        --output-json results/calibration_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

# ── Ensure repo src is on path ────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose_calibration_methods")

_TARGETS_YAML = _REPO_ROOT / "configs" / "calibration" / "targets.yaml"
_ENV_YAML     = _REPO_ROOT / "configs" / "env"        / "berlin_inner_ring.yaml"


# ---------------------------------------------------------------------------
# Helpers (mirror scripts/calibrate_structural.py)
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _calibrate_lambda(env_cfg: dict) -> float:
    """Mirror run_baseline.calibrate_and_print_lambda (no simulation import)."""
    from hotelling.spatial.assembly import calibrate_lambda
    from hotelling.spatial.loader import _compute_phi_i
    import geopandas as gpd

    grid_path = Path(env_cfg.get("grid_path", "data/processed/demand_grid.parquet"))
    if not grid_path.is_absolute():
        grid_path = _REPO_ROOT / grid_path
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid not found: {grid_path}")

    grid = gpd.read_parquet(grid_path)
    if "phi_i" not in grid.columns:
        logger.info("phi_i absent; computing from constituent columns.")
        phi_series = _compute_phi_i(grid)
        grid = grid.copy()
        grid["phi_i"] = phi_series.values

    lam = calibrate_lambda(grid, target_footfall_share=0.125)
    print(f"\n{'='*60}")
    print(f"  Calibrated λ = {lam:.4f}")
    print(
        f"  (α=12.5%, Σω={grid['Einwohner'].sum():.0f}, "
        f"Σφ={grid['phi_i'].sum():.4f})"
    )
    print("  → Set 'lambda_val: {:.1f}' in configs/env/berlin_inner_ring.yaml".format(lam))
    print(f"{'='*60}\n")
    return lam


def _resolve_lambda(env_cfg: dict, cli_lambda: float | None) -> float:
    """Mirror scripts/calibrate_structural.py lambda resolution."""
    if cli_lambda is not None:
        return float(cli_lambda)

    lam = float(env_cfg.get("lambda_val", 0))
    if lam == 1500.0:
        logger.info(
            "lambda_val=1500.0 (placeholder). Computing calibrated value …"
        )
        lam = _calibrate_lambda(env_cfg)
    if lam <= 0:
        raise ValueError(
            "lambda_val unset or non-positive in env YAML. "
            "Run scripts/run_baseline.py --calibrate-only first, or pass "
            "--lambda-val on the CLI."
        )
    return lam


def _chain_price_means_under(
    *, env_cfg: dict, grid_path: str, stores_path: str,
    travel_times_path: str, lambda_val: float, result: dict,
) -> Dict[str, float]:
    """Build the calibration city with `result`'s parameters, solve
    Bertrand-Nash, and report the mean Nash price per chain type.

    Returns
    -------
    {'discount': mean_p_D, 'standard': mean_p_S, 'bio': mean_p_B}
    """
    from hotelling.calibration.structural import _build_calibration_city
    from hotelling.calibration.moments import _firm_chain_types
    from hotelling.core.equilibrium import bertrand_nash

    city = _build_calibration_city(
        grid_path=grid_path,
        stores_path=stores_path,
        travel_times_path=travel_times_path,
        lambda_val=lambda_val,
        env_cfg=env_cfg,
        transport_cost=float(result["t"]),
        costs=result["c"],
        mu=float(result["mu"]),
        a0=float(result["a0"]),
        q_S=float(result["q_S"]),
        q_B=float(result["q_B"]),
        alpha_L=float(result["alpha_L"]),
        alpha_H=float(result["alpha_H"]),
    )
    prices, _efforts = bertrand_nash(city, float(result["t"]), cache_path=None)
    chain_types = _firm_chain_types(city)
    out: Dict[str, float] = {}
    for tau in ("discount", "standard", "bio"):
        mask = chain_types == tau
        if mask.sum() == 0:
            out[tau] = float("nan")
        else:
            out[tau] = float(prices[mask].mean())
    return out


def _fmt(x, w=12, p=4):
    if isinstance(x, float) and (x != x):  # NaN
        return f"{'nan':>{w}}"
    try:
        return f"{float(x):>{w}.{p}f}"
    except (TypeError, ValueError):
        return f"{str(x):>{w}}"


def _print_comparison(
    mom: dict, foc: dict,
    mom_prices: Dict[str, float], foc_prices: Dict[str, float],
) -> None:
    print("\n" + "=" * 86)
    print("  CALIBRATION-METHOD COMPARISON  —  ADR-026 (mom_2param) vs ADR-032 (foc_inversion)")
    print("=" * 86)

    print("\n  Data-only parameters (identical across methods, sanity check)")
    print(f"  {'Parameter':<26} {'mom_2param':>16} {'foc_inversion':>16} {'Δ (foc − mom)':>16}")
    print("  " + "-" * 80)
    for label, mkey, fkey in [
        ("transport_cost t (EUR/min)", "t", "t"),
    ]:
        m = mom[mkey]; f = foc[fkey]
        print(f"  {label:<26} {_fmt(m, 16, 6)} {_fmt(f, 16, 6)} {_fmt(f - m, 16, 6)}")
    for tau in ("discount", "standard", "bio"):
        m = mom["c"][tau]; f = foc["c"][tau]
        print(f"  {'c_' + tau[0].upper():<26} {_fmt(m, 16, 4)} {_fmt(f, 16, 4)} {_fmt(f - m, 16, 4)}")

    print("\n  Solved structural parameters")
    print(f"  {'Parameter':<26} {'mom_2param':>16} {'foc_inversion':>16} {'Δ (foc − mom)':>16}")
    print("  " + "-" * 80)
    for label, key in [
        ("mu",        "mu"),
        ("a0",        "a0"),
        ("q_S",       "q_S"),
        ("q_B",       "q_B"),
        ("alpha_L",   "alpha_L"),
        ("alpha_H",   "alpha_H"),
        ("alpha_ratio", "alpha_ratio"),
    ]:
        m = mom[key]; f = foc[key]
        print(f"  {label:<26} {_fmt(m, 16, 6)} {_fmt(f, 16, 6)} {_fmt(f - m, 16, 6)}")

    print("\n  FOC-inversion diagnostics (foc_inversion only)")
    diag = foc["foc_diagnostics"]
    print(f"    mu_D, mu_S, mu_B (FOC-implied) = "
          f"{diag['mu_by_type']['discount']:.4f}, "
          f"{diag['mu_by_type']['standard']:.4f}, "
          f"{diag['mu_by_type']['bio']:.4f}")
    print(f"    spread (max − min):              {diag['spread_absolute']:.4f}")
    print(f"    spread relative to mean:         {diag['spread_relative']:.4%}")
    if diag["spread_relative"] > 0.30:
        print("    ⚠ Spread > 30% — shares may be inconsistent with FOC.")

    print(f"\n    s_B/s_D used: {foc['s_B_over_s_D_used']:.4f}   "
          f"s_S/s_D used: {foc['s_S_over_s_D_used']:.4f}")

    print("\n  Moment fit (achieved by each method)")
    print(f"  {'Moment':<28} {'Target':>10} {'mom_2param':>14} {'foc_inversion':>14}")
    print("  " + "-" * 80)
    moment_keys = [
        "mean_gross_margin",
        "outside_share",
        "chain_share_discount",
        "chain_share_bio",
        "bio_income_gradient",
    ]
    for key in moment_keys:
        # Targets may differ slightly between methods (foc uses derived
        # discount/bio targets from the resolved shares)
        tgt_m = mom["moments_target"].get(key, float("nan"))
        mod_m = mom["moments_model"].get(key, float("nan"))
        mod_f = foc["moments_model"].get(key, float("nan"))
        print(f"  {key:<28} {_fmt(tgt_m, 10, 4)} {_fmt(mod_m, 14, 4)} {_fmt(mod_f, 14, 4)}")

    print("\n  Mean Bertrand-Nash price per chain type under each calibration")
    print(f"  {'Chain type':<26} {'mom_2param':>16} {'foc_inversion':>16} {'Δ (foc − mom)':>16}")
    print("  " + "-" * 80)
    for tau in ("discount", "standard", "bio"):
        m = mom_prices[tau]; f = foc_prices[tau]
        print(f"  {'mean p_' + tau[0].upper() + ' (EUR)':<26} "
              f"{_fmt(m, 16, 4)} {_fmt(f, 16, 4)} {_fmt(f - m, 16, 4)}")

    print("\n  Recommendation")
    if foc["foc_diagnostics"]["spread_relative"] < 0.15:
        rec = ("FOC-spread is tight (<15%). foc_inversion is internally "
               "consistent with the Nash-FOC; prefer for the main run.")
    elif foc["foc_diagnostics"]["spread_relative"] < 0.30:
        rec = ("FOC-spread is moderate (15–30%). Defensible but report the "
               "spread as a robustness caveat in the thesis.")
    else:
        rec = ("FOC-spread is large (>30%). Inputs are inconsistent with "
               "Bertrand-Nash play. Replace the store-count s_B/s_D proxy "
               "with empirical data (BKartA, Google Maps, GfK panel) before "
               "trusting the foc_inversion output.")
    print(f"    {rec}")
    print("=" * 86 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of mom_2param vs foc_inversion."
    )
    parser.add_argument(
        "--lambda-val", type=float, default=None,
        help="Override lambda_val (default: env YAML).",
    )
    parser.add_argument(
        "--max-nfev", type=int, default=60,
        help="Max residual evaluations for mom_2param least_squares (default: 60).",
    )
    parser.add_argument(
        "--s-b-over-s-d", type=float, default=None,
        help="Override targets.foc_inversion.s_B_over_s_D for the foc_inversion run.",
    )
    parser.add_argument(
        "--no-refine-a0", action="store_true",
        help="Skip the a_0 root-find in the foc_inversion run.",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Optional path to write a JSON artefact with both result dicts.",
    )
    args = parser.parse_args()

    targets = _load_yaml(_TARGETS_YAML)
    env_cfg = _load_yaml(_ENV_YAML)
    lambda_val = _resolve_lambda(env_cfg, args.lambda_val)
    logger.info("Using lambda_val = %.4f", lambda_val)

    if args.s_b_over_s_d is not None:
        targets.setdefault("foc_inversion", {})
        targets["foc_inversion"]["s_B_over_s_D"] = float(args.s_b_over_s_d)
        logger.info(
            "Overriding targets.foc_inversion.s_B_over_s_D = %.6f from CLI",
            args.s_b_over_s_d,
        )

    grid_path = str(_REPO_ROOT / env_cfg.get("grid_path", "data/processed/demand_grid.parquet"))
    stores_path = str(
        _REPO_ROOT / env_cfg.get("stores_path", "data/processed/supermarkets.parquet")
    )
    travel_times_path = str(
        _REPO_ROOT / env_cfg.get("travel_times_path", "data/processed/travel_times.parquet")
    )

    # ── Run mom_2param ──────────────────────────────────────────────────
    from hotelling.calibration.structural import calibrate_structural
    logger.info("Running mom_2param (max_nfev=%d) …", args.max_nfev)
    mom = calibrate_structural(
        targets=targets, env_cfg=env_cfg,
        grid_path=grid_path, stores_path=stores_path,
        travel_times_path=travel_times_path,
        lambda_val=lambda_val, max_nfev=args.max_nfev,
    )
    mom.setdefault("method", "mom_2param")

    # ── Run foc_inversion ───────────────────────────────────────────────
    from hotelling.calibration.foc_inversion import calibrate_foc_inversion
    foc_block = dict(targets.get("foc_inversion", {}) or {})
    if foc_block.get("alpha_ratio") is None:
        foc_block["alpha_ratio"] = env_cfg.get(
            "alpha_ratio", targets.get("alpha_ratio", 2.5)
        )
        targets = {**targets, "foc_inversion": foc_block}
    if "s_B_over_s_D" not in foc_block:
        raise KeyError(
            "targets.foc_inversion.s_B_over_s_D not set. Edit "
            "configs/calibration/targets.yaml or pass --s-b-over-s-d."
        )
    refine_a0 = (not args.no_refine_a0) and bool(foc_block.get("refine_a0", True))
    force_chain_specific = bool(foc_block.get("force_chain_specific_costs", True))
    logger.info(
        "Running foc_inversion (s_B/s_D=%.4f, refine_a0=%s, "
        "force_chain_specific_costs=%s) …",
        float(foc_block["s_B_over_s_D"]), refine_a0, force_chain_specific,
    )
    foc = calibrate_foc_inversion(
        targets=targets, env_cfg=env_cfg,
        grid_path=grid_path, stores_path=stores_path,
        travel_times_path=travel_times_path,
        lambda_val=lambda_val,
        refine_a0=refine_a0,
        force_chain_specific_costs=force_chain_specific,
    )

    # ── Realised Nash prices under each ─────────────────────────────────
    logger.info("Computing Bertrand-Nash benchmarks under each calibration …")
    mom_prices = _chain_price_means_under(
        env_cfg=env_cfg, grid_path=grid_path, stores_path=stores_path,
        travel_times_path=travel_times_path, lambda_val=lambda_val, result=mom,
    )
    foc_prices = _chain_price_means_under(
        env_cfg=env_cfg, grid_path=grid_path, stores_path=stores_path,
        travel_times_path=travel_times_path, lambda_val=lambda_val, result=foc,
    )

    _print_comparison(mom, foc, mom_prices, foc_prices)

    # ── JSON artefact ───────────────────────────────────────────────────
    if args.output_json is not None:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _to_jsonable(obj):
            if isinstance(obj, dict):
                return {str(k): _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        artefact = {
            "mom_2param": _to_jsonable(mom),
            "foc_inversion": _to_jsonable(foc),
            "nash_prices": {
                "mom_2param": mom_prices,
                "foc_inversion": foc_prices,
            },
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(artefact, f, indent=2, default=str)
        print(f"Diagnostic JSON written → {out_path}")


if __name__ == "__main__":
    main()
