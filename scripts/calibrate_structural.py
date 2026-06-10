#!/usr/bin/env python
"""Structural calibration CLI — method-of-moments for (mu, a0) only.

Loads empirical targets from configs/calibration/targets.yaml, fixes transport
cost t, marginal costs c, qualities q_S/q_B, and alpha_ratio from external data
(ADR-024/025/028), then solves for logit scale mu and outside option a0.

Usage
-----
    conda activate py314

    # Smoke test (no YAML write):
    python scripts/calibrate_structural.py --dry-run --max-nfev 25

    # Full calibration (~1-2 min; City built once):
    python scripts/calibrate_structural.py

    # Custom output path:
    python scripts/calibrate_structural.py --output-yaml configs/env/my_calibrated.yaml
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibrate_structural")

_TARGETS_YAML = _REPO_ROOT / "configs" / "calibration" / "targets.yaml"
_ENV_YAML = _REPO_ROOT / "configs" / "env" / "berlin_inner_ring.yaml"
_DEFAULT_OUTPUT = _REPO_ROOT / "configs" / "env" / "berlin_inner_ring_calibrated.yaml"


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
    """Mirror run_baseline.py lambda resolution."""
    if cli_lambda is not None:
        return cli_lambda

    lam = float(env_cfg.get("lambda_val", 0))
    if lam == 1500.0:
        logger.info(
            "lambda_val=1500.0 (placeholder). Computing calibrated value …"
        )
        lam = _calibrate_lambda(env_cfg)
    return lam


def _print_report(result: dict) -> None:
    t = result["t"]
    c = result["c"]
    validation = result["validation_targets"]
    print("\n" + "=" * 72)
    print("  STRUCTURAL CALIBRATION REPORT")
    print("=" * 72)

    print("\n  Data-only parameters (fixed before solve)")
    print(f"    transport_cost t:     {t:.6f}  EUR/min (one-way minutes)")
    print(f"    marginal_cost_D:      {c['discount']:.4f}  EUR/basket")
    print(f"    marginal_cost_S:      {c['standard']:.4f}  EUR/basket")
    print(f"    marginal_cost_B:      {c['bio']:.4f}  EUR/basket")
    print(f"    q_S:                  {result['q_S']:.4f}  EUR (price ladder)")
    print(f"    q_B:                  {result['q_B']:.4f}  EUR (price ladder)")
    print(f"    alpha_L:              {result['alpha_L']:.6f}")
    print(f"    alpha_H:              {result['alpha_H']:.6f}")
    print(f"    alpha_ratio (H/L):    {result['alpha_ratio']:.4f}  (exogenous)")
    print(f"    pi_H_bar:             {result['pi_H_bar']:.6f}")

    print("\n  Solved structural parameters")
    print(f"    logit_scale (mu):     {result['mu']:.6f}")
    print(f"    outside_option (a0):  {result['a0']:.6f}")

    print("\n  Moment fit (objective)")
    print(f"  {'Moment':<28} {'Target':>12} {'Model':>12} {'Rel.err':>12}")
    print("  " + "-" * 66)
    for label, key in (
        ("mean_gross_margin", "mean_gross_margin"),
        ("outside_share", "outside_share"),
    ):
        target = result["moments_target"][key]
        model = result["moments_model"][key]
        rel_err = (model - target) / target if target != 0 else float("nan")
        print(f"  {label:<28} {target:12.6f} {model:12.6f} {rel_err:12.6f}")

    print("\n  Validation (not targeted)")
    print(f"  {'Moment':<28} {'Reference':>12} {'Model':>12}")
    print("  " + "-" * 66)
    model = result["moments_model"]
    print(
        f"  {'chain_share_discount':<28} "
        f"{validation['chain_share_discount']:12.6f} "
        f"{model['chain_share_discount']:12.6f}"
    )
    print(
        f"  {'chain_share_standard':<28} "
        f"{validation['chain_share_standard']:12.6f} "
        f"{model['chain_share_standard']:12.6f}"
    )
    print(
        f"  {'chain_share_bio':<28} "
        f"{validation['chain_share_bio']:12.6f} "
        f"{model['chain_share_bio']:12.6f}"
    )
    print(
        f"  {'bio_income_gradient':<28} "
        f"{validation['bio_income_gradient_ref']:12.6f} "
        f"{model['bio_income_gradient']:12.6f}"
    )
    print("  (chain shares: store-count proxy reference; not in objective)")

    print(f"\n  residual_norm: {result['residual_norm']:.6e}")
    print(f"  success:       {result['success']}")
    print(f"  nfev:          {result['nfev']}")
    print("=" * 72 + "\n")


def _write_calibrated_yaml(
    env_cfg: dict,
    result: dict,
    output_path: Path,
) -> Path:
    out_cfg = copy.deepcopy(env_cfg)
    out_cfg["transport_cost"] = float(result["t"])
    out_cfg["logit_scale"] = float(result["mu"])
    out_cfg["outside_option"] = float(result["a0"])
    out_cfg["q_S"] = float(result["q_S"])
    out_cfg["q_B"] = float(result["q_B"])
    out_cfg["alpha_L"] = float(result["alpha_L"])
    out_cfg["alpha_H"] = float(result["alpha_H"])
    out_cfg["marginal_cost_D"] = float(result["c"]["discount"])
    out_cfg["marginal_cost_S"] = float(result["c"]["standard"])
    out_cfg["marginal_cost_B"] = float(result["c"]["bio"])

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False, default_flow_style=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structural calibration (method of moments) for Berlin inner ring."
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=None,
        help="Override lambda_val (default: env YAML; auto-calibrate if 1500.0)",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=40,
        help="Maximum residual evaluations for scipy least_squares (default: 40)",
    )
    parser.add_argument(
        "--output-yaml",
        type=str,
        default=str(_DEFAULT_OUTPUT),
        help="Path for calibrated env YAML output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print report; do not write output YAML",
    )
    args = parser.parse_args()

    targets = _load_yaml(_TARGETS_YAML)
    env_cfg = _load_yaml(_ENV_YAML)

    lambda_val = _resolve_lambda(env_cfg, args.lambda_val)
    logger.info("Using lambda_val = %.4f", lambda_val)

    from hotelling.calibration.structural import calibrate_structural

    grid_path = str(_REPO_ROOT / env_cfg.get("grid_path", "data/processed/demand_grid.parquet"))
    stores_path = str(
        _REPO_ROOT / env_cfg.get("stores_path", "data/processed/supermarkets.parquet")
    )
    travel_times_path = str(
        _REPO_ROOT
        / env_cfg.get("travel_times_path", "data/processed/travel_times.parquet")
    )

    logger.info("Starting structural calibration (max_nfev=%d) …", args.max_nfev)
    result = calibrate_structural(
        targets=targets,
        env_cfg=env_cfg,
        grid_path=grid_path,
        stores_path=stores_path,
        travel_times_path=travel_times_path,
        lambda_val=lambda_val,
        max_nfev=args.max_nfev,
    )

    _print_report(result)

    if not args.dry_run:
        out_path = _write_calibrated_yaml(
            env_cfg, result, Path(args.output_yaml)
        )
        print(f"Calibrated env YAML written → {out_path}")

    print(
        "Note: benchmarks_cache.npz will be auto-invalidated because "
        "_param_signature includes mu/alpha/costs; the next run_baseline.py "
        "recomputes Nash/monopoly automatically."
    )
    print(
        "Note: City is built once and mu/a0 are mutated in place — full "
        "calibration is ~1-2 minutes, not ~30."
    )


if __name__ == "__main__":
    main()
