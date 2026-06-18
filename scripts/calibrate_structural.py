#!/usr/bin/env python
"""Structural calibration CLI.

Two methods are supported, selectable via --method:

  mom_2param    (default, ADR-026)
      The existing 2-parameter least-squares method-of-moments solve for
      (mu, a0) holding (q_S, q_B, alpha_ratio) fixed as priors. Inverts the
      equilibrium mapping by repeated Bertrand-Nash solves; ~60 solver
      evaluations. Suitable when only outside_share and mean_gross_margin
      moments are trusted and q is taken from the price ladder.

  foc_inversion (ADR-032)
      Closed-form inversion of (mu, q_S, q_B) using the Nash-FOC identity
      and the aggregate logit log-share-ratio. Requires one external input:
      the empirical s_B/s_D inside-market-share ratio (configured under the
      foc_inversion: block in targets.yaml). Optionally refines a_0 by 1D
      root-find. Addresses Assumptions 1 and 2 from the calibration audit.

Both methods write to the same output YAML schema; a 'calibration_method'
key in the output records which one was used.

Usage
-----
    conda activate py314

    # Default behaviour (MoM, preserves prior runs):
    python scripts/calibrate_structural.py

    # FOC-inversion (uses targets.yaml :: foc_inversion block):
    python scripts/calibrate_structural.py --method foc_inversion

    # Smoke test, no YAML write:
    python scripts/calibrate_structural.py --method foc_inversion --dry-run
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
    method = result.get("method", "mom_2param")
    t = result["t"]
    c = result["c"]
    print("\n" + "=" * 72)
    print(f"  STRUCTURAL CALIBRATION REPORT  —  method = {method}")
    print("=" * 72)
    print("\n  Data-only parameters (fixed before solve)")
    print(f"    transport_cost t:     {t:.6f}  EUR/min (one-way minutes)")
    print(f"    marginal_cost_D:      {c['discount']:.4f}  EUR/basket")
    print(f"    marginal_cost_S:      {c['standard']:.4f}  EUR/basket")
    print(f"    marginal_cost_B:      {c['bio']:.4f}  EUR/basket")

    print("\n  Solved structural parameters")
    print(f"    logit_scale (mu):     {result['mu']:.6f}")
    print(f"    outside_option (a0):  {result['a0']:.6f}")
    print(f"    q_S:                  {result['q_S']:.6f}")
    print(f"    q_B:                  {result['q_B']:.6f}")
    print(f"    alpha_L:              {result['alpha_L']:.6f}")
    print(f"    alpha_H:              {result['alpha_H']:.6f}")
    print(f"    alpha_ratio (H/L):    {result['alpha_ratio']:.6f}")
    print(f"    pi_H_bar:             {result['pi_H_bar']:.6f}")

    if method == "foc_inversion":
        _print_foc_diagnostics(result)

    print("\n  Moment fit")
    print(f"  {'Moment':<28} {'Target':>12} {'Model':>12} {'Rel.err':>12}")
    print("  " + "-" * 66)
    rows = [
        ("mean_gross_margin", "mean_gross_margin"),
        ("outside_share", "outside_share"),
        ("chain_share_discount", "chain_share_discount"),
        ("chain_share_bio", "chain_share_bio"),
        ("bio_income_gradient", "bio_income_gradient"),
    ]
    for label, key in rows:
        target = result["moments_target"].get(key, float("nan"))
        model = result["moments_model"].get(key, float("nan"))
        if not (isinstance(target, (int, float)) and target != 0):
            rel_err = float("nan")
        else:
            rel_err = (model - target) / target
        print(f"  {label:<28} {target:12.6f} {model:12.6f} {rel_err:12.6f}")

    if method == "mom_2param":
        print(f"\n  residual_norm: {result['residual_norm']:.6e}")
        print(f"  success:       {result['success']}")
        print(f"  nfev:          {result['nfev']}")
    print("=" * 72 + "\n")


def _print_foc_diagnostics(result: dict) -> None:
    """Print the FOC-inversion-specific diagnostic block."""
    diag = result["foc_diagnostics"]
    shares = result["shares_resolved"]
    access = result["accessibility"]
    print("\n  FOC-inversion inputs (resolved)")
    print(f"    s_B_over_s_D used:    {result['s_B_over_s_D_used']:.6f}")
    print(f"    s_S_over_s_D used:    {result['s_S_over_s_D_used']:.6f}  "
          "(store-count default if not in targets.yaml)")
    print(f"    s_outside target:     {result['moments_target']['outside_share']:.6f}")
    print(f"    a0_refined:           {result['a0_refined']}")

    print("\n  Absolute inside shares (resolved)")
    print(f"    s_D:                  {shares['discount']:.6f}")
    print(f"    s_S:                  {shares['standard']:.6f}")
    print(f"    s_B:                  {shares['bio']:.6f}")

    print("\n  Per-chain-type μ from Nash-FOC inversion  μ_τ = (p_τ − c_τ)(1 − s_τ)")
    print(f"    μ_D:                  {diag['mu_by_type']['discount']:.6f}")
    print(f"    μ_S:                  {diag['mu_by_type']['standard']:.6f}")
    print(f"    μ_B:                  {diag['mu_by_type']['bio']:.6f}")
    print(f"    μ share-weighted:     {diag['mu_share_weighted']:.6f}")
    print(f"    μ simple mean:        {diag['mu_simple_mean']:.6f}")
    print(f"    Spread (absolute):    {diag['spread_absolute']:.6f}")
    print(f"    Spread (relative):    {diag['spread_relative']:.4%}")
    if diag["spread_relative"] > 0.30:
        print("    ⚠ Spread > 30% — share inputs may be inconsistent with FOC.")
        print("       Inspect chain_share targets or use chain-specific costs.")

    print("\n  Population-weighted spatial accessibility A_τ "
          "(higher = more accessible)")
    print(f"    A_D:                  {access['discount']:.6f}")
    print(f"    A_S:                  {access['standard']:.6f}")
    print(f"    A_B:                  {access['bio']:.6f}")


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
    # Method stamp (auditability)
    out_cfg["calibration_method"] = str(result.get("method", "mom_2param"))
    out_cfg["calibration_alpha_ratio"] = float(result["alpha_ratio"])
    if result.get("method") == "foc_inversion":
        out_cfg["calibration_s_B_over_s_D"] = float(result["s_B_over_s_D_used"])
        out_cfg["calibration_s_S_over_s_D"] = float(result["s_S_over_s_D_used"])
        out_cfg["calibration_mu_foc_spread_relative"] = float(
            result["foc_diagnostics"]["spread_relative"]
        )

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
    parser.add_argument(
        "--method",
        type=str,
        default="mom_2param",
        choices=["mom_2param", "foc_inversion"],
        help=(
            "Calibration method. 'mom_2param' (default) = existing 5-moment "
            "MoM solver; 'foc_inversion' = closed-form Nash-FOC + accessibility "
            "inversion using s_B/s_D from targets.yaml. See ADR-026 and ADR-032."
        ),
    )
    parser.add_argument(
        "--s-b-over-s-d",
        type=float,
        default=None,
        help=(
            "Override targets.foc_inversion.s_B_over_s_D from the command line. "
            "Only used when --method foc_inversion. Useful for sensitivity sweeps."
        ),
    )
    parser.add_argument(
        "--no-refine-a0",
        action="store_true",
        help=(
            "When --method foc_inversion, skip the 1-D root-find that adjusts "
            "a_0 to match outside_share_target. a_0 will be carried over from "
            "env_cfg.outside_option."
        ),
    )
    args = parser.parse_args()

    targets = _load_yaml(_TARGETS_YAML)
    env_cfg = _load_yaml(_ENV_YAML)

    lambda_val = _resolve_lambda(env_cfg, args.lambda_val)
    logger.info("Using lambda_val = %.4f", lambda_val)

    # ── Optional CLI override of s_B_over_s_D ─────────────────────────────
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
        _REPO_ROOT
        / env_cfg.get("travel_times_path", "data/processed/travel_times.parquet")
    )

    if args.method == "mom_2param":
        from hotelling.calibration.structural import calibrate_structural

        logger.info("Method: mom_2param  (ADR-026 — existing MoM solver, max_nfev=%d)", args.max_nfev)
        result = calibrate_structural(
            targets=targets,
            env_cfg=env_cfg,
            grid_path=grid_path,
            stores_path=stores_path,
            travel_times_path=travel_times_path,
            lambda_val=lambda_val,
            max_nfev=args.max_nfev,
        )
        result.setdefault("method", "mom_2param")
    elif args.method == "foc_inversion":
        from hotelling.calibration.foc_inversion import calibrate_foc_inversion

        foc_block = dict(targets.get("foc_inversion", {}) or {})
        if foc_block.get("alpha_ratio") is None:
            foc_block["alpha_ratio"] = env_cfg.get(
                "alpha_ratio", targets.get("alpha_ratio", 2.5)
            )
            targets = {**targets, "foc_inversion": foc_block}
        if "s_B_over_s_D" not in foc_block:
            raise KeyError(
                "Method 'foc_inversion' requires targets.foc_inversion.s_B_over_s_D "
                "(no CLI override provided). Edit configs/calibration/targets.yaml "
                "or pass --s-b-over-s-d."
            )
        refine_a0 = (not args.no_refine_a0) and bool(foc_block.get("refine_a0", True))
        force_chain_specific = bool(foc_block.get("force_chain_specific_costs", True))
        logger.info(
            "Method: foc_inversion  (ADR-032 — closed-form; s_B/s_D=%.4f, refine_a0=%s, "
            "force_chain_specific_costs=%s)",
            float(foc_block["s_B_over_s_D"]),
            refine_a0, force_chain_specific,
        )
        result = calibrate_foc_inversion(
            targets=targets,
            env_cfg=env_cfg,
            grid_path=grid_path,
            stores_path=stores_path,
            travel_times_path=travel_times_path,
            lambda_val=lambda_val,
            refine_a0=refine_a0,
            force_chain_specific_costs=force_chain_specific,
        )
    else:
        raise ValueError(f"Unknown --method: {args.method!r}")

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
