#!/usr/bin/env python
"""Exact verification of effort calibration (ADR-031).

Solves the joint price-effort Bertrand-Nash equilibrium with calibrated
(beta_effort, kappa0) and checks that equilibrium efforts are interior and
that price-side moments drift only slightly when effort is activated.

Usage
-----
    conda activate py314

    python scripts/check_effort_calibration.py \\
        --env-config configs/env/berlin_inner_ring_calibrated.yaml \\
        --m-effort 5 --e-max 1.0 --lambda-val 429.2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("check_effort_calibration")

_TARGETS_YAML = _REPO_ROOT / "configs" / "calibration" / "targets.yaml"
_DEFAULT_ENV = _REPO_ROOT / "configs" / "env" / "berlin_inner_ring_calibrated.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _margin_target(targets: dict) -> float:
    if targets.get("use_common_margin", True):
        return float(targets["gross_margin_common"])
    margins = targets["gross_margin_by_chain"]
    return float(
        sum(margins[tau] for tau in ("discount", "standard", "bio")) / 3.0
    )


def run_check(
    env_cfg: dict,
    targets: dict,
    *,
    e_max: float,
    m_effort: int,
) -> str:
    """Solve joint Nash, report effort distribution and moment drift."""
    from hotelling.calibration.moments import (
        _choice_mass,
        _mean_gross_margin_from,
        _outside_share_from,
    )
    from hotelling.core.equilibrium import bertrand_nash
    from hotelling.spatial.loader import load_berlin_city

    if m_effort <= 1:
        logger.warning(
            "m_effort=%d — effort grid collapses to {0}; "
            "use m_effort>1 for effort-activated verification.",
            m_effort,
        )

    beta_effort = float(env_cfg.get("beta_effort", 0.001))
    kappa0 = float(env_cfg.get("kappa0", 1.0))
    transport_cost = float(env_cfg.get("transport_cost", 0.01))
    basket = float(targets["basket_price_standard_eur"])
    tgt_margin = _margin_target(targets)
    tgt_outside = float(targets["outside_share_target"])

    print("\n" + "=" * 72)
    print("  EFFORT CALIBRATION VERIFICATION (ADR-031)")
    print("=" * 72)
    print(f"  beta_effort = {beta_effort:.6f}")
    print(f"  kappa0      = {kappa0:.6f}")
    print(f"  e_max       = {e_max}")
    print(f"  m_effort    = {m_effort}")
    print(f"  transport_cost = {transport_cost}")
    print()

    grid_path = _REPO_ROOT / env_cfg.get("grid_path", "data/processed/demand_grid.parquet")
    stores_path = _REPO_ROOT / env_cfg.get("stores_path", "data/processed/supermarkets.parquet")
    travel_times_path = _REPO_ROOT / env_cfg.get(
        "travel_times_path", "data/processed/travel_times.parquet"
    )

    logger.info("Loading Berlin city …")
    city, firms = load_berlin_city(
        grid_path=grid_path,
        stores_path=stores_path,
        travel_times_path=travel_times_path,
        lambda_val=float(env_cfg["lambda_val"]),
        q_S=float(env_cfg.get("q_S", 6.0)),
        q_B=float(env_cfg.get("q_B", 18.0)),
        alpha_L=float(env_cfg.get("alpha_L", 0.5)),
        alpha_H=float(env_cfg.get("alpha_H", 1.5)),
        beta_effort=beta_effort,
        kappa0=kappa0,
        transport_cost=transport_cost,
        a0=float(env_cfg.get("outside_option", env_cfg.get("a0", -1.0))),
        mu=float(env_cfg.get("logit_scale", env_cfg.get("mu", 0.25))),
        nan_fill_minutes=float(env_cfg.get("nan_fill_minutes", 120.0)),
        marginal_cost_D=float(env_cfg.get("marginal_cost_D", 0.0)),
        marginal_cost_S=float(env_cfg.get("marginal_cost_S", 0.0)),
        marginal_cost_B=float(env_cfg.get("marginal_cost_B", 0.0)),
        store_size=float(env_cfg.get("store_size", 600.0)),
        rent_scale=float(env_cfg.get("rent_scale", 0.0)),
        rent_normalization=str(env_cfg.get("rent_normalization", "mean_ratio")),
        dense_distances=bool(env_cfg.get("dense_distances", True)),
    )
    logger.info("City loaded: %d stores.", len(firms))

    logger.info("Solving joint price-effort Bertrand-Nash …")
    p_nash, e_nash = bertrand_nash(city, transport_cost=transport_cost, cache_path=None)

    interior_mask = (e_nash > 1e-9) & (e_nash < e_max)
    interior_fraction = float(interior_mask.mean())
    e_mean = float(e_nash.mean())
    e_min = float(e_nash.min())
    e_max_actual = float(e_nash.max())

    wtp_full_pct = beta_effort * e_max / basket
    wtp_equil_pct = beta_effort * e_mean / basket

    print("  Joint Nash equilibrium effort")
    print(f"    e* mean/min/max:   {e_mean:.4f} / {e_min:.4f} / {e_max_actual:.4f}")
    print(f"    interior_fraction: {interior_fraction:.4f}  "
          f"(target >= 0.80, 0 < e* < {e_max})")
    print(f"    wtp_full_pct:      {wtp_full_pct:.4f}  (beta*e_max/basket)")
    print(f"    wtp_equil_pct:     {wtp_equil_pct:.4f}  (beta*mean(e*)/basket)")
    print(f"    p_nash mean:       {float(p_nash.mean()):.4f}")

    inside, outside = _choice_mass(city, transport_cost, p_nash, e_nash)
    model_margin = _mean_gross_margin_from(city, p_nash, inside)
    model_outside = _outside_share_from(city, outside)
    margin_drift = model_margin - tgt_margin
    outside_drift = model_outside - tgt_outside

    print("\n  Price-side moment drift (effort ON vs calibration targets)")
    print(f"  {'Moment':<28} {'Target':>12} {'Model':>12} {'Drift':>12}")
    print("  " + "-" * 66)
    print(
        f"  {'mean_gross_margin':<28} {tgt_margin:12.6f} "
        f"{model_margin:12.6f} {margin_drift:12.6f}"
    )
    print(
        f"  {'outside_share':<28} {tgt_outside:12.6f} "
        f"{model_outside:12.6f} {outside_drift:12.6f}"
    )

    print()
    verdict_parts = []
    if interior_fraction >= 0.80:
        verdict_parts.append("interior_fraction OK")
    else:
        verdict_parts.append("interior_fraction LOW")

    if abs(margin_drift) < 0.02:
        verdict_parts.append("margin drift OK")
    else:
        verdict_parts.append("margin drift LARGE")

    if interior_fraction >= 0.80 and abs(margin_drift) < 0.02:
        verdict = "PASS"
        guidance = (
            "Joint Nash effort is interior and price moments are stable. "
            "Proceed with effort-activated Q-learning runs."
        )
    else:
        verdict = "WARN"
        hints = []
        if interior_fraction < 0.80:
            hints.append(
                "lower effort_interior_target_rho or effort_importance_X in "
                "targets.yaml and re-run calibrate_structural.py"
            )
        if abs(margin_drift) >= 0.02:
            hints.append(
                "re-solve mu/a0 with effort ON if outside-share drift is large"
            )
        guidance = " ".join(hints) if hints else "Review calibration parameters."

    print(f"  VERDICT: {verdict}  ({'; '.join(verdict_parts)})")
    print(f"  {guidance}")
    print("=" * 72 + "\n")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify effort calibration via joint Bertrand-Nash solve (ADR-031).",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=str(_DEFAULT_ENV),
        help="Calibrated env YAML with beta_effort and kappa0",
    )
    parser.add_argument(
        "--m-effort",
        type=int,
        default=5,
        help="Discrete effort grid size (must be >1 to activate effort; default 5)",
    )
    parser.add_argument(
        "--e-max",
        type=float,
        default=1.0,
        help="Effort upper bound (must match targets.effort_e_max; default 1.0)",
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=None,
        help="Override lambda_val in env config",
    )
    args = parser.parse_args()

    env_cfg_path = Path(args.env_config)
    if not env_cfg_path.is_absolute():
        env_cfg_path = _REPO_ROOT / env_cfg_path
    env_cfg = _load_yaml(env_cfg_path)

    if args.lambda_val is not None:
        env_cfg["lambda_val"] = args.lambda_val
        logger.info("CLI override: lambda_val = %.4f", args.lambda_val)

    targets = _load_yaml(_TARGETS_YAML)
    run_check(
        env_cfg,
        targets,
        e_max=args.e_max,
        m_effort=args.m_effort,
    )


if __name__ == "__main__":
    main()
