#!/usr/bin/env python
"""Berlin Inner-Ringbahn Baseline Simulation — Phase 0 Q-learning burn-in.

Runs the Phase 0 burn-in (Q-learning incumbents only, no LLM CEO, no
entrant) on the real Berlin spatial grid and reports:

    - Calvano collusion index Δ = (p_mean - p_Nash) / (p_Mono - p_Nash)
    - Mean converged price vs Bertrand-Nash and joint-monopoly benchmarks
    - Steps to convergence
    - Price and effort convergence plot (if matplotlib available)

Outputs are saved to a timestamped run folder under results/runs/,
with results/index.csv tracking all runs.

Usage
-----
    # Activate the environment first:
    conda activate py314   # or: source .venv/bin/activate

    # Basic run with default config:
    python scripts/run_baseline.py

    # Override seed:
    python scripts/run_baseline.py --seed 123

    # Calibrate lambda only (print value, do not run simulation):
    python scripts/run_baseline.py --calibrate-only

    # Use custom lambda (override the config file value):
    python scripts/run_baseline.py --lambda-val 1234.5

Calibration note
----------------
Before the first full run, calibrate λ:
    python scripts/run_baseline.py --calibrate-only
Copy the printed λ value into configs/env/berlin_inner_ring.yaml as lambda_val.

Transport cost note
-------------------
dist2_km2 in City holds travel-time minutes (not km²). transport_cost is
therefore in €/min. The default 0.01 €/min means a 10-minute trip costs
0.1 € disutility — comparable to logit scale μ=0.25. Adjust after checking
that Bertrand-Nash prices fall within the price grid [min_price, max_price].
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
logger = logging.getLogger("run_baseline")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_ENV_YAMLS = {
    "inner_ring": _REPO_ROOT / "configs" / "env" / "berlin_inner_ring.yaml",
    "full":       _REPO_ROOT / "configs" / "env" / "berlin_full.yaml",
}


def load_config(
    env_yaml:    Path = _ENV_YAMLS["inner_ring"],
    agents_yaml: Path = _REPO_ROOT / "configs" / "agents" / "qlearning_baseline.yaml",
    phase0_yaml: Path = _REPO_ROOT / "configs" / "simulation" / "phase0_baseline.yaml",
) -> dict:
    """Load and merge the three config YAML files into one nested dict."""
    def _load(p: Path) -> dict:
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        with p.open() as f:
            return yaml.safe_load(f) or {}

    env_cfg    = _load(env_yaml)
    agent_cfg  = _load(agents_yaml)
    phase0_cfg = _load(phase0_yaml)

    # Map YAML field names to loader kwarg names
    env_cfg.setdefault("mu",   env_cfg.pop("logit_scale",   0.25))
    env_cfg.setdefault("a0",   env_cfg.pop("outside_option", -1.0))

    return {"env": env_cfg, "agents": agent_cfg, "phase0": phase0_cfg}


# ---------------------------------------------------------------------------
# Lambda calibration
# ---------------------------------------------------------------------------

def calibrate_and_print_lambda(env_cfg: dict, env_name: str = "berlin_inner_ring") -> float:
    """Load demand_grid.parquet, call calibrate_lambda, print result."""
    from hotelling.spatial.assembly import calibrate_lambda
    import geopandas as gpd

    grid_path = Path(env_cfg.get("grid_path", "data/processed/demand_grid.parquet"))
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid not found: {grid_path}")

    grid = gpd.read_parquet(grid_path)

    # phi_i may be absent if assemble_simulation_grid has not been re-run.
    # Compute from constituents if needed (same logic as loader._compute_phi_i).
    if "phi_i" not in grid.columns:
        logger.info("phi_i absent; computing from constituent columns.")
        from hotelling.spatial.loader import _compute_phi_i  # type: ignore[import]
        phi_series = _compute_phi_i(grid)
        grid = grid.copy()
        grid["phi_i"] = phi_series.values

    lam = calibrate_lambda(grid, target_footfall_share=0.125)
    cfg_file = f"configs/env/{env_name}.yaml"
    print(f"\n{'='*60}")
    print(f"  Calibrated λ = {lam:.4f}")
    print(f"  (α=12.5%, Σω={grid['Einwohner'].sum():.0f}, Σφ={grid['phi_i'].sum():.4f})")
    print(f"  → Set 'lambda_val: {lam:.1f}' in {cfg_file}")
    print(f"{'='*60}\n")
    return lam


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def plot_convergence(result: dict, seed: int | None, output_dir: Path) -> None:
    """Plot price convergence and save to PNG (skips gracefully if matplotlib absent)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping convergence plot.")
        return

    steps  = result.get("step_history",  [])
    prices = result.get("price_history", [])
    p_nash = result.get("p_nash", None)
    p_mono = result.get("p_mono", None)

    if not steps:
        return

    seed_str = str(seed) if seed is not None else "noseed"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, prices, lw=1, color="steelblue", label="Mean market price")
    if p_nash is not None:
        ax.axhline(p_nash, ls="--", color="green",  lw=1.5, label=f"Nash  p={p_nash:.4f}")
    if p_mono is not None:
        ax.axhline(p_mono, ls="--", color="red",    lw=1.5, label=f"Mono  p={p_mono:.4f}")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Mean price (€)")
    ax.set_title(
        f"Berlin baseline — Phase 0 convergence (seed={seed_str})\n"
        f"Δ = {result.get('delta', float('nan')):.4f}  |  "
        f"converged={result.get('converged', False)}  |  "
        f"n_steps={result.get('n_steps', 0):,}"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "convergence.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Convergence plot saved → %s", plot_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(result: dict) -> None:
    """Print a formatted summary of simulation results to stdout."""
    print("\n" + "="*60)
    print("  BERLIN BASELINE — Phase 0 Q-learning burn-in")
    print("="*60)
    print(f"  Converged:          {result.get('converged', '?')}")
    print(f"  Steps completed:    {result.get('n_steps', 0):,}")
    print(f"  Elapsed:            {result.get('elapsed_s', 0):.1f} s")
    print()
    print(f"  Δ (Calvano index):  {result.get('delta', float('nan')):.4f}")
    print(f"    (Δ≈0 = competitive, Δ≈1 = monopoly, Δ>1 = super-monopoly)")
    print()
    print(f"  Mean final price:   {result.get('mean_final_price', float('nan')):.4f}")
    print(f"  Bertrand-Nash p:    {result.get('p_nash',           float('nan')):.4f}")
    print(f"  Joint-monopoly p:   {result.get('p_mono',           float('nan')):.4f}")
    if "epsilon_mean" in result:
        print(f"  Epsilon (mean):     {result['epsilon_mean']:.4f}")
    print()

    final_prices = result.get("final_prices", {})
    if final_prices:
        prices_arr = np.array(list(final_prices.values()))
        print(f"  Per-store prices:   min={prices_arr.min():.4f}  "
              f"mean={prices_arr.mean():.4f}  max={prices_arr.max():.4f}  "
              f"std={prices_arr.std():.4f}")
    print("="*60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Berlin Q-learning baseline simulation (inner-ring or full grid)."
    )
    parser.add_argument("--seed",           type=int,   default=None,
                        help="Random seed (overrides config)")
    parser.add_argument("--lambda-val",     type=float, default=None,
                        help="Override lambda_val in env config")
    parser.add_argument("--output-dir",     type=str,   default="results",
                        help="Directory for output files")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Print calibrated lambda and exit (no simulation)")
    parser.add_argument("--T-burnin",       type=int,   default=None,
                        help="Override T_burnin (e.g. 10000 for a quick test)")
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help=(
            "Load configs/env/berlin_full.yaml (full Berlin demand grid + full "
            "supermarket set) instead of berlin_inner_ring.yaml.  Requires the "
            "full-grid GEO pipeline outputs "
            "(demand_grid_full.parquet, supermarkets_full.parquet, "
            "travel_times_full.parquet).  Uses the sparse catchment CSR "
            "representation; Bertrand-Nash benchmarks are skipped until the "
            "Prompt-4 catchment kernels are implemented."
        ),
    )
    parser.add_argument(
        "--with-effort",
        action="store_true",
        help=(
            "Load configs/agents/qlearning_effort.yaml (m_effort=5, joint action "
            "space 75) instead of qlearning_baseline.yaml. Verify calibration with "
            "scripts/check_effort_calibration.py before using for results."
        ),
    )
    parser.add_argument(
        "--m-effort",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Override agents.m_effort in the loaded config (e.g. --m-effort 3). "
            "Applied after --with-effort; default is 1 (price-only Calvano baseline)."
        ),
    )
    args = parser.parse_args()

    # --- Load config ---
    # --full-grid selects berlin_full.yaml; otherwise inner-ring.
    # --with-effort selects the effort-activated agent config.
    _env_yaml = _ENV_YAMLS["full"] if args.full_grid else _ENV_YAMLS["inner_ring"]
    _env_name = "berlin_full" if args.full_grid else "berlin_inner_ring"
    _agents_yaml = (
        _REPO_ROOT / "configs" / "agents" / "qlearning_effort.yaml"
        if args.with_effort
        else _REPO_ROOT / "configs" / "agents" / "qlearning_baseline.yaml"
    )
    config = load_config(env_yaml=_env_yaml, agents_yaml=_agents_yaml)

    # --- Apply CLI overrides ---
    if args.seed is not None:
        config["phase0"]["seed"] = args.seed
    if args.lambda_val is not None:
        config["env"]["lambda_val"] = args.lambda_val
    if args.T_burnin is not None:
        config["phase0"]["T_burnin"] = args.T_burnin
    if args.m_effort is not None:
        config["agents"]["m_effort"] = args.m_effort
        logger.info("CLI override: agents.m_effort = %d", args.m_effort)

    output_dir = _REPO_ROOT / args.output_dir

    # --- Calibrate lambda ---
    if args.calibrate_only:
        calibrate_and_print_lambda(config["env"], env_name=_env_name)
        return

    if config["env"].get("lambda_val", 0) <= 0:
        logger.warning(
            "lambda_val is 0 or not set. Run with --calibrate-only first, "
            "then set lambda_val in %s.",
            _env_yaml,
        )

    # --- Optionally auto-calibrate lambda if placeholder value ---
    if float(config["env"].get("lambda_val", 0)) == 1500.0:
        logger.info(
            "lambda_val=1500.0 (placeholder). "
            "Computing calibrated value automatically …"
        )
        try:
            lam = calibrate_and_print_lambda(config["env"], env_name=_env_name)
            config["env"]["lambda_val"] = lam
        except Exception as exc:
            logger.warning("Auto-calibration failed: %s. Using placeholder λ=1500.", exc)

    # --- Run simulation ---
    from hotelling.simulation.runner import run_single_session

    seed = config["phase0"].get("seed", None)
    logger.info(
        "Starting Berlin baseline run: env=%s, seed=%s, T_burnin=%d, N_stores=auto.",
        _env_name, seed, int(config["phase0"].get("T_burnin", 1_000_000)),
    )

    # Add output_dir to config so runner knows where to write
    config["output_dir"] = str(_REPO_ROOT / args.output_dir / "runs")

    result = run_single_session(config)

    # --- Output ---
    print_summary(result)

    # Plot into the run's own folder (not a separate flat folder)
    run_out = Path(result.get("output_dir", str(_REPO_ROOT / args.output_dir)))
    plot_convergence(result, seed=seed, output_dir=run_out)

    print(f"  Run folder: {result.get('output_dir', '?')}")
    print(f"  Index:      {_REPO_ROOT / args.output_dir / 'index.csv'}")
    print()

    # --- Sanity check hint ---
    delta = result.get("delta", float("nan"))
    if 0.6 <= delta <= 0.95:
        print("  ✓ Δ is in the Calvano (2020) expected range [0.7, 0.85].")
    elif delta < 0.3:
        print("  ✗ Δ is very low. Check: transport_cost, price grid range,")
        print("    and whether Bertrand-Nash benchmark converged.")
    elif delta > 1.1:
        print("  ✗ Δ > 1. Prices above monopoly level; check price grid max.")
    print()


if __name__ == "__main__":
    main()
