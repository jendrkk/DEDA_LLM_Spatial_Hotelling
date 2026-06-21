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

    # Run against the calibrated config:
    python scripts/run_baseline.py \\
        --env-config configs/env/berlin_inner_ring_calibrated.yaml \\
        --T-burnin 1000000 --seed 42

    # Long run, minimal disk (price animation only):
    python scripts/run_baseline.py --env-config configs/env/berlin_inner_ring_calibrated.yaml \\
        --T-burnin 2000000 --lean --seed 42

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

    deltas_by_chain = result.get("deltas_by_chain")
    if deltas_by_chain:
        parts = []
        for key in ("global", "discount", "standard", "bio"):
            val = deltas_by_chain.get(key)
            if val is not None and val == val:  # skip NaN
                parts.append(f"{key}={val:.4f}")
        if parts:
            print("  Chain-specific Calvano Δ:")
            print(f"    {' / '.join(parts)}")
            print()

    chain_price_table = result.get("chain_price_table")
    if chain_price_table:
        print("  Per-chain prices (n | learned | Nash | mono)")
        print(f"  {'chain':<10} {'n':>5} {'learned':>10} {'Nash':>10} {'mono':>10}")
        for ct in ("global", "discount", "standard", "bio"):
            row = chain_price_table.get(ct)
            if row:
                print(
                    f"  {ct:<10} {row['n']:5d} "
                    f"{row['learned']:10.4f} {row['nash']:10.4f} {row['mono']:10.4f}"
                )
        print()

    if result.get("realized_outside_share") is not None:
        ros = result.get("realized_outside_share")
        rcs = result.get("realized_chain_shares") or {}
        if ros == ros or rcs:  # has outside share or chain shares
            print("  Realized Bertrand-Nash moments (calibration check)")
            if ros == ros:
                print(f"    outside share = {ros:.4f}  (calibration target ~0.04)")
            if rcs:
                print(
                    f"    chain shares  discount / standard / bio = "
                    f"{rcs.get('discount', float('nan')):.4f} / "
                    f"{rcs.get('standard', float('nan')):.4f} / "
                    f"{rcs.get('bio', float('nan')):.4f}"
                )
            print()
    print(f"  Mean final price:   {result.get('mean_final_price', float('nan')):.4f}")
    print(f"  Bertrand-Nash p:    {result.get('p_nash',           float('nan')):.4f}")
    print(f"  Joint-monopoly p:   {result.get('p_mono',           float('nan')):.4f}")
    if "epsilon_mean" in result:
        print(f"  Epsilon (mean):     {result['epsilon_mean']:.4f}")
    beta = result.get("beta_decay")
    if beta is not None:
        import math as _math
        T = result.get("n_steps", 0)
        eps_final = _math.exp(-beta * T) if T > 0 else 1.0
        print(f"  β (decay rate):     {beta:.2e}  (ε at T={T:,}: {eps_final:.6f})")
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
        "--env-config",
        type=str,
        default=None,
        help=(
            "Path to an env YAML to use instead of the default inner-ring / "
            "--full-grid config (e.g. configs/env/berlin_inner_ring_calibrated.yaml). "
            "Overrides --full-grid when both are given."
        ),
    )
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
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=None,
        metavar="INT",
        help="Override agents.k_neighbors (default from config: 1).",
    )
    parser.add_argument("--lean", action="store_true",
                        help="Save only essential dense-log data: price/effort indices, steps, "
                             "grids, and aggregate.parquet. Skips the demand & profit arrays "
                             "(recomputable post-hoc via market_clearing on the stored indices), "
                             "halving disk and avoiding the >5GB DenseLog warning on long runs. "
                             "This is exactly what 06_spatial_animations needs (it colours stores "
                             "by price).")
    parser.add_argument("--dense-stride", type=int, default=None, metavar="INT",
                        help="Record only every INT-th step in the dense log (default 1). "
                             "Use for very long runs, e.g. --dense-stride 20 on a 20M-step run.")
    parser.add_argument("--dense-tail", type=int, default=None, metavar="INT",
                        help="Always densely record the last INT steps regardless of stride "
                             "(captures the converged regime at full resolution).")
    parser.add_argument(
        "--local-sum",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="Use the local-market price-summary state instead of k-neighbors. "
             "Bare --local-sum = demand-overlap competitor set (default "
             "definition); --local-sum N = the N nearest stores. Omit to keep "
             "the k-neighbors state. Composes with --with-effort.",
    )
    parser.add_argument(
        "--local-sum-d",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="Detailed local-summary state: condition on TWO binned price "
             "summaries — the total local market AND the same-chain-type "
             "local market (state_size = n_price_bins^2, ~ like k=2). Bare = "
             "demand-overlap set; N = N nearest. Mutually exclusive with "
             "--local-sum.",
    )
    parser.add_argument(
        "--base-states",
        type=int,
        nargs="?",
        const=15,
        default=None,
        metavar="B",
        help=(
            "Design 4 state: (own_prev_price_bin, same-type_competitor_mean_bin). "
            "State size = m × B. Bare --base-states uses B=15 (default); "
            "--base-states 10 uses B=10 bins. Mutually exclusive with "
            "--local-sum, --local-sum-d, --full-states, --calvano-states, --strategic-states."
        ),
    )
    parser.add_argument(
        "--full-states",
        type=int,
        nargs="?",
        const=7,
        default=None,
        metavar="B",
        help=(
            "Design 5 state: (own_price, same-type_mean, cross-type_mean). "
            "State size = m × B × B. Default B=7 → state_size=735. "
            "Mutually exclusive with other state-mode flags."
        ),
    )
    parser.add_argument(
        "--calvano-states",
        type=int,
        choices=[1, 2, 3],
        default=None,
        metavar="K",
        help=(
            "Calvano local duopoly state: (own_price, rival_1_price, ..., rival_K_price). "
            "K same-chain-type nearest rivals. State size = m^(K+1). "
            "K=1 → 225, K=2 → 3375, K=3 → 50625. "
            "Mutually exclusive with other state-mode flags."
        ),
    )
    parser.add_argument(
        "--strategic-states",
        type=int,
        nargs="?",
        const=10,
        default=None,
        metavar="B",
        help=(
            "Strategic hybrid state: (own_price, same-type_comp_mean, market_regime). "
            "regime ∈ {competitive, neutral, supra-competitive} from all-type local mean "
            "vs Nash benchmark. State size = m × B × 3. Default B=10 → 450. "
            "Requires precomputed Nash prices (auto_price_grid=true). "
            "Mutually exclusive with other state-mode flags."
        ),
    )
    parser.add_argument(
        "--no-auto-beta",
        action="store_true",
        help=(
            "Disable automatic exploration decay (β) adaptation to T_burnin. "
            "Uses the config file's beta_decay value directly (default 4e-6 = Calvano). "
            "By default, β is automatically scaled for T > 1M steps."
        ),
    )
    parser.add_argument(
        "--chs-grid",
        action="store_true",
        help=(
            "Use chain-type-specific price grids instead of a single global grid. "
            "Each chain type (discount/standard/bio) gets its own linspace grid "
            "spanning [MC_τ, p_M_τ + ξ·span_τ] (MC≠0) or "
            "[max(0, p_N_τ − ξ·span_τ), p_M_τ + ξ·span_τ] (MC=0). "
            "Requires auto_price_grid=true and dense_distances=true."
        ),
    )
    args = parser.parse_args()

    # --- Load config ---
    # --env-config overrides --full-grid and the default inner-ring env.
    # --with-effort selects the effort-activated agent config.
    if args.env_config is not None:
        _env_yaml = Path(args.env_config)
        if not _env_yaml.is_absolute():
            _env_yaml = _REPO_ROOT / _env_yaml
        if not _env_yaml.exists():
            raise FileNotFoundError(f"--env-config not found: {_env_yaml}")
        _env_name = _env_yaml.stem
    elif args.full_grid:
        _env_yaml = _ENV_YAMLS["full"]
        _env_name = "berlin_full"
    else:
        _env_yaml = _ENV_YAMLS["inner_ring"]
        _env_name = "berlin_inner_ring"
    _agents_yaml = (
        _REPO_ROOT / "configs" / "agents" / "qlearning_effort.yaml"
        if args.with_effort
        else _REPO_ROOT / "configs" / "agents" / "qlearning_baseline.yaml"
    )
    config = load_config(env_yaml=_env_yaml, agents_yaml=_agents_yaml)
    config["env_config_path"] = str(_env_yaml)

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
    if args.k_neighbors is not None:
        config["agents"]["k_neighbors"] = args.k_neighbors
        logger.info("k_neighbors override: %d", args.k_neighbors)
    if args.lean:
        config["phase0"]["store_demand_profit"] = False
        logger.info("--lean: store_demand_profit=False "
                    "(demand/profit arrays not written; recomputable post-hoc).")
    if args.dense_stride is not None:
        config["phase0"]["dense_stride"] = args.dense_stride
        logger.info("dense_stride override: %d", args.dense_stride)
    if args.dense_tail is not None:
        config["phase0"]["dense_tail"] = args.dense_tail
        logger.info("dense_tail override: %d", args.dense_tail)
    _state_flags = [
        args.local_sum is not None,
        args.local_sum_d is not None,
        args.base_states is not None,
        args.full_states is not None,
        args.calvano_states is not None,
        args.strategic_states is not None,
    ]
    if sum(_state_flags) > 1:
        parser.error(
            "At most one state-mode flag is allowed: "
            "--local-sum, --local-sum-d, --base-states, "
            "--full-states, --calvano-states, --strategic-states."
        )
    if args.local_sum is not None and args.local_sum_d is not None:
        parser.error("--local-sum and --local-sum-d are mutually exclusive.")
    if args.local_sum_d is not None:
        config["agents"]["state_mode"] = "local_summary"
        config["agents"]["local_summary_detailed"] = True
        config["agents"]["local_sum_n"] = (
            None if args.local_sum_d == 0 else args.local_sum_d
        )
        logger.info(
            "state_mode=local_summary (DETAILED: total + same-type), "
            "local_sum_n=%s, n_price_bins=%s",
            config["agents"]["local_sum_n"],
            config["agents"].get("n_price_bins", 15),
        )
    elif args.local_sum is not None:
        config["agents"]["state_mode"] = "local_summary"
        config["agents"]["local_sum_n"] = (
            None if args.local_sum == 0 else args.local_sum
        )
        logger.info(
            "state_mode=local_summary, local_sum_n=%s, n_price_bins=%s, "
            "summary_stats=%s",
            config["agents"]["local_sum_n"],
            config["agents"].get("n_price_bins", 15),
            config["agents"].get("summary_stats", ["mean"]),
        )
    if args.base_states is not None:
        config["agents"]["state_mode"] = "design4_ownprice"
        config["agents"]["n_comp_bins"] = args.base_states
        logger.info(
            "state_mode=design4_ownprice (own_price + same-type competitor mean), "
            "n_comp_bins=%d, state_size=%d",
            args.base_states,
            int(config["agents"].get("m", 15)) * args.base_states,
        )
    if args.full_states is not None:
        config["agents"]["state_mode"] = "design5_full"
        config["agents"]["n_comp_bins"] = args.full_states
        logger.info(
            "state_mode=design5_full, B=%d, state_size=%d",
            args.full_states,
            int(config["agents"].get("m", 15)) * args.full_states ** 2,
        )
    if args.calvano_states is not None:
        config["agents"]["state_mode"] = "calvano_local"
        config["agents"]["calvano_k"] = args.calvano_states
        _m = int(config["agents"].get("m", 15))
        logger.info(
            "state_mode=calvano_local, k=%d, state_size=%d",
            args.calvano_states,
            _m ** (args.calvano_states + 1),
        )
    if args.strategic_states is not None:
        config["agents"]["state_mode"] = "strategic_hybrid"
        config["agents"]["n_comp_bins"] = args.strategic_states
        logger.info(
            "state_mode=strategic_hybrid, B=%d, state_size=%d",
            args.strategic_states,
            int(config["agents"].get("m", 15)) * args.strategic_states * 3,
        )
    # ── Valid CLI combinations ──────────────────────────────────────────────────
    # Grid:  --chs-grid (optional, composes with any state mode)
    # State: exactly one of:
    #   (none)                 → state_mode=neighbors (default)
    #   --local-sum [N]        → state_mode=local_summary (legacy)
    #   --local-sum-d [N]      → state_mode=local_summary detailed (legacy)
    #   --base-states [B]      → state_mode=design4_ownprice
    #   --full-states [B]      → state_mode=design5_full
    #   --calvano-states K     → state_mode=calvano_local
    #   --strategic-states [B] → state_mode=strategic_hybrid
    # Beta:   --no-auto-beta (optional, composes with any state mode)
    # Effort: --with-effort (optional, composes with any state mode)
    # ──────────────────────────────────────────────────────────────────────────
    if args.no_auto_beta:
        config["agents"]["beta_decay_auto"] = False
        logger.info(
            "--no-auto-beta: using config beta_decay=%.2e directly.",
            float(config["agents"].get("beta_decay", 4e-6)),
        )
    if args.chs_grid:
        config["agents"]["chain_specific_grid"] = True
        logger.info("--chs-grid: chain-type-specific price grids enabled.")

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

    logger.info(
        "Final lambda_val = %.4f (env config: %s)",
        float(config["env"]["lambda_val"]),
        _env_yaml,
    )

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
