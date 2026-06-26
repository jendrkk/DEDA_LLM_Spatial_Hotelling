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
    qi = result.get("qtable_init", "zero")
    if qi != "zero":
        print(f"  Q-table init:       {qi}")
    print(f"  Elapsed:            {result.get('elapsed_s', 0):.1f} s")
    print()
    # ── Calvano Δ table: price (analogue) + profit (canonical) ──────────────
    _dp  = result.get("deltas_by_chain") or {}
    _dpi = result.get("deltas_profit_by_chain") or {}

    def _fv(v) -> str:
        """Format a delta value; 'n/a' when None or NaN."""
        return f"{v:.4f}" if (v is not None and v == v) else "  n/a"

    print("  Calvano Δ  (Δ≈0 = Nash, Δ≈1 = monopoly; profit Δ is canonical)")
    print(f"  {'':13s}{'global':>8}  {'discount':>8}  {'standard':>8}  {'bio':>7}")
    print(
        f"  {'Δ price':13s}"
        f"{_fv(_dp.get('global')):>8}  "
        f"{_fv(_dp.get('discount')):>8}  "
        f"{_fv(_dp.get('standard')):>8}  "
        f"{_fv(_dp.get('bio')):>7}"
        f"   ← price analogue"
    )
    print(
        f"  {'Δ profit':13s}"
        f"{_fv(_dpi.get('global')):>8}  "
        f"{_fv(_dpi.get('discount')):>8}  "
        f"{_fv(_dpi.get('standard')):>8}  "
        f"{_fv(_dpi.get('bio')):>7}"
        f"   ← canonical (Calvano 2020 eq. 9)"
    )
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
    import math as _math
    if (result.get("beta_schedule") == "two_stage"
            and result.get("beta1") is not None):
        _b1  = result["beta1"]
        _b2  = result["beta2"]
        _t0s = result.get("t0_schedule", 0)
        _etr = result.get("epsilon_transition", 0.10)
        _emn = result.get("epsilon_min", 3e-4)
        _mea = result.get("mean_epsilon_approx", float("nan"))
        print(f"  ε schedule:         two-stage")
        print(f"    β₁ = {_b1:.2e}  β₂ = {_b2:.2e}  (β₂/β₁ = {_b2/_b1:.1f}×)")
        print(f"    t₀ = {_t0s:,}  ε(t₀) = {_etr:.3f}  ε_min = {_emn:.1e}  "
              f"mean(ε) ≈ {_mea:.3f}")
    else:
        beta = result.get("beta_decay")
        if beta is not None:
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
        "--graph-states",
        nargs=5,
        default=None,
        metavar=("K", "M", "B", "GRID", "MATCH"),
        help=(
            "Reciprocal rival-graph state: (own_price, rival_1_bin, ..., rival_K_bin) where "
            "the K rivals each store observes are chosen by max-weight diversion-ratio "
            "b-matching on an UNDIRECTED graph (max degree K, so reciprocity is structural; "
            "isolated stores -> local monopolists). Five positionals: K (rivals/store int>0), "
            "M (action price bins int>0), B (rival price bins int>0), GRID in {CS,G} "
            "(chain-specific or global own action grid), MATCH in {SC,A} (same-chain-type or "
            "any-chain-type rivals). State size = M*B^K. All Q-learning params are taken from "
            "the graph_states_params section of qlearning_baseline.yaml (which overrides every "
            "other section); the 5 positionals override the structural knobs. Requires "
            "catchment_minutes + auto_price_grid + dense benchmarks. Writes an interactive "
            "rival_graph.html into the run folder. Mutually exclusive with the other "
            "state-mode flags and with --chs-grid."
        ),
    )
    parser.add_argument(
        "--hybrid-states",
        action="store_true",
        default=False,
        help=(
            "Hybrid profit-gap state: (own_price_bin, prev_profit_quantile_bin, "
            "gap1_bin, gap2_bin) where gap = (p_own − p_rival) / p_rival for the "
            "two nearest same-chain-type rivals. "
            "State size = m × n_profit × n_gap² = 18×5×81 = 7290 (defaults). "
            "Parameters are set via hybrid_n_profit / hybrid_n_gap / hybrid_gap_lo / "
            "hybrid_gap_hi in qlearning_baseline.yaml. "
            "Mutually exclusive with --local-sum, --local-sum-d, --base-states, "
            "--full-states, --calvano-states, --strategic-states."
        ),
    )
    parser.add_argument(
        "--no-auto-beta",
        action="store_true",
        help=(
            "Disable all automatic β adaptation. Uses the config file's beta_decay "
            "value directly as a single-stage exponential with no ε_min floor "
            "(pure Calvano backward-compatibility mode)."
        ),
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        choices=["two_stage", "single"],
        default=None,
        metavar="MODE",
        help=(
            "Exploration decay schedule when beta_decay_auto=true (i.e. without "
            "--no-auto-beta). 'two_stage' (default): slow β₁ in stage 1 (t ≤ t₀) "
            "and rapid β₂ collapse in stage 2 (t > t₀), with t₀ = explore_fraction "
            "× T_burnin. 'single': legacy single-exponential auto-adapted to T_burnin "
            "(compute_beta_decay). Edit explore_fraction / epsilon_transition / "
            "epsilon_min in qlearning_baseline.yaml to tune the two-stage shape."
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
    parser.add_argument(
        "--qtable-init",
        type=str,
        choices=["zero", "nash-anchor", "solve", "optimistic"],
        default=None,
        metavar="MODE",
        help=(
            "Q-table initialization strategy. "
            "'zero' = all zeros (current default). "
            "'nash-anchor' = π_i(a_i; p^N_{-i})/(1−δ): hold others at Nash, "
            "sweep own action. "
            "'solve' = Calvano eq.(8): average over nearest neighbor's actions. "
            "'optimistic' = π_i^mono/(1−δ) for all (s,a). "
            "Requires benchmark computation (dense_distances=True). "
            "Default (omitted) = zero."
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
    # store_effort: True only when --with-effort is active AND --lean is not set.
    # Price-only runs (the default) never need effort_idx.npy or effort_grid.npy.
    config["phase0"]["store_effort"] = bool(args.with_effort) and not bool(args.lean)
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
        bool(args.hybrid_states),
        args.graph_states is not None,
    ]
    if sum(_state_flags) > 1:
        parser.error(
            "At most one state-mode flag is allowed: "
            "--local-sum, --local-sum-d, --base-states, "
            "--full-states, --calvano-states, --strategic-states, --hybrid-states, "
            "--graph-states."
        )
    if args.graph_states is not None and args.chs_grid:
        parser.error(
            "--graph-states already selects its own grid via the GRID positional "
            "({CS,G}); do not also pass --chs-grid."
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
    if args.hybrid_states:
        config["agents"]["state_mode"] = "hybrid_profit_gap"
        _m   = int(config["agents"].get("m", 18))
        _np  = int(config["agents"].get("hybrid_n_profit", 5))
        _ng  = int(config["agents"].get("hybrid_n_gap", 9))
        _glo = float(config["agents"].get("hybrid_gap_lo", -0.20))
        _ghi = float(config["agents"].get("hybrid_gap_hi",  0.20))
        logger.info(
            "state_mode=hybrid_profit_gap | m=%d × n_profit=%d × n_gap=%d² = %d states "
            "| gap ∈ [%.2f, %.2f]",
            _m, _np, _ng, _m * _np * _ng * _ng, _glo, _ghi,
        )
    if args.graph_states is not None:
        _gs = args.graph_states
        try:
            _gk, _gm, _gb = int(_gs[0]), int(_gs[1]), int(_gs[2])
        except (ValueError, TypeError):
            parser.error(f"--graph-states: K, M, B must be integers (got {_gs[:3]!r}).")
        _ggrid = str(_gs[3]).upper()
        _gmatch = str(_gs[4]).upper()
        if _gk <= 0 or _gm <= 0 or _gb <= 0:
            parser.error("--graph-states: K, M, B must all be > 0.")
        if _ggrid not in ("CS", "G"):
            parser.error(f"--graph-states GRID must be 'CS' or 'G' (got {_gs[3]!r}).")
        if _gmatch not in ("SC", "A"):
            parser.error(f"--graph-states MATCH must be 'SC' or 'A' (got {_gs[4]!r}).")
        # graph_states_params overrides ALL other Q-learning params; the 5 CLI
        # positionals then override the structural knobs.
        _gsp = dict(config["agents"].get("graph_states_params", {}) or {})
        config["agents"].update(_gsp)
        config["agents"]["state_mode"] = "graph_states"
        config["agents"]["graph_k"] = _gk
        config["agents"]["m"] = _gm
        config["agents"]["graph_n_rival_bins"] = _gb
        config["agents"]["graph_own_grid_type"] = _ggrid
        config["agents"]["chain_specific_grid"] = (_ggrid == "CS")
        config["agents"]["graph_rival_match"] = _gmatch
        config["agents"]["m_effort"] = 1
        logger.info(
            "state_mode=graph_states | k=%d, m=%d, B=%d, grid=%s, match=%s | "
            "state_size=%d | Q-learning params from graph_states_params (override).",
            _gk, _gm, _gb, _ggrid, _gmatch, _gm * _gb ** _gk,
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
    #   --hybrid-states        → state_mode=hybrid_profit_gap (7290 states)
    #   --graph-states K M B GRID MATCH → state_mode=graph_states
    # Beta:   --no-auto-beta (optional, composes with any state mode)
    # Effort: --with-effort (optional, composes with any state mode)
    # ──────────────────────────────────────────────────────────────────────────
    if args.no_auto_beta:
        config["agents"]["beta_decay_auto"] = False
        logger.info(
            "--no-auto-beta: using config beta_decay=%.2e directly "
            "(single-stage, no ε_min floor).",
            float(config["agents"].get("beta_decay", 4e-6)),
        )
    if args.beta_schedule is not None:
        config["agents"]["beta_schedule"] = args.beta_schedule
        logger.info("--beta-schedule: %s", args.beta_schedule)
    if args.chs_grid:
        config["agents"]["chain_specific_grid"] = True
        logger.info("--chs-grid: chain-type-specific price grids enabled.")
    if args.qtable_init is not None:
        config["qtable_init"] = args.qtable_init.replace("-", "_")
        logger.info("--qtable-init: %s", args.qtable_init)

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
    # Sanity check uses the canonical profit Δ; falls back to price Δ / legacy
    # `delta` key if profit Δ is unavailable (e.g. dense_distances=False run).
    _dpi_global = (
        (result.get("deltas_profit_by_chain") or {}).get("global")
        or result.get("delta", float("nan"))
    )
    _delta_check = _dpi_global if (_dpi_global is not None and _dpi_global == _dpi_global) else float("nan")
    if 0.6 <= _delta_check <= 0.95:
        print("  ✓ Δ profit is in the Calvano (2020) expected range [0.7, 0.85].")
    elif _delta_check < 0.3:
        print("  ✗ Δ profit is very low. Check: transport_cost, price grid range,")
        print("    and whether Bertrand-Nash benchmark converged.")
    elif _delta_check > 1.1:
        print("  ✗ Δ profit > 1. Supra-monopoly profits; check price grid max.")
    print()


if __name__ == "__main__":
    main()
