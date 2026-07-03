#!/usr/bin/env python
"""CEO-only strategic game: Phase-0 burn-in then LLM-CEO Phase 2.

Usage
-----
    conda activate py314
    export GEMINI_API_KEY=...        # Google AI Studio key (for the CEO calls)

    # Quick mechanics check WITHOUT any API calls (matched control):
    python scripts/run_strategic.py --T-burnin 5000 --T-game 1000 --no-ceo

    # CEO run, no groups (single envelope per chain):
    python scripts/run_strategic.py --T-burnin 200000 --T-game 5000 --T-CEO 100 --groups no_groups

    # With one division, richer local state:
    python scripts/run_strategic.py --groups competition_only --local-sum-d
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_strategic")


def _load(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description="CEO-only strategic game (Phase 0 -> Phase 2).")
    ap.add_argument("--env-config", type=str,
                    default="configs/env/berlin_inner_ring_calibrated.yaml")
    ap.add_argument("--groups", type=str, default="no_groups",
                    help="config name in configs/groups/ (no_groups, competition_only, ...)")
    ap.add_argument("--T-burnin", type=int, default=None)
    ap.add_argument("--T-game", type=int, default=None)
    ap.add_argument("--T-CEO", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--with-effort", action="store_true",
                    help="load qlearning_effort.yaml (m_effort=5) instead of baseline")
    ap.add_argument("--with-comm", action="store_true",
                    help="enable the CEO cheap-talk coordination signal (and its "
                         "memory in the prompt)")
    ap.add_argument("--tier-commit", dest="tier_commit", action="store_true",
                    help="Facilitating device: when all same-type chains mutually "
                         "signal willingness at a tier price >= the current level, "
                         "bind each same-type store's price FLOOR to that tier "
                         "(enforced coordinated ratchet). Requires --with-comm.")
    ap.add_argument("--temp", type=float, default=None,
                    help="CEO LLM sampling temperature; Google AI Studio accepts "
                         "floats in [0.0, 2.0] (default 0 = deterministic). 0.3–0.7 "
                         "gives useful CEO exploration; >1.2 risks malformed JSON.")
    ap.add_argument("--T-measure", type=int, default=None,
                    help="length (in periods) of the post-settling window the CEO "
                         "observes each epoch; defaults to T_CEO. Set << T_CEO so the "
                         "CEO reads the steady state, not the transient after a change.")
    ap.add_argument("--m-effort", type=int, default=None)
    ap.add_argument("--local-sum-d", action="store_true",
                    help="use detailed local-summary Q-state (total + same-type)")
    ap.add_argument("--model", type=str, default=None, help="override CEO LLM model string")
    ap.add_argument("--no-ceo", action="store_true", help="matched control: no CEO calls")
    ap.add_argument("--from-run", type=str, default=None,
                    help="load a converged Q-table + matching env/agent config from a "
                         "run_baseline run directory (skips Phase-0 burn-in)")
    ap.add_argument("--group-analytics", action="store_true",
                    help="enrich the CEO prompt with per-group competitive analytics (2.0)")
    ap.add_argument("--strategic-analytics", dest="strategic_analytics",
                    action="store_true",
                    help="Enrich the CEO prompt with per-type Nash/joint-monopoly "
                         "headroom and a coordinated tier-step profit counterfactual "
                         "(analytic; oracle-ish, gate for a clean control arm).")
    ap.add_argument("--save-LLM-con", dest="save_llm_con", action="store_true",
                    help="dev tool: save each CEO LLM prompt + full response (incl. "
                         "reasoning if returned) to results/.../LLM_communication/"
                         "[chain]_[epoch].txt")
    ap.add_argument("--output-dir", type=str, default="results/strategic_runs")
    ap.add_argument(
        "--graph-states", nargs=5, default=None,
        metavar=("K", "M", "B", "GRID", "MATCH"),
        help="Reciprocal rival-graph Q-state, identical to run_baseline --graph-states: "
             "K rivals/store (undirected max-weight diversion-ratio b-matching), M own price "
             "bins, B rival price bins, GRID in {CS,G} (chain-specific or global own grid), "
             "MATCH in {SC,A} (same-chain-type or any-chain-type rivals). state_size = M*B^K. "
             "All Q-learning params come from graph_states_params in qlearning_baseline.yaml; "
             "the 5 positionals override the structural knobs. Forces m_effort=1. Mutually "
             "exclusive with --local-sum-d, --chs-grid, and --with-effort.",
    )
    ap.add_argument(
        "--qtable-init", type=str, default=None,
        choices=["zero", "nash-anchor", "solve", "optimistic"], metavar="MODE",
        help="Q-table initialization for a FRESH burn-in (ignored with --from-run, which loads "
             "a converged table). 'zero' (default), 'nash-anchor', 'solve', 'optimistic'. "
             "Non-zero modes require dense benchmarks.",
    )
    ap.add_argument(
        "--beta-schedule", type=str, default=None, choices=["two_stage", "single"],
        metavar="MODE",
        help="Exploration (epsilon) decay schedule when beta_decay_auto is true. 'two_stage' "
             "(default): slow stage 1 then rapid stage-2 collapse; 'single': legacy "
             "single-exponential. Edit explore_fraction / epsilon_transition / epsilon_min in "
             "the agent config (or graph_states_params) to tune the shape.",
    )
    ap.add_argument(
        "--no-auto-beta", action="store_true",
        help="Disable automatic beta adaptation; use the config beta_decay directly as a "
             "single-stage exponential with no epsilon_min floor.",
    )
    ap.add_argument(
        "--chs-grid", action="store_true",
        help="Chain-type-specific price grids (composes with non-graph state modes). "
             "--graph-states selects its own grid via the GRID positional, so do not combine "
             "the two.",
    )
    ap.add_argument("--lean", action="store_true",
                    help="Dense log stores only price/effort indices + steps + grids "
                         "(skips demand & profit arrays, which are recomputed post-hoc by "
                         "visualize_run / RunBundle). Lossless for analysis; ~89%% smaller "
                         "dense log. Recommended for long production runs.")
    ap.add_argument("--dense-stride", type=int, default=None, metavar="INT",
                    help="Record only every INT-th step in the dense log (default from "
                         "phase2.yaml = 1). E.g. --dense-stride 1000 on a 9M-step run.")
    ap.add_argument("--dense-tail", type=int, default=None, metavar="INT",
                    help="Always densely record the last INT steps regardless of stride "
                         "(captures the converged collusive regime at full resolution), "
                         "e.g. --dense-tail 200000.")
    args = ap.parse_args()

    env_yaml = Path(args.env_config)
    if not env_yaml.is_absolute():
        env_yaml = _REPO_ROOT / env_yaml
    env_cfg = _load(env_yaml)
    env_cfg.setdefault("mu", env_cfg.pop("logit_scale", 0.25))
    env_cfg.setdefault("a0", env_cfg.pop("outside_option", -1.0))

    agents_yaml = (_REPO_ROOT / "configs" / "agents" /
                   ("qlearning_effort.yaml" if args.with_effort else "qlearning_baseline.yaml"))
    agent_cfg = _load(agents_yaml)
    groups_cfg = _load(_REPO_ROOT / "configs" / "groups" / f"{args.groups}.yaml")
    ceo_cfg = _load(_REPO_ROOT / "configs" / "agents" / "chain_ceo.yaml")
    phase2_cfg = _load(_REPO_ROOT / "configs" / "simulation" / "phase2.yaml")

    if args.m_effort is not None:
        agent_cfg["m_effort"] = args.m_effort
    if args.local_sum_d:
        agent_cfg["state_mode"] = "local_summary"
        agent_cfg["local_summary_detailed"] = True
        agent_cfg["local_sum_n"] = None
    if args.model is not None:
        ceo_cfg["model"] = args.model

    # ── State-mode / schedule / grid CLI (parity with run_baseline) ─────────────
    if args.graph_states is not None:
        if args.local_sum_d:
            ap.error("--graph-states and --local-sum-d are mutually exclusive.")
        if args.chs_grid:
            ap.error("--graph-states already selects its grid via the GRID positional "
                     "({CS,G}); do not also pass --chs-grid.")
        if args.with_effort:
            ap.error("--graph-states forces m_effort=1 (price-only Calvano baseline); "
                     "it is incompatible with --with-effort.")
        _gs = args.graph_states
        try:
            _gk, _gm, _gb = int(_gs[0]), int(_gs[1]), int(_gs[2])
        except (ValueError, TypeError):
            ap.error(f"--graph-states: K, M, B must be integers (got {_gs[:3]!r}).")
        _ggrid = str(_gs[3]).upper()
        _gmatch = str(_gs[4]).upper()
        if _gk <= 0 or _gm <= 0 or _gb <= 0:
            ap.error("--graph-states: K, M, B must all be > 0.")
        if _ggrid not in ("CS", "G"):
            ap.error(f"--graph-states GRID must be 'CS' or 'G' (got {_gs[3]!r}).")
        if _gmatch not in ("SC", "A"):
            ap.error(f"--graph-states MATCH must be 'SC' or 'A' (got {_gs[4]!r}).")
        # graph_states_params overrides ALL other Q-learning params; the 5 CLI
        # positionals then override the structural knobs (identical to run_baseline).
        _gsp = dict(agent_cfg.get("graph_states_params", {}) or {})
        agent_cfg.update(_gsp)
        agent_cfg["state_mode"] = "graph_states"
        agent_cfg["graph_k"] = _gk
        agent_cfg["m"] = _gm
        agent_cfg["graph_n_rival_bins"] = _gb
        agent_cfg["graph_own_grid_type"] = _ggrid
        agent_cfg["chain_specific_grid"] = (_ggrid == "CS")
        agent_cfg["graph_rival_match"] = _gmatch
        agent_cfg["m_effort"] = 1
        logger.info(
            "state_mode=graph_states | k=%d, m=%d, B=%d, grid=%s, match=%s | "
            "state_size=%d | Q-learning params from graph_states_params (override).",
            _gk, _gm, _gb, _ggrid, _gmatch, _gm * _gb ** _gk,
        )
    if args.no_auto_beta:
        agent_cfg["beta_decay_auto"] = False
        logger.info("--no-auto-beta: single-stage beta_decay, no epsilon_min floor.")
    if args.beta_schedule is not None:
        agent_cfg["beta_schedule"] = args.beta_schedule
        logger.info("--beta-schedule: %s", args.beta_schedule)
    if args.chs_grid:
        agent_cfg["chain_specific_grid"] = True
        logger.info("--chs-grid: chain-type-specific price grids enabled.")

    if args.T_burnin is not None:
        phase2_cfg["T_burnin"] = args.T_burnin
    if args.T_game is not None:
        phase2_cfg["T_game"] = args.T_game
    if args.T_CEO is not None:
        phase2_cfg["T_CEO"] = args.T_CEO
    if args.seed is not None:
        phase2_cfg["seed"] = args.seed
    if args.no_ceo:
        phase2_cfg["no_ceo"] = True
    if args.group_analytics:
        ceo_cfg["group_analytics"] = True
    if args.strategic_analytics:
        ceo_cfg["strategic_analytics"] = True
    if args.tier_commit:
        if not args.with_comm:
            ap.error("--tier-commit requires --with-comm (it acts on the "
                     "coordination signal).")
        ceo_cfg["tier_commit"] = True
    if args.save_llm_con:
        ceo_cfg["save_communication"] = True
    if args.temp is not None:
        t = float(args.temp)
        if not (0.0 <= t <= 2.0):
            ap.error(f"--temp {t} out of range; Google AI Studio accepts [0.0, 2.0].")
        if t > 1.2:
            logger.warning("--temp %.2f is high; the CEO may emit malformed JSON "
                           "(falls back to retained envelope on parse failure).", t)
        ceo_cfg["temperature"] = t
    if args.T_measure is not None:
        phase2_cfg["T_measure"] = args.T_measure

    # ── Dense-log size controls (parity with run_baseline.py) ───────────────────
    if args.lean:
        phase2_cfg["store_demand_profit"] = False
        logger.info("--lean: store_demand_profit=False (demand/profit arrays not "
                    "written; reconstructed on demand by visualize_run / RunBundle).")
    # store_effort: only meaningful with effort active AND not lean. Mirrors
    # run_baseline: a lean price-only run writes no effort arrays. run_strategic_session
    # already gates store_effort on store_demand_profit, but set it explicitly so the
    # written phase2 config records the intent.
    phase2_cfg["store_effort"] = bool(args.with_effort) and not bool(args.lean)
    if args.dense_stride is not None:
        phase2_cfg["dense_stride"] = args.dense_stride
        logger.info("dense_stride override: %d", args.dense_stride)
    if args.dense_tail is not None:
        phase2_cfg["dense_tail"] = args.dense_tail
        logger.info("dense_tail override: %d", args.dense_tail)

    config = {
        "env": env_cfg, "agents": agent_cfg, "groups": groups_cfg, "ceo": ceo_cfg,
        "phase2": phase2_cfg, "output_dir": str(_REPO_ROOT / args.output_dir),
        "env_config_path": str(env_yaml),
        "with_effort": bool(args.with_effort),
        "with_comm": bool(args.with_comm),
    }
    config["qtable_init"] = (
        args.qtable_init.replace("-", "_") if args.qtable_init is not None else "zero"
    )

    if float(config["env"].get("lambda_val", 0)) <= 0:
        logger.warning("lambda_val <= 0; run scripts/run_baseline.py --calibrate-only first.")

    if args.from_run is not None:
        from_dir = Path(args.from_run)
        if not from_dir.is_absolute():
            from_dir = _REPO_ROOT / from_dir
        base_cfg = _load(from_dir / "config.yaml")
        # The loaded Q-table is indexed by the baseline run's price grid + state
        # encoding; reuse its env + agents blocks verbatim to guarantee a match.
        config["env"] = base_cfg.get("env", config["env"])
        config["agents"] = base_cfg.get("agents", config["agents"])
        config["from_run"] = str(from_dir)
        loaded_m_effort = int(config["agents"].get("m_effort", 1))
        if args.with_effort and loaded_m_effort <= 1:
            ap.error(
                f"--with-effort requires a --from-run baseline trained WITH effort "
                f"(m_effort>1); the loaded run has m_effort={loaded_m_effort} "
                f"(price-only). Omit --from-run to run a fresh effort burn-in, or "
                f"point --from-run at an effort-trained baseline."
            )
        if not args.with_effort and loaded_m_effort > 1:
            logger.warning("--from-run baseline was trained WITH effort "
                           "(m_effort=%d) but --with-effort is OFF; effort will be "
                           "frozen at index 0 for the strategic game.", loaded_m_effort)
        if args.local_sum_d or args.m_effort is not None:
            logger.warning("--from-run overrides --local-sum-d/--m-effort with the "
                           "baseline run's agent config (Q-table compatibility).")

    from hotelling.simulation.runner import run_strategic_session

    logger.info("Strategic run: groups=%s, no_ceo=%s, model=%s",
                args.groups, config["phase2"].get("no_ceo"), config["ceo"].get("model"))
    result = run_strategic_session(config)

    print("\n" + "=" * 60)
    print("  STRATEGIC GAME — Phase 2 (CEO-only)")
    print("=" * 60)
    print(f"  no_ceo (control):   {config['phase2'].get('no_ceo')}")
    print(f"  CEO epochs:         {result.get('n_epochs')}")
    _sr = result.get("ceo_success_rate")
    _pct = 0.0 if _sr is None or _sr != _sr else 100.0 * _sr
    print(f"  CEO call success:   {result.get('ceo_calls_success')}/"
          f"{result.get('ceo_calls_total')} "
          f"({_pct:.0f}%)")
    if result.get("ceo_all_failed"):
        print("  *** WARNING: ALL CEO CALLS FAILED — Δ BELOW IS INVALID ***")
        print("  *** Inspect <run>/llm_calls.jsonl for FAILED records.       ***")
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
    print(f"  Run folder:         {result.get('output_dir')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
