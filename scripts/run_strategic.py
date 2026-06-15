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
    ap.add_argument("--save-LLM-con", dest="save_llm_con", action="store_true",
                    help="dev tool: save each CEO LLM prompt + full response (incl. "
                         "reasoning if returned) to results/.../LLM_communication/"
                         "[chain]_[epoch].txt")
    ap.add_argument("--output-dir", type=str, default="results/strategic_runs")
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
    if args.save_llm_con:
        ceo_cfg["save_communication"] = True

    config = {
        "env": env_cfg, "agents": agent_cfg, "groups": groups_cfg, "ceo": ceo_cfg,
        "phase2": phase2_cfg, "output_dir": str(_REPO_ROOT / args.output_dir),
        "env_config_path": str(env_yaml),
    }

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
        if args.with_effort or args.local_sum_d or args.m_effort is not None:
            logger.warning("--from-run overrides --with-effort/--local-sum-d/--m-effort "
                           "with the baseline run's agent config (Q-table compatibility).")

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
    d = result.get("deltas_by_chain", {})
    print(f"  Δ global:           {d.get('global')}")
    print(f"  Δ D/S/B:            {d.get('discount')} / {d.get('standard')} / {d.get('bio')}")
    print(f"  Run folder:         {result.get('output_dir')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
