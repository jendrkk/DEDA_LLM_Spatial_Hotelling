"""Hydra-driven batch and sweep runner.

Responsibility: provide functions to run a single simulation session from a
Hydra config dict, and to run a full parameter sweep in parallel using
multiprocessing.Pool.

Public API: run_single_session, run_sweep

Key dependencies: multiprocessing, pathlib, hotelling.simulation.engine,
    hydra-core (optional)

References:
    Hydra (Yadan 2019) https://hydra.cc/;
    Calvano et al. (2020 AER) §III - batch training protocol.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_beta_decay(
    T_burnin: int,
    *,
    epsilon_target: float = 0.02,
    f_explore: float = 0.75,
    beta_calvano: float = 4.0e-6,
    T_calvano_max: int = 1_000_000,
    beta_floor: float = 1.0e-7,
) -> float:
    """Compute exploration decay rate β adapted to the run length.

    For short runs (T ≤ T_calvano_max), returns the Calvano (2020) default.
    For longer runs, computes β so that ε(f_explore × T) = ε_target, ensuring
    meaningful exploration persists through 75% of the run.

    Formula:  β = −ln(ε_target) / (f_explore × T_burnin)

    Parameters
    ----------
    T_burnin : total simulation steps
    epsilon_target : ε value at end of exploration phase (default 0.02)
    f_explore : fraction of T_burnin for exploration (default 0.75)
    beta_calvano : Calvano et al. (2020) default β for T ≤ T_calvano_max
    T_calvano_max : threshold below which Calvano default is used
    beta_floor : minimum β to prevent near-zero decay on very long runs

    Returns
    -------
    float — exploration decay rate β
    """
    if T_burnin <= T_calvano_max:
        return beta_calvano

    import logging
    beta = -math.log(epsilon_target) / (f_explore * T_burnin)
    beta = max(beta, beta_floor)
    logging.getLogger(__name__).info(
        "Dynamic β: T_burnin=%d > %d → β=%.2e "
        "(ε at %.0f%%·T = %.4f, ε at T = %.6f). "
        "Calvano default would give ε(T) = %.2e.",
        T_burnin, T_calvano_max, beta,
        f_explore * 100,
        math.exp(-beta * f_explore * T_burnin),
        math.exp(-beta * T_burnin),
        math.exp(-beta_calvano * T_burnin),
    )
    return beta


def compute_two_stage_schedule(
    T_burnin: int,
    *,
    explore_fraction: float = 0.65,
    epsilon_transition: float = 0.10,
    epsilon_min: float = 3e-4,
    beta_floor: float = 1e-8,
) -> dict:
    """Two-stage exploration decay adaptive to T_burnin.

    Stage 1 (0 ≤ t ≤ t₀): ε(t) = max(ε_min, exp(−β₁·t))
    Stage 2 (t > t₀):      ε(t) = max(ε_min, ε_transition · exp(−β₂·(t−t₀)))

    Continuous at t₀: ε(t₀⁻) = ε(t₀⁺) = ε_transition (no jump).

    Calibration:
        t₀  = round(f₀ × T_burnin)
        β₁  = −ln(ε_transition) / t₀
        β₂  = ln(ε_transition / ε_min) / (T_burnin − t₀)
        β₂/β₁ = [ln(ε_transition/ε_min)·f₀] / [(−ln ε_transition)·(1−f₀)]
                (constant for given shape params, independent of T_burnin)

    Collusion rationale:
        Long stage 1 (f₀≈0.65) lets agents discover that high-price coordination
        raises Q-values.  The rapid stage-2 collapse (β₂/β₁≈4.7× at defaults)
        locks in the collusive attractor before deviation rediscovery destabilizes it.

    Mean exploration:
        mean(ε) ≈ f₀·g(ε_transition) + (1−f₀)·h(ε_transition, ε_min)
        where g(x) = (1−x)/(−ln x),  h(a,b) = (a−b)/ln(a/b)
        (≈ 0.26 at defaults; comparable to Calvano's single-stage ≈ 0.245)

    Parameters
    ----------
    T_burnin : total simulation steps
    explore_fraction : f₀ ∈ (0,1); t₀ = round(f₀ × T_burnin)
    epsilon_transition : ε(t₀) target at the stage switch [0.05, 0.20]
    epsilon_min : ε floor and ε(T) target [1e-4, 1e-3]
    beta_floor : hard lower bound on both β values (avoids numerical degeneracy)

    Returns
    -------
    dict — keys: beta1, beta2, t0, epsilon_min, epsilon_transition,
           beta_ratio, mean_epsilon_approx
    """
    t0 = max(1, round(explore_fraction * T_burnin))
    t_exploit = max(1, T_burnin - t0)

    beta1 = max(beta_floor, -math.log(epsilon_transition) / t0)
    beta2 = max(beta_floor, math.log(epsilon_transition / epsilon_min) / t_exploit)
    beta_ratio = beta2 / beta1

    g = (1.0 - epsilon_transition) / (-math.log(epsilon_transition))
    h = (epsilon_transition - epsilon_min) / math.log(epsilon_transition / epsilon_min)
    mean_eps = explore_fraction * g + (1.0 - explore_fraction) * h

    import logging as _log_sched
    _log_sched.getLogger(__name__).info(
        "Two-stage ε schedule: T=%d | t₀=%d (f₀=%.0f%%) | "
        "β₁=%.2e → ε(t₀)=%.3f | β₂=%.2e → ε(T)=%.2e | "
        "β₂/β₁=%.1f× | mean(ε)≈%.3f",
        T_burnin, t0, explore_fraction * 100,
        beta1, epsilon_transition,
        beta2, epsilon_min,
        beta_ratio, mean_eps,
    )

    return {
        "beta1": beta1,
        "beta2": beta2,
        "t0": t0,
        "epsilon_min": epsilon_min,
        "epsilon_transition": epsilon_transition,
        "beta_ratio": beta_ratio,
        "mean_epsilon_approx": mean_eps,
    }


def run_single_session(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single simulation session from a Hydra config dict.

    Builds the City, Firm list, agents, and SimulationEngine from the config,
    runs the engine, and returns session metrics.

    Parameters
    ----------
    config : Hydra config dict (usually OmegaConf DictConfig converted to dict)

    Returns
    -------
    dict with keys: run_id, seed, converged, n_steps, delta,
        mean_final_price, p_nash, p_mono, price_history,
        effort_history, step_history, final_prices, elapsed_s
    """
    import json
    import time
    import uuid
    from datetime import datetime

    import pandas as pd
    import yaml

    from hotelling.spatial.loader import load_berlin_city
    from hotelling.env.market_env import HotellingMarketEnv
    from hotelling.agents.qlearning import QLearningAgent
    from hotelling.simulation.phases import Phase0BurnIn
    from hotelling.simulation.recorder import SimulationRecorder
    from hotelling.core.equilibrium import bertrand_nash, joint_monopoly

    t_start = time.time()
    run_id = str(uuid.uuid4())[:8]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{run_id}"
    output_dir = Path(config.get("output_dir", "results/runs")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the config used for this run immediately (reproducibility)
    config_save_path = output_dir / "config.yaml"
    with config_save_path.open("w") as _f:
        yaml.dump(config, _f, default_flow_style=False, allow_unicode=True)

    env_cfg = config.get("env", {})
    agent_cfg = config.get("agents", {})
    phase0_cfg = config.get("phase0", {})
    seed = phase0_cfg.get("seed", None)

    # --- 1. Load Berlin City and Firms ---
    _catchment_minutes_raw = env_cfg.get("catchment_minutes", None)
    _catchment_minutes = (
        float(_catchment_minutes_raw) if _catchment_minutes_raw is not None else None
    )
    _dense_distances = bool(env_cfg.get("dense_distances", True))

    city, firms = load_berlin_city(
        grid_path=env_cfg.get("grid_path", "data/processed/demand_grid.parquet"),
        stores_path=env_cfg.get("stores_path", "data/processed/supermarkets.parquet"),
        travel_times_path=env_cfg.get(
            "travel_times_path", "data/processed/travel_times.parquet"
        ),
        lambda_val=float(env_cfg["lambda_val"]),
        q_S=float(env_cfg.get("q_S", 0.8)),
        q_B=float(env_cfg.get("q_B", 1.5)),
        alpha_L=float(env_cfg.get("alpha_L", 0.5)),
        alpha_H=float(env_cfg.get("alpha_H", 1.5)),
        beta_effort=float(env_cfg.get("beta_effort", 0.001)),
        kappa0=float(env_cfg.get("kappa0", 1.0)),
        store_size=float(env_cfg.get("store_size", 600.0)),
        transport_cost=float(env_cfg.get("transport_cost", 0.01)),
        a0=float(env_cfg.get("a0", -1.0)),
        mu=float(env_cfg.get("mu", 0.25)),
        nan_fill_minutes=float(env_cfg.get("nan_fill_minutes", 120.0)),
        marginal_cost_D=float(env_cfg.get("marginal_cost_D", 0.0)),
        marginal_cost_S=float(env_cfg.get("marginal_cost_S", 0.0)),
        marginal_cost_B=float(env_cfg.get("marginal_cost_B", 0.0)),
        rent_scale=float(env_cfg.get("rent_scale", 0.0)),
        rent_normalization=str(env_cfg.get("rent_normalization", "mean_ratio")),
        dense_distances=_dense_distances,
        catchment_minutes=_catchment_minutes,
        catchment_k_min=int(env_cfg.get("catchment_k_min", 12)),
        catchment_k_max=int(env_cfg.get("catchment_k_max", 80)),
        precompute_expweights=bool(env_cfg.get("precompute_expweights", False)),
        low_precision_storage=bool(env_cfg.get("low_precision_storage", False)),
    )

    # --- 1b. Pre-compute benchmarks and derive Calvano price grid ---
    tc = float(env_cfg.get("transport_cost", 0.01))
    benchmark_cache_pre = (
        Path(env_cfg.get("grid_path", "data/processed/demand_grid.parquet")).parent
        / "benchmarks_cache.npz"
    )
    auto_grid = bool(agent_cfg.get("auto_price_grid", True))
    p_nash_pre = p_mono_pre = None
    p_nash_arr = None
    p_mono_arr = None

    # Benchmarks require the dense (M×N) distance matrix.  On the sparse /
    # full-grid path (dense_distances=False) city.dist2_km2 is None; the
    # catchment-aware solvers from Prompt 4 will provide these benchmarks once
    # implemented.  Until then, fall back to manual price-grid bounds from
    # the agent config and log a clear warning.
    _can_run_benchmarks = city.dist2_km2 is not None
    grid_mode = str(agent_cfg.get("price_grid_mode", "union"))
    if auto_grid and _can_run_benchmarks:
        p_nash_arr, _ = bertrand_nash(city, transport_cost=tc, cache_path=benchmark_cache_pre)
        p_mono_arr, _ = joint_monopoly(city, transport_cost=tc, cache_path=benchmark_cache_pre)
        p_nash_pre = float(p_nash_arr.mean())
        p_mono_pre = float(p_mono_arr.mean())
        xi = float(agent_cfg.get("price_grid_xi", 0.1))
        mc_min = min(getattr(f, "marginal_cost", 0.0) for f in firms)
        if grid_mode == "union":
            nash_lo = float(p_nash_arr.min())
            mono_hi = float(p_mono_arr.max())
            uspan = mono_hi - nash_lo
            if uspan > 1e-6:
                grid_min = max(mc_min, nash_lo - xi * uspan)
                grid_max = mono_hi + xi * uspan
            else:
                grid_min = agent_cfg.get("min_price", None)
                grid_max = agent_cfg.get("max_price", None)
            import logging as _log

            _m = int(agent_cfg.get("m", 15))
            _log.getLogger(__name__).info(
                "Price grid (union): [%.2f, %.2f] spans per-store Nash..mono "
                "[%.2f, %.2f]; m=%d -> step=%.3f EUR. Recommend m>=21 for adequate "
                "per-chain resolution.",
                grid_min,
                grid_max,
                nash_lo,
                mono_hi,
                _m,
                (grid_max - grid_min) / max(_m - 1, 1),
            )
        else:
            span = p_mono_pre - p_nash_pre
            if span > 1e-6:
                grid_min = max(mc_min, p_nash_pre - xi * span)
                grid_max = p_mono_pre + xi * span
            else:
                grid_min = agent_cfg.get("min_price", None)
                grid_max = agent_cfg.get("max_price", None)
    else:
        if auto_grid and not _can_run_benchmarks:
            import logging as _log
            _log.getLogger(__name__).warning(
                "dense_distances=False: city.dist2_km2 is None — skipping "
                "Bertrand-Nash / joint-monopoly benchmark computation. "
                "Set min_price / max_price in the agent config, or implement "
                "the catchment-aware benchmark solvers (Prompt 4)."
            )
        grid_min = agent_cfg.get("min_price", None)
        grid_max = agent_cfg.get("max_price", None)

    # --- 1c. Chain-type-specific grids (optional) ---
    import numpy as np
    import logging as _log_chs
    chain_type_grids: dict | None = None
    if bool(agent_cfg.get("chain_specific_grid", False)) and _can_run_benchmarks:
        _m = int(agent_cfg.get("m", 15))
        _xi = float(agent_cfg.get("price_grid_xi", 0.1))
        _chain_types_arr = np.array([f.chain_type for f in firms], dtype=object)
        _mc_arr = np.array([f.marginal_cost for f in firms], dtype=np.float64)
        _mc_all_zero = bool((_mc_arr == 0.0).all())

        chain_type_grids = {}
        for ct in ("discount", "standard", "bio"):
            ct_mask = _chain_types_arr == ct
            if ct_mask.sum() == 0:
                continue
            p_nash_ct = float(p_nash_arr[ct_mask].mean())
            p_mono_ct = float(p_mono_arr[ct_mask].mean())
            mc_ct = float(_mc_arr[ct_mask].mean())
            span_ct = p_mono_ct - p_nash_ct

            if _mc_all_zero:
                ct_lo = max(0.0, p_nash_ct - _xi * max(span_ct, 1e-6))
                ct_hi = p_mono_ct + _xi * max(span_ct, 1e-6)
            else:
                ct_lo = max(0.0, mc_ct)
                ct_hi = p_mono_ct + _xi * max(span_ct, 1e-6)

            chain_type_grids[ct] = np.linspace(ct_lo, ct_hi, _m)
            _log_chs.getLogger(__name__).info(
                "Chain grid %s: [%.4f, %.4f] (Nash=%.4f, Mono=%.4f, MC=%.4f, m=%d)",
                ct, ct_lo, ct_hi, p_nash_ct, p_mono_ct, mc_ct, _m,
            )

        if chain_type_grids:
            all_lo = min(g.min() for g in chain_type_grids.values())
            all_hi = max(g.max() for g in chain_type_grids.values())
            grid_min = float(all_lo)
            grid_max = float(all_hi)
    elif bool(agent_cfg.get("chain_specific_grid", False)) and not _can_run_benchmarks:
        import logging as _log_chs2
        _log_chs2.getLogger(__name__).warning(
            "--chs-grid requires dense_distances=True and benchmark computation. "
            "Falling back to global grid."
        )

    # --- 1d. graph_states: reciprocal rival observation graph (built at Bertrand-Nash) ---
    _graph_rivals = None
    if str(agent_cfg.get("state_mode", "neighbors")) == "graph_states":
        if not _can_run_benchmarks or p_nash_arr is None or p_mono_arr is None:
            raise ValueError(
                "--graph-states requires dense benchmarks (dense_distances=true, "
                "auto_price_grid=true); Bertrand-Nash / monopoly arrays are unavailable."
            )
        if city.catch_indptr is None:
            raise ValueError(
                "--graph-states requires a sparse catchment; set catchment_minutes "
                "in the env config (e.g. catchment_minutes: 8.0)."
            )
        import logging as _log_gs
        from hotelling.env.rival_graph import (
            build_rival_graph, compute_competition_matrix, diversion_edge_weights,
        )
        _gs_log = _log_gs.getLogger(__name__)

        _graph_own_grid = str(agent_cfg.get("graph_own_grid_type", "G"))
        _xi_gs = float(agent_cfg.get("price_grid_xi", 0.1))
        _chain_types_gs = np.array([f.chain_type for f in firms], dtype=object)

        # "G": one global grid over the union of per-chain Nash..mono bands, dropping the
        # dead [MC, p_N) region (NO mc_min floor). "CS": chain_type_grids already built.
        if _graph_own_grid == "G":
            _los, _his = [], []
            for _ct in ("discount", "standard", "bio"):
                _mct = _chain_types_gs == _ct
                if _mct.sum() == 0:
                    continue
                _pn = float(p_nash_arr[_mct].mean())
                _pm = float(p_mono_arr[_mct].mean())
                _gc = max(_pm - _pn, 1e-6)
                _los.append(_pn - _xi_gs * _gc)
                _his.append(_pm + _xi_gs * _gc)
            grid_min = float(min(_los))
            grid_max = float(max(_his))
            chain_type_grids = None  # force a single global grid in the env
            _gs_log.info(
                "graph_states 'G' global grid: [%.2f, %.2f]; m=%d -> step=%.3f EUR.",
                grid_min, grid_max, int(agent_cfg.get("m", 18)),
                (grid_max - grid_min) / max(int(agent_cfg.get("m", 18)) - 1, 1),
            )

        _eff0 = np.zeros(len(firms), dtype=np.float64)
        _M_mat, _E_vec = compute_competition_matrix(city, p_nash_arr, _eff0)
        _W_mat = diversion_edge_weights(_M_mat, _E_vec)
        _graph_obj = build_rival_graph(
            _W_mat,
            int(agent_cfg.get("graph_k", 2)),
            match_mode=str(agent_cfg.get("graph_rival_match", "A")),
            chain_types=_chain_types_gs,
            candidate_topn=int(agent_cfg.get("graph_candidate_topn", 8)),
            min_edge_weight=float(agent_cfg.get("graph_min_edge_weight", 0.0)),
        )
        _graph_rivals = _graph_obj.rivals
        np.save(output_dir / "graph_rivals.npy", _graph_obj.rivals)

        # Interactive rival-graph map into the run folder (best-effort).
        try:
            from hotelling.viz.rival_graph_map import write_rival_graph_map
            write_rival_graph_map(
                output_dir / "rival_graph.html", firms, _graph_obj,
                title=(
                    f"Rival graph (k={agent_cfg.get('graph_k', 2)}, "
                    f"grid={_graph_own_grid}, match={agent_cfg.get('graph_rival_match', 'A')})"
                ),
            )
        except Exception as _map_exc:  # noqa: BLE001
            _gs_log.warning("Rival-graph map generation failed: %s", _map_exc)

    # --- 2. Create environment ---
    env = HotellingMarketEnv(
        city=city,
        firms=firms,
        m=int(agent_cfg.get("m", 15)),
        m_effort=int(agent_cfg.get("m_effort", 5)),
        e_max=float(agent_cfg.get("e_max", 10.0)),
        k_neighbors=int(agent_cfg.get("k_neighbors", 1)),
        transport_cost=float(env_cfg.get("transport_cost", 0.01)),
        min_price=float(grid_min) if grid_min is not None else None,
        max_price=float(grid_max) if grid_max is not None else None,
        state_mode=str(agent_cfg.get("state_mode", "neighbors")),
        local_sum_n=agent_cfg.get("local_sum_n", None),
        n_price_bins=int(agent_cfg.get("n_price_bins", 15)),
        summary_stats=tuple(agent_cfg.get("summary_stats", ("mean",))),
        local_summary_detailed=bool(
            agent_cfg.get("local_summary_detailed", False)
        ),
        chain_type_grids=chain_type_grids,
        n_comp_bins=int(agent_cfg.get("n_comp_bins", 15)),
        p_nash_arr=p_nash_arr,
        p_mono_arr=p_mono_arr,
        calvano_k=int(agent_cfg.get("calvano_k", 1)),
        hybrid_n_profit=int(agent_cfg.get("hybrid_n_profit", 5)),
        hybrid_n_gap=int(agent_cfg.get("hybrid_n_gap", 9)),
        hybrid_gap_lo=float(agent_cfg.get("hybrid_gap_lo", -0.20)),
        hybrid_gap_hi=float(agent_cfg.get("hybrid_gap_hi",  0.20)),
        graph_rivals=_graph_rivals,
        graph_k=int(agent_cfg.get("graph_k", 2)),
        graph_n_rival_bins=int(agent_cfg.get("graph_n_rival_bins", 10)),
        graph_rival_match=str(agent_cfg.get("graph_rival_match", "A")),
    )

    import logging as _log_state
    _log_state.getLogger(__name__).info(
        "State config: mode=%s, state_size=%d, action_size=%d, "
        "Q-table cells per store=%d",
        env.state_mode, env.state_size, env._action_size,
        env.state_size * env._action_size,
    )

    T_burnin = int(phase0_cfg.get("T_burnin", 1_000_000))

    # --- Exploration schedule: two-stage (default) or single-stage fallback ---
    _beta_auto = bool(agent_cfg.get("beta_decay_auto", True))
    _sched_params: dict | None = None
    if _beta_auto:
        _use_two_stage = str(agent_cfg.get("beta_schedule", "two_stage")) == "two_stage"
        if _use_two_stage:
            _sched_params = compute_two_stage_schedule(
                T_burnin,
                explore_fraction=float(agent_cfg.get("explore_fraction", 0.65)),
                epsilon_transition=float(agent_cfg.get("epsilon_transition", 0.10)),
                epsilon_min=float(agent_cfg.get("epsilon_min", 3e-4)),
            )
            agent_cfg["beta_decay"] = _sched_params["beta1"]
            agent_cfg["beta1"] = _sched_params["beta1"]
            agent_cfg["beta2"] = _sched_params["beta2"]
            agent_cfg["t0_schedule"] = _sched_params["t0"]
            agent_cfg["epsilon_min"] = _sched_params["epsilon_min"]
            agent_cfg["epsilon_transition"] = _sched_params["epsilon_transition"]
            agent_cfg["mean_epsilon_approx"] = _sched_params["mean_epsilon_approx"]
        else:
            _beta = compute_beta_decay(T_burnin)
            agent_cfg["beta_decay"] = _beta
            agent_cfg["beta_schedule"] = "single"
            agent_cfg["beta1"] = None
            agent_cfg["beta2"] = None
            agent_cfg["t0_schedule"] = 0
            agent_cfg["mean_epsilon_approx"] = float("nan")
    else:
        agent_cfg["beta_schedule"] = "single"
        agent_cfg.setdefault("beta1", None)
        agent_cfg.setdefault("beta2", None)
        agent_cfg.setdefault("t0_schedule", 0)
        agent_cfg.setdefault("mean_epsilon_approx", float("nan"))

    use_batch = bool(agent_cfg.get("use_batch", True))
    agents: Dict[str, Any] | None = None
    batch_agent = None

    if use_batch:
        from hotelling.agents.batch_qlearning import BatchQLearningAgent

        _b1 = agent_cfg.get("beta1")
        _b2 = agent_cfg.get("beta2")
        _t0_sched = int(agent_cfg.get("t0_schedule", 0))
        batch_agent = BatchQLearningAgent(
            n_agents=len(firms),
            m=int(agent_cfg.get("m", 15)),
            m_effort=int(agent_cfg.get("m_effort", 5)),
            k=int(agent_cfg.get("k_neighbors", 1)),
            alpha=float(agent_cfg.get("alpha_lr", 0.15)),
            beta_decay=float(agent_cfg.get("beta_decay", 4e-6)),
            delta=float(agent_cfg.get("delta", 0.95)),
            seed=int(seed) if seed is not None else None,
            state_mode=str(agent_cfg.get("state_mode", "neighbors")),
            state_size=env.state_size,
            epsilon_min=float(agent_cfg.get("epsilon_min", 3e-4)),
            beta1=float(_b1) if _b1 is not None else None,
            beta2=float(_b2) if _b2 is not None else None,
            t0=_t0_sched,
            epsilon_transition=float(agent_cfg.get("epsilon_transition", 0.10)),
        )
    else:
        agents = {
            str(f.id): QLearningAgent(
                firm_id=str(f.id),
                m=int(agent_cfg.get("m", 15)),
                m_effort=int(agent_cfg.get("m_effort", 5)),
                e_max=float(agent_cfg.get("e_max", 10.0)),
                k=int(agent_cfg.get("k_neighbors", 1)),
                alpha=float(agent_cfg.get("alpha_lr", 0.15)),
                beta_decay=float(agent_cfg.get("beta_decay", 4e-6)),
                delta=float(agent_cfg.get("delta", 0.95)),
                update_mode=str(agent_cfg.get("update_mode", "sync")),
                seed=(int(seed) + i) if seed is not None else None,
            )
            for i, f in enumerate(firms)
        }

    # --- Q-table initialization (Calvano eq. 8 and variants) ---
    _qtable_init_mode_str = str(config.get("qtable_init", "zero"))
    if _qtable_init_mode_str != "zero" and batch_agent is not None and _can_run_benchmarks:
        import logging as _log_qi_init
        from hotelling.agents.qtable_init import QtableInitMode, compute_q_init

        _qi_mode = QtableInitMode.from_cli(_qtable_init_mode_str)
        _log_qi_init.getLogger(__name__).info(
            "Computing Q-table initialization: mode=%s ...", _qi_mode.value,
        )
        _q_init = compute_q_init(
            _qi_mode,
            env=env,
            city=city,
            n_agents=len(firms),
            state_size=env.state_size,
            action_size=env._action_size,
            m=int(agent_cfg.get("m", 15)),
            m_effort=int(agent_cfg.get("m_effort", 1)),
            delta=float(agent_cfg.get("delta", 0.95)),
            p_nash_arr=p_nash_arr,
            p_mono_arr=p_mono_arr,
            transport_cost=float(env_cfg.get("transport_cost", 0.01)),
        )
        batch_agent.set_q_init(_q_init)
    elif _qtable_init_mode_str != "zero" and not _can_run_benchmarks:
        import logging as _log_qi
        _log_qi.getLogger(__name__).warning(
            "--qtable-init=%s requires benchmark computation "
            "(dense_distances=True, auto_price_grid=True). "
            "Falling back to zero initialization.",
            _qtable_init_mode_str,
        )
    elif _qtable_init_mode_str != "zero" and batch_agent is None:
        import logging as _log_qi2
        _log_qi2.getLogger(__name__).warning(
            "--qtable-init=%s is only supported with use_batch=True. "
            "Falling back to zero initialization.",
            _qtable_init_mode_str,
        )

    record_every = int(phase0_cfg.get("record_every", phase0_cfg.get("check_interval", 1_000)))

    dense_log = None
    recorder = None
    if use_batch:
        from hotelling.simulation.dense_log import DenseLog

        dense_log = DenseLog(
            run_dir=output_dir,
            T=T_burnin,
            N=len(firms),
            agent_ids=[str(f.id) for f in firms],
            price_grid=env.price_grid,
            effort_grid=env.effort_grid,
            store_price_grids=getattr(env, '_store_price_grids', None),
            store_demand_profit=bool(
                phase0_cfg.get("store_demand_profit", True)
            ),
            float_dtype=str(
                phase0_cfg.get("float_dtype", "float32")
            ),
            dense_stride=int(
                phase0_cfg.get("dense_stride", 1)
            ),
            dense_tail=(
                int(phase0_cfg["dense_tail"])
                if phase0_cfg.get("dense_tail") is not None
                else None
            ),
            store_effort=bool(phase0_cfg.get("store_effort", True)),
        )
    else:
        recorder = SimulationRecorder(
            run_dir=output_dir,
            run_id=run_id,
        )

    phase0_cfg_with_recorder: Dict[str, Any] = {
        **phase0_cfg,
        "_recorder": recorder,
        "benchmark_cache_path": str(benchmark_cache_pre),
        "p_nash_precomputed": p_nash_pre,
        "p_mono_precomputed": p_mono_pre,
    }
    if batch_agent is not None:
        phase0_cfg_with_recorder["_batch_agent"] = batch_agent
    if dense_log is not None:
        phase0_cfg_with_recorder["_dense_log"] = dense_log

    phase0 = Phase0BurnIn(phase0_cfg_with_recorder)
    phase0_result = phase0.run(
        agents=agents,
        env=env,
        city=city,
        transport_cost=float(env_cfg.get("transport_cost", 0.01)),
        seed=seed,
        batch_agent=batch_agent,
    )

    # Flush outputs
    agents_parquet_path = None
    if recorder is not None:
        agents_parquet_path = recorder.flush()
    if dense_log is not None:
        dense_log.flush()

    if batch_agent is not None:
        batch_agent.save_qtable(output_dir / "qtable.npz")

    import numpy as np
    np.save(output_dir / "price_grid.npy", env.price_grid)
    if chain_type_grids is not None:
        for ct, grid in chain_type_grids.items():
            np.save(output_dir / f"price_grid_{ct}.npy", grid)

    deltas_by_chain: Dict[str, float] = {}
    chain_price_table: Dict[str, Any] = {}
    realized_outside_share = float("nan")
    realized_chain_shares: Dict[str, float] = {}
    deltas_profit_by_chain: Dict[str, float] = {}

    if p_nash_arr is not None and p_mono_arr is not None:
        N = len(firms)
        final_prices = phase0_result.get("final_prices", {})
        p_learned = np.array(
            [float(final_prices.get(str(f.id), np.nan)) for f in firms],
            dtype=np.float64,
        )
        chain_types = np.array([f.chain_type for f in firms], dtype=object)

        def _delta(mask: np.ndarray) -> float:
            if mask.sum() == 0:
                return float("nan")
            pl = float(np.nanmean(p_learned[mask]))
            pn = float(p_nash_arr[mask].mean())
            pm = float(p_mono_arr[mask].mean())
            denom = pm - pn
            if abs(denom) < 1e-9:
                return float("nan")
            return float(np.clip((pl - pn) / denom, -0.5, 1.5))

        deltas_by_chain = {
            "global": _delta(np.ones(N, dtype=bool)),
            "discount": _delta(chain_types == "discount"),
            "standard": _delta(chain_types == "standard"),
            "bio": _delta(chain_types == "bio"),
        }
        chain_price_table = {}
        for ct in ("discount", "standard", "bio", "global"):
            m = np.ones(N, bool) if ct == "global" else (chain_types == ct)
            if m.sum() > 0:
                chain_price_table[ct] = {
                    "n": int(m.sum()),
                    "learned": float(np.nanmean(p_learned[m])),
                    "nash": float(p_nash_arr[m].mean()),
                    "mono": float(p_mono_arr[m].mean()),
                }

        from hotelling.core.market import cell_choice_mass

        _qual = np.array([f.quality for f in firms], dtype=np.float64)
        _eff = np.zeros(N, dtype=np.float64)
        try:
            inside, outside = cell_choice_mass(
                prices=p_nash_arr,
                efforts=_eff,
                dist2_km2=city.dist2_km2,
                cell_pop=city.cell_pop,
                lambda_phi=city.lambda_phi,
                pi_H=city.pi_H,
                pi_H_lambda_phi=city.pi_H_lambda_phi,
                alpha=city.alpha,
                quality=_qual,
                beta=city.beta,
                transport_cost=tc,
                mu=city.mu,
                a0=city.a0,
                transport_exponent=getattr(city, "transport_exponent", 1.0),
            )
            total_mass = float((city.cell_pop + city.lambda_phi).sum())
            realized_outside_share = float(outside.sum() / total_mass)
            D = inside.sum(axis=0)
            tot_inside = float(D.sum())
            realized_chain_shares = {
                ct: float(D[chain_types == ct].sum() / tot_inside)
                for ct in ("discount", "standard", "bio")
            }
        except Exception:
            realized_outside_share = float("nan")
            realized_chain_shares = {}

        # ── Gross profit Δ ────────────────────────────────────────────────
        #    Canonical Calvano Δ on profits: (Σπ* − Σπᴺ) / (Σπᴹ − Σπᴺ)
        #
        #    Phase-0 / price-only mode: all efforts are frozen at zero for
        #    all three benchmark evaluations so the footing is identical.
        #    Demands at p_learned, p_nash, and p_mono are recomputed
        #    analytically via market_clearing (zero effort), avoiding any
        #    dependence on per-period demand noise in the simulation output.
        #
        #    In --with-effort mode this remains a gross price-driven index
        #    (effort=0 for all three); a joint (price, effort) cartel
        #    benchmark is a documented follow-up.
        try:
            from hotelling.core.market import market_clearing as _mc_prd
            import logging as _log_prd
            _prd_log = _log_prd.getLogger(__name__)

            _costs_prd = np.array([f.marginal_cost for f in firms], dtype=np.float64)
            _e0_prd = np.zeros(N, dtype=np.float64)

            if not np.any(np.isnan(p_learned)):
                _d_lrn, _ = _mc_prd(p_learned,    _e0_prd, city, tc)
                _d_nsh, _ = _mc_prd(p_nash_arr,   _e0_prd, city, tc)
                _d_mno, _ = _mc_prd(p_mono_arr,   _e0_prd, city, tc)

                _pi_lrn = (p_learned    - _costs_prd) * _d_lrn
                _pi_nsh = (p_nash_arr   - _costs_prd) * _d_nsh
                _pi_mno = (p_mono_arr   - _costs_prd) * _d_mno

                def _dpi_baseline(mask: np.ndarray) -> float:
                    if mask.sum() == 0:
                        return float("nan")
                    rn = float(np.nansum(_pi_lrn[mask]))
                    nn = float(_pi_nsh[mask].sum())
                    mn = float(_pi_mno[mask].sum())
                    d = mn - nn
                    if abs(d) < 1e-9:
                        return float("nan")
                    return float(np.clip((rn - nn) / d, -0.5, 1.5))

                deltas_profit_by_chain = {
                    "global":   _dpi_baseline(np.ones(N, dtype=bool)),
                    "discount": _dpi_baseline(chain_types == "discount"),
                    "standard": _dpi_baseline(chain_types == "standard"),
                    "bio":      _dpi_baseline(chain_types == "bio"),
                }
            else:
                _prd_log.warning("Baseline profit-Δ skipped: NaN in learned prices.")
        except Exception as _e_prd:
            import logging as _log_prd_exc
            _log_prd_exc.getLogger(__name__).warning(
                "Baseline profit-Δ computation failed: %s", _e_prd
            )

    phase0_result["deltas_by_chain"] = deltas_by_chain
    phase0_result["chain_price_table"] = chain_price_table
    phase0_result["realized_outside_share"] = realized_outside_share
    phase0_result["realized_chain_shares"] = realized_chain_shares
    phase0_result["deltas_profit_by_chain"] = deltas_profit_by_chain

    # Save metadata.json
    metadata = {
        "run_id": run_id,
        "run_name": run_name,
        "env_config_path": config.get("env_config_path"),
        "seed": seed,
        "converged": phase0_result.get("converged", False),
        "n_steps": phase0_result.get("n_steps", 0),
        "delta": phase0_result.get("delta", None),
        "mean_final_price": phase0_result.get("mean_final_price", None),
        "p_nash": phase0_result.get("p_nash", None),
        "p_mono": phase0_result.get("p_mono", None),
        "elapsed_s": round(time.time() - t_start, 2),
        "n_firms": len(firms),
        "T_burnin": T_burnin,
        "record_every": record_every,
        "agents_parquet": str(agents_parquet_path) if agents_parquet_path else None,
        "dense_log_meta": str(output_dir / "dense_log_meta.json")
        if dense_log is not None
        else None,
        "qtable": str(output_dir / "qtable.npz") if batch_agent is not None else None,
        "use_batch": use_batch,
        "state_mode": agent_cfg.get("state_mode", "neighbors"),
        "local_sum_n": agent_cfg.get("local_sum_n", None),
        "n_price_bins": agent_cfg.get("n_price_bins", 15),
        "summary_stats": agent_cfg.get("summary_stats", ["mean"]),
        "local_summary_detailed": bool(
            agent_cfg.get("local_summary_detailed", False)
        ),
        "local_summary_channels": [list(ch) for ch in env._ls_channels],
        "price_grid_mode": agent_cfg.get("price_grid_mode", "union"),
        "grid_min": float(grid_min) if grid_min is not None else None,
        "grid_max": float(grid_max) if grid_max is not None else None,
        "n_comp_bins": int(agent_cfg.get("n_comp_bins", 15)),
        "calvano_k": int(agent_cfg.get("calvano_k", 1)),
        "graph_k": int(agent_cfg.get("graph_k", 2)),
        "graph_n_rival_bins": int(agent_cfg.get("graph_n_rival_bins", 10)),
        "graph_own_grid_type": str(agent_cfg.get("graph_own_grid_type", "G")),
        "graph_rival_match": str(agent_cfg.get("graph_rival_match", "A")),
        "graph_rival_bin_axis": (
            "per_chain_type"
            if (
                bool(agent_cfg.get("chain_specific_grid", False))
                and str(agent_cfg.get("graph_rival_match", "A")).upper() == "SC"
            )
            else "global_union"
        ),
        "graph_candidate_topn": int(agent_cfg.get("graph_candidate_topn", 8)),
        "graph_min_edge_weight": float(agent_cfg.get("graph_min_edge_weight", 0.0)),
        "hybrid_n_profit": int(agent_cfg.get("hybrid_n_profit", 5)),
        "hybrid_n_gap": int(agent_cfg.get("hybrid_n_gap", 9)),
        "hybrid_gap_lo": float(agent_cfg.get("hybrid_gap_lo", -0.20)),
        "hybrid_gap_hi": float(agent_cfg.get("hybrid_gap_hi",  0.20)),
        "chain_specific_grid": bool(agent_cfg.get("chain_specific_grid", False)),
        "beta_decay": float(agent_cfg.get("beta_decay", 4e-6)),
        "beta_decay_auto": bool(agent_cfg.get("beta_decay_auto", True)),
        "beta_schedule": str(agent_cfg.get("beta_schedule", "two_stage")),
        "beta1": float(agent_cfg["beta1"]) if agent_cfg.get("beta1") is not None else None,
        "beta2": float(agent_cfg["beta2"]) if agent_cfg.get("beta2") is not None else None,
        "t0_schedule": int(agent_cfg.get("t0_schedule", 0)),
        "epsilon_min": float(agent_cfg.get("epsilon_min", 3e-4)),
        "epsilon_transition": float(agent_cfg.get("epsilon_transition", 0.10)),
        "explore_fraction": float(agent_cfg.get("explore_fraction", 0.65)),
        "mean_epsilon_approx": float(agent_cfg.get("mean_epsilon_approx", float("nan"))),
        "qtable_init": str(config.get("qtable_init", "zero")),
        "deltas_by_chain": deltas_by_chain,
        "chain_price_table": chain_price_table,
        "realized_outside_share": realized_outside_share,
        "realized_chain_shares": realized_chain_shares,
        "deltas_profit_by_chain": deltas_profit_by_chain,
        "profit_delta_is_gross": True,
    }
    with (output_dir / "metadata.json").open("w") as _f:
        json.dump(metadata, _f, indent=2)

    # Save aggregate history (step, mean_price, mean_effort) to aggregate.parquet
    agg_df = pd.DataFrame({
        "step": phase0_result.get("step_history", []),
        "mean_price": phase0_result.get("price_history", []),
        "mean_effort": phase0_result.get("effort_history", []),
    })
    pbc = phase0_result.get("price_history_by_chain", {})
    for ct in ("discount", "standard", "bio"):
        col = list(pbc.get(ct, []))
        if len(col) < len(agg_df):
            col = col + [float("nan")] * (len(agg_df) - len(col))
        agg_df[f"mean_price_{ct}"] = col[: len(agg_df)]
    agg_df.to_parquet(output_dir / "aggregate.parquet", index=False)

    # Append one row to the global index CSV
    index_path = Path(config.get("output_dir", "results/runs")).parent / "index.csv"
    index_row = pd.DataFrame([{
        "run_name": run_name,
        "run_id": run_id,
        "seed": seed,
        "converged": metadata["converged"],
        "n_steps": metadata["n_steps"],
        "delta": metadata["delta"],
        "mean_final_price": metadata["mean_final_price"],
        "p_nash": metadata["p_nash"],
        "p_mono": metadata["p_mono"],
        "n_firms": metadata["n_firms"],
        "elapsed_s": metadata["elapsed_s"],
    }])
    if index_path.exists():
        index_row.to_csv(index_path, mode="a", header=False, index=False)
    else:
        index_row.to_csv(index_path, mode="w", header=True, index=False)

    elapsed = time.time() - t_start
    return {
        "run_id": run_id,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "seed": seed,
        "elapsed_s": round(elapsed, 2),
        "beta_decay": float(agent_cfg.get("beta_decay", 4e-6)),
        "beta_schedule": str(agent_cfg.get("beta_schedule", "two_stage")),
        "beta1": agent_cfg.get("beta1"),
        "beta2": agent_cfg.get("beta2"),
        "t0_schedule": int(agent_cfg.get("t0_schedule", 0)),
        "epsilon_min": float(agent_cfg.get("epsilon_min", 3e-4)),
        "epsilon_transition": float(agent_cfg.get("epsilon_transition", 0.10)),
        "mean_epsilon_approx": float(agent_cfg.get("mean_epsilon_approx", float("nan"))),
        "qtable_init": str(config.get("qtable_init", "zero")),
        **phase0_result,
    }


def run_sweep(
    config_dir: Path,
    sweep_config_name: str,
    n_jobs: int = -1,
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run a parameter sweep defined in configs/sweep/.

    Loads the sweep config, expands the grid, and runs each configuration
    in parallel using multiprocessing.Pool (n_jobs=-1 = all CPUs).

    Parameters
    ----------
    config_dir : directory containing Hydra config files
    sweep_config_name : name of the sweep YAML (without .yaml extension)
    n_jobs : number of parallel workers; -1 uses os.cpu_count()
    output_dir : directory to write per-run Parquet files

    Returns
    -------
    list of result dicts from all sessions, one per parameter combination
    """
    raise NotImplementedError


def _build_components(config: dict, output_dir=None) -> dict:
    """Build City, Firms, env, two-stage-warmed batch agent, benchmarks, grids.

    Mirrors run_single_session's construction (kept separate so the Phase-0
    baseline stays untouched) so the strategic Phase-0 burn-in is behaviourally
    identical to the baseline up to the CEO layer — including ``--graph-states``
    (reciprocal rival graph + global/chain grids) and the two-stage exploration
    schedule. Returns a dict of components for the strategic run.

    When ``output_dir`` is given and state_mode == "graph_states", writes
    ``graph_rivals.npy`` and (best-effort) ``rival_graph.html`` into it. For a
    ``--from-run`` graph-states run the rivals are LOADED from the source run's
    ``graph_rivals.npy`` so the loaded Q-table's state encoding matches exactly.
    """
    import logging
    from pathlib import Path

    import geopandas as gpd
    import numpy as np

    from hotelling.agents.batch_qlearning import BatchQLearningAgent
    from hotelling.core.equilibrium import bertrand_nash, joint_monopoly
    from hotelling.env.market_env import HotellingMarketEnv
    from hotelling.spatial.loader import load_berlin_city

    _log = logging.getLogger("hotelling.strategic")

    env_cfg = config["env"]
    agent_cfg = config["agents"]
    seed = config["phase2"].get("seed", None)
    with_effort = bool(config.get("with_effort", False))
    from_run = config.get("from_run")

    _cm_raw = env_cfg.get("catchment_minutes", None)
    city, firms = load_berlin_city(
        grid_path=env_cfg.get("grid_path", "data/processed/demand_grid.parquet"),
        stores_path=env_cfg.get("stores_path", "data/processed/supermarkets.parquet"),
        travel_times_path=env_cfg.get("travel_times_path", "data/processed/travel_times.parquet"),
        lambda_val=float(env_cfg["lambda_val"]),
        q_S=float(env_cfg.get("q_S", 0.8)), q_B=float(env_cfg.get("q_B", 1.5)),
        alpha_L=float(env_cfg.get("alpha_L", 0.5)), alpha_H=float(env_cfg.get("alpha_H", 1.5)),
        beta_effort=float(env_cfg.get("beta_effort", 0.001)),
        kappa0=float(env_cfg.get("kappa0", 1.0)),
        store_size=float(env_cfg.get("store_size", 600.0)),
        transport_cost=float(env_cfg.get("transport_cost", 0.01)),
        a0=float(env_cfg.get("a0", -1.0)), mu=float(env_cfg.get("mu", 0.25)),
        nan_fill_minutes=float(env_cfg.get("nan_fill_minutes", 120.0)),
        marginal_cost_D=float(env_cfg.get("marginal_cost_D", 0.0)),
        marginal_cost_S=float(env_cfg.get("marginal_cost_S", 0.0)),
        marginal_cost_B=float(env_cfg.get("marginal_cost_B", 0.0)),
        rent_scale=float(env_cfg.get("rent_scale", 0.0)),
        rent_normalization=str(env_cfg.get("rent_normalization", "mean_ratio")),
        dense_distances=bool(env_cfg.get("dense_distances", True)),
        catchment_minutes=(float(_cm_raw) if _cm_raw is not None else None),
        catchment_k_min=int(env_cfg.get("catchment_k_min", 12)),
        catchment_k_max=int(env_cfg.get("catchment_k_max", 80)),
        precompute_expweights=bool(env_cfg.get("precompute_expweights", False)),
        low_precision_storage=bool(env_cfg.get("low_precision_storage", False)),
    )

    tc = float(env_cfg.get("transport_cost", 0.01))
    bench_cache = (
        Path(env_cfg.get("grid_path", "data/processed/demand_grid.parquet")).parent
        / "benchmarks_cache.npz"
    )
    auto_grid = bool(agent_cfg.get("auto_price_grid", True))
    grid_mode = str(agent_cfg.get("price_grid_mode", "union"))
    p_nash_arr = p_mono_arr = None
    e_nash_arr = None
    grid_min = agent_cfg.get("min_price", None)
    grid_max = agent_cfg.get("max_price", None)

    if auto_grid and city.dist2_km2 is not None:
        p_nash_arr, e_nash_arr = bertrand_nash(city, transport_cost=tc, cache_path=bench_cache)
        p_mono_arr, _ = joint_monopoly(
            city, transport_cost=tc, cache_path=bench_cache,
            effort_fixed=(e_nash_arr if with_effort else None),
        )
        xi = float(agent_cfg.get("price_grid_xi", 0.1))
        mc_min = min(getattr(f, "marginal_cost", 0.0) for f in firms)
        if grid_mode == "union":
            nash_lo, mono_hi = float(p_nash_arr.min()), float(p_mono_arr.max())
            span = mono_hi - nash_lo
        else:
            nash_lo, mono_hi = float(p_nash_arr.mean()), float(p_mono_arr.mean())
            span = mono_hi - nash_lo
        if span > 1e-6:
            grid_min = max(mc_min, nash_lo - xi * span)
            grid_max = mono_hi + xi * span

    # --- Chain-type-specific grids (GRID=CS and other chain_specific_grid runs) ---
    _bc_chain_grids: dict | None = None
    if bool(agent_cfg.get("chain_specific_grid", False)) and city.dist2_km2 is not None and p_nash_arr is not None:
        _xi_bc = float(agent_cfg.get("price_grid_xi", 0.1))
        _m_bc = int(agent_cfg.get("m", 25))
        _chain_types_bc = np.array([f.chain_type for f in firms], dtype=object)
        _mc_bc = np.array([f.marginal_cost for f in firms], dtype=np.float64)
        _mc_all_zero_bc = bool((_mc_bc == 0.0).all())
        _bc_chain_grids = {}
        for ct in ("discount", "standard", "bio"):
            ct_mask = _chain_types_bc == ct
            if ct_mask.sum() == 0:
                continue
            p_nash_ct = float(p_nash_arr[ct_mask].mean())
            p_mono_ct = float(p_mono_arr[ct_mask].mean())
            mc_ct = float(_mc_bc[ct_mask].mean())
            span_ct = p_mono_ct - p_nash_ct
            if _mc_all_zero_bc:
                ct_lo = max(0.0, p_nash_ct - _xi_bc * max(span_ct, 1e-6))
                ct_hi = p_mono_ct + _xi_bc * max(span_ct, 1e-6)
            else:
                ct_lo = max(0.0, mc_ct)
                ct_hi = p_mono_ct + _xi_bc * max(span_ct, 1e-6)
            _bc_chain_grids[ct] = np.linspace(ct_lo, ct_hi, _m_bc)
            _log.info("Chain grid (strategic) %s: [%.4f, %.4f]", ct, ct_lo, ct_hi)
        if _bc_chain_grids:
            grid_min = float(min(g.min() for g in _bc_chain_grids.values()))
            grid_max = float(max(g.max() for g in _bc_chain_grids.values()))

    # --- graph_states: reciprocal rival observation graph (mirrors run_single_session 1d) ---
    _graph_rivals = None
    _graph_obj = None
    _graph_own_grid = str(agent_cfg.get("graph_own_grid_type", "G"))
    if str(agent_cfg.get("state_mode", "neighbors")) == "graph_states":
        if not (auto_grid and city.dist2_km2 is not None) or p_nash_arr is None or p_mono_arr is None:
            raise ValueError(
                "--graph-states requires dense benchmarks (dense_distances=true, "
                "auto_price_grid=true); Bertrand-Nash / monopoly arrays are unavailable."
            )
        if city.catch_indptr is None:
            raise ValueError(
                "--graph-states requires a sparse catchment; set catchment_minutes "
                "in the env config (e.g. catchment_minutes: 8.0)."
            )
        _xi_gs = float(agent_cfg.get("price_grid_xi", 0.1))
        _chain_types_gs = np.array([f.chain_type for f in firms], dtype=object)

        # "G": one global grid over the union of per-chain Nash..mono bands, dropping the
        # dead [MC, p_N) region (NO mc_min floor). "CS": _bc_chain_grids already built above.
        if _graph_own_grid == "G":
            _los, _his = [], []
            for _ct in ("discount", "standard", "bio"):
                _mct = _chain_types_gs == _ct
                if _mct.sum() == 0:
                    continue
                _pn = float(p_nash_arr[_mct].mean())
                _pm = float(p_mono_arr[_mct].mean())
                _gc = max(_pm - _pn, 1e-6)
                _los.append(_pn - _xi_gs * _gc)
                _his.append(_pm + _xi_gs * _gc)
            grid_min = float(min(_los))
            grid_max = float(max(_his))
            _bc_chain_grids = None  # force a single global grid in the env
            _log.info(
                "graph_states 'G' global grid: [%.2f, %.2f]; m=%d -> step=%.3f EUR.",
                grid_min, grid_max, int(agent_cfg.get("m", 18)),
                (grid_max - grid_min) / max(int(agent_cfg.get("m", 18)) - 1, 1),
            )

        # --from-run: load the EXACT rivals the loaded Q-table was trained against.
        _loaded_from_run = False
        _from_src: Path | None = None
        if from_run:
            _from_src = Path(from_run)
            _gr_path = _from_src / "graph_rivals.npy"
            if _gr_path.exists():
                _graph_rivals = np.load(_gr_path).astype(np.int64)
                _loaded_from_run = True
                _log.info("graph_states: loaded graph_rivals.npy from --from-run (%s).", _gr_path)
            else:
                _log.warning(
                    "graph_states --from-run: no graph_rivals.npy in %s; rebuilding the "
                    "graph deterministically (b-matching ties may differ from the source run).",
                    _from_src,
                )

        if not _loaded_from_run:
            from hotelling.env.rival_graph import (
                build_rival_graph, compute_competition_matrix, diversion_edge_weights,
            )
            _eff0 = np.zeros(len(firms), dtype=np.float64)
            _M_mat, _E_vec = compute_competition_matrix(city, p_nash_arr, _eff0)
            _W_mat = diversion_edge_weights(_M_mat, _E_vec)
            _graph_obj = build_rival_graph(
                _W_mat,
                int(agent_cfg.get("graph_k", 2)),
                match_mode=str(agent_cfg.get("graph_rival_match", "A")),
                chain_types=_chain_types_gs,
                candidate_topn=int(agent_cfg.get("graph_candidate_topn", 8)),
                min_edge_weight=float(agent_cfg.get("graph_min_edge_weight", 0.0)),
            )
            _graph_rivals = _graph_obj.rivals

        # Persist artefacts into the strategic run folder (parity with baseline).
        if output_dir is not None:
            out_p = Path(output_dir)
            np.save(out_p / "graph_rivals.npy", _graph_rivals)
            if _graph_obj is not None:
                try:
                    from hotelling.viz.rival_graph_map import write_rival_graph_map
                    write_rival_graph_map(
                        out_p / "rival_graph.html", firms, _graph_obj,
                        title=(
                            f"Rival graph (k={agent_cfg.get('graph_k', 2)}, "
                            f"grid={_graph_own_grid}, match={agent_cfg.get('graph_rival_match', 'A')})"
                        ),
                    )
                except Exception as _map_exc:  # noqa: BLE001
                    _log.warning("Rival-graph map generation failed: %s", _map_exc)
            elif _loaded_from_run and _from_src is not None:
                _src_html = _from_src / "rival_graph.html"
                if _src_html.exists():
                    import shutil
                    try:
                        shutil.copyfile(_src_html, out_p / "rival_graph.html")
                    except Exception as _cp_exc:  # noqa: BLE001
                        _log.warning("Could not copy rival_graph.html from --from-run: %s", _cp_exc)

    env = HotellingMarketEnv(
        city=city, firms=firms,
        m=int(agent_cfg.get("m", 25)), m_effort=int(agent_cfg.get("m_effort", 1)),
        e_max=float(agent_cfg.get("e_max", 10.0)),
        k_neighbors=int(agent_cfg.get("k_neighbors", 1)), transport_cost=tc,
        min_price=float(grid_min) if grid_min is not None else None,
        max_price=float(grid_max) if grid_max is not None else None,
        state_mode=str(agent_cfg.get("state_mode", "neighbors")),
        local_sum_n=agent_cfg.get("local_sum_n", None),
        n_price_bins=int(agent_cfg.get("n_price_bins", 15)),
        summary_stats=tuple(agent_cfg.get("summary_stats", ("mean",))),
        local_summary_detailed=bool(agent_cfg.get("local_summary_detailed", False)),
        chain_type_grids=_bc_chain_grids,
        n_comp_bins=int(agent_cfg.get("n_comp_bins", 15)),
        p_nash_arr=p_nash_arr,
        p_mono_arr=p_mono_arr,
        calvano_k=int(agent_cfg.get("calvano_k", 1)),
        hybrid_n_profit=int(agent_cfg.get("hybrid_n_profit", 5)),
        hybrid_n_gap=int(agent_cfg.get("hybrid_n_gap", 9)),
        hybrid_gap_lo=float(agent_cfg.get("hybrid_gap_lo", -0.20)),
        hybrid_gap_hi=float(agent_cfg.get("hybrid_gap_hi", 0.20)),
        graph_rivals=_graph_rivals,
        graph_k=int(agent_cfg.get("graph_k", 2)),
        graph_n_rival_bins=int(agent_cfg.get("graph_n_rival_bins", 10)),
        graph_rival_match=str(agent_cfg.get("graph_rival_match", "A")),
    )

    # --- Two-stage exploration schedule (mirrors run_single_session) ---
    # Drives the Phase-0 burn-in so the strategic warm-up is identical to the
    # baseline. Computed against the strategic burn-in length (phase2.T_burnin).
    T_burnin = int(config["phase2"].get("T_burnin", 200_000))
    _beta_auto = bool(agent_cfg.get("beta_decay_auto", True))
    _sched_params: dict | None = None
    if _beta_auto:
        if str(agent_cfg.get("beta_schedule", "two_stage")) == "two_stage":
            _sched_params = compute_two_stage_schedule(
                T_burnin,
                explore_fraction=float(agent_cfg.get("explore_fraction", 0.65)),
                epsilon_transition=float(agent_cfg.get("epsilon_transition", 0.10)),
                epsilon_min=float(agent_cfg.get("epsilon_min", 3e-4)),
            )
            agent_cfg["beta_decay"] = _sched_params["beta1"]
            agent_cfg["beta1"] = _sched_params["beta1"]
            agent_cfg["beta2"] = _sched_params["beta2"]
            agent_cfg["t0_schedule"] = _sched_params["t0"]
            agent_cfg["epsilon_min"] = _sched_params["epsilon_min"]
            agent_cfg["epsilon_transition"] = _sched_params["epsilon_transition"]
            agent_cfg["mean_epsilon_approx"] = _sched_params["mean_epsilon_approx"]
        else:
            _beta = compute_beta_decay(T_burnin)
            agent_cfg["beta_decay"] = _beta
            agent_cfg["beta_schedule"] = "single"
            agent_cfg["beta1"] = None
            agent_cfg["beta2"] = None
            agent_cfg["t0_schedule"] = 0
            agent_cfg["mean_epsilon_approx"] = float("nan")
    else:
        agent_cfg["beta_schedule"] = "single"
        agent_cfg.setdefault("beta1", None)
        agent_cfg.setdefault("beta2", None)
        agent_cfg.setdefault("t0_schedule", 0)
        agent_cfg.setdefault("mean_epsilon_approx", float("nan"))

    _b1 = agent_cfg.get("beta1")
    _b2 = agent_cfg.get("beta2")
    _t0_sched = int(agent_cfg.get("t0_schedule", 0))
    batch_agent = BatchQLearningAgent(
        n_agents=len(firms), m=int(agent_cfg.get("m", 25)),
        m_effort=int(agent_cfg.get("m_effort", 1)), k=int(agent_cfg.get("k_neighbors", 1)),
        alpha=float(agent_cfg.get("alpha_lr", 0.15)),
        beta_decay=float(agent_cfg.get("beta_decay", 4e-6)),
        delta=float(agent_cfg.get("delta", 0.95)),
        seed=int(seed) if seed is not None else None,
        state_mode=str(agent_cfg.get("state_mode", "neighbors")),
        state_size=env.state_size,
        epsilon_min=float(agent_cfg.get("epsilon_min", 3e-4)),
        beta1=float(_b1) if _b1 is not None else None,
        beta2=float(_b2) if _b2 is not None else None,
        t0=_t0_sched,
        epsilon_transition=float(agent_cfg.get("epsilon_transition", 0.10)),
    )
    grid_gdf = gpd.read_parquet(env_cfg.get("grid_path", "data/processed/demand_grid.parquet"))
    return {
        "city": city, "firms": firms, "env": env, "batch_agent": batch_agent,
        "p_nash_arr": p_nash_arr, "p_mono_arr": p_mono_arr, "e_nash_arr": e_nash_arr,
        "grid_min": grid_min, "grid_max": grid_max, "grid_gdf": grid_gdf,
        "chain_type_grids": _bc_chain_grids,
        "graph_rivals": _graph_rivals,
        "graph_obj": _graph_obj,
        "graph_own_grid_type": _graph_own_grid,
        "two_stage_sched": _sched_params,
    }


def run_strategic_session(config: dict) -> dict:
    """Phase-0 burn-in -> Phase-2 CEO-only strategic game; write a run folder.

    config keys: env, agents, groups, ceo, phase2, output_dir.
    """
    import json
    import time
    import uuid
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import yaml

    import logging
    _log = logging.getLogger("hotelling.strategic")

    from hotelling.simulation.engine import BatchSimulationEngine
    from hotelling.simulation.phases import Phase2StrategicGame
    from hotelling.agents.chain_ceo import build_chain_ceos
    from hotelling.llm.client import LLMClient
    from hotelling.llm.ceo_state import build_consumer_zones
    from hotelling.envelope.groups import (
        assign_groups, composite_group_keys, build_store_metadata,
    )

    t_start = time.time()
    run_id = str(uuid.uuid4())[:8]
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}"
    out_root = Path(config.get("output_dir", "results/strategic_runs"))
    output_dir = out_root / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    comp = _build_components(config, output_dir=output_dir)
    env, batch_agent, firms = comp["env"], comp["batch_agent"], comp["firms"]
    p_nash_arr, p_mono_arr = comp["p_nash_arr"], comp["p_mono_arr"]
    e_nash_arr = comp.get("e_nash_arr")

    # Price-grid artefacts (parity with run_baseline; graph_rivals.npy +
    # rival_graph.html are already written inside _build_components).
    np.save(output_dir / "price_grid.npy", env.price_grid)
    _ctg = comp.get("chain_type_grids")
    if _ctg:
        for _ct, _grid in _ctg.items():
            np.save(output_dir / f"price_grid_{_ct}.npy", _grid)

    phase2_cfg = config["phase2"]
    with_effort = bool(config.get("with_effort", False))
    with_comm = bool(config.get("with_comm", False))
    T_measure = phase2_cfg.get("T_measure", None)
    seed = phase2_cfg.get("seed", None)
    T_burnin = int(phase2_cfg.get("T_burnin", 200_000))
    T_game = int(phase2_cfg.get("T_game", 5_000))
    T_CEO = int(phase2_cfg.get("T_CEO", 100))
    record_every = int(phase2_cfg.get("record_every", max(1, T_CEO)))
    no_ceo = bool(phase2_cfg.get("no_ceo", False))

    ceo_cfg = config.get("ceo", {}) or {}
    rpm = int(ceo_cfg.get("requests_per_minute", 15))
    rpd = int(ceo_cfg.get("requests_per_day", 1500))
    if not no_ceo:
        n_chains = len({str(f.chain) for f in firms})
        n_epochs_planned = T_game // T_CEO
        est_calls = n_chains * n_epochs_planned
        if rpd and est_calls > rpd:
            raise ValueError(
                f"Planned CEO calls ({n_chains} chains x {n_epochs_planned} epochs "
                f"= {est_calls}) exceed the daily limit ({rpd}). Reduce --T-game, "
                f"raise --T-CEO, or split the run across days."
            )
        floor_min = (est_calls / rpm) if rpm else 0.0
        _log.info(
            "CEO call budget: %d calls (%d chains x %d epochs); >= %.1f min "
            "wall-clock at %d rpm; daily cap %d. Model=%s.",
            est_calls, n_chains, n_epochs_planned, floor_min, rpm, rpd,
            ceo_cfg.get("model"),
        )

    # ── Phase 0: warm the per-store Q-tables (resets agent+env internally) ──
    from_run = config.get("from_run")
    if from_run:
        import numpy as np  # noqa: PLC0415
        qpath = Path(from_run) / "qtable.npz"
        if not qpath.exists():
            raise FileNotFoundError(
                f"--from-run given but no qtable.npz in {from_run}. "
                "Re-run run_baseline (it now writes qtable.npz)."
            )
        batch_agent.load_qtable(qpath)
        base_grid_path = Path(from_run) / "price_grid.npy"
        if base_grid_path.exists():
            base_grid = np.load(base_grid_path)
            if base_grid.shape != env.price_grid.shape or not np.allclose(
                base_grid, env.price_grid, atol=1e-4
            ):
                raise ValueError(
                    "price grid mismatch between the loaded run and the strategic env; "
                    "the env/benchmark configuration differs. Use the same --env-config."
                )
        env.reset(seed=seed)  # initialise env state without touching the loaded Q-table
        burn_result = {"epsilon_mean": float(batch_agent.epsilon_mean)}
        _log.info("Loaded converged Q-table from %s; skipping Phase-0 burn-in.", from_run)
    else:
        # Optional non-zero Q-table init (parity with run_baseline --qtable-init).
        _qi_mode_str = str(config.get("qtable_init", "zero"))
        if _qi_mode_str != "zero" and p_nash_arr is not None and comp["city"].dist2_km2 is not None:
            from hotelling.agents.qtable_init import QtableInitMode, compute_q_init
            _log.info("Computing Q-table initialization: mode=%s ...", _qi_mode_str)
            _q_init = compute_q_init(
                QtableInitMode.from_cli(_qi_mode_str),
                env=env, city=comp["city"], n_agents=len(firms),
                state_size=env.state_size, action_size=env._action_size,
                m=int(config["agents"].get("m", 25)),
                m_effort=int(config["agents"].get("m_effort", 1)),
                delta=float(config["agents"].get("delta", 0.95)),
                p_nash_arr=p_nash_arr, p_mono_arr=p_mono_arr,
                transport_cost=float(config["env"].get("transport_cost", 0.01)),
            )
            batch_agent.set_q_init(_q_init)
        elif _qi_mode_str != "zero":
            _log.warning(
                "--qtable-init=%s requires dense benchmarks; falling back to zero init.",
                _qi_mode_str,
            )
        burn = BatchSimulationEngine(
            env=env, batch_agent=batch_agent, max_steps=T_burnin,
            record_every=record_every, recorder=None, dense_log=None,
        )
        burn_result = burn.run(seed=seed)

    # ── Groups, zones, CEOs ────────────────────────────────────────────────
    groups_cfg = config.get("groups", {}) or {}
    active_divisions = list(groups_cfg.get("active_divisions", []))
    division_params = {
        "threshold_n_rivals": int(groups_cfg.get("competition_threshold_n_rivals", 3)),
        "radius_m": float(groups_cfg.get("competition_radius_m", 500.0)),
        "status_threshold": float(groups_cfg.get("neighbourhood_status_threshold", 0.5)),
    }
    group_keys = composite_group_keys(active_divisions, division_params)
    metadata = build_store_metadata(
        firms, grid_gdf=comp["grid_gdf"], radius_m=division_params["radius_m"]
    )
    labels_map = assign_groups(metadata, active_divisions, division_params)
    store_chain = [str(f.chain) for f in firms]
    store_chain_type = [str(f.chain_type) for f in firms]
    store_group_labels = [labels_map[str(f.id)] for f in firms]
    zones = build_consumer_zones(comp["grid_gdf"], firms, n_side=3)

    save_comm = bool(ceo_cfg.get("save_communication", False))
    client = LLMClient(
        model=str(ceo_cfg.get("model", "gemini/gemma-4-31b-it")),
        temperature=float(ceo_cfg.get("temperature", 0)),
        max_tokens=int(ceo_cfg.get("max_tokens", 2048)),
        max_retries=int(ceo_cfg.get("max_retries", 3)),
        log_path=ceo_cfg.get("log_path", str(output_dir / "llm_calls.jsonl")),
        requests_per_minute=rpm,
        requests_per_day=rpd,
        reasoning_effort=ceo_cfg.get("reasoning_effort", "none"),
        force_reasoning_effort=bool(ceo_cfg.get("force_reasoning_effort", False)),
        instructor_mode=str(ceo_cfg.get("instructor_mode", "json")),
        timeout=float(ceo_cfg.get("timeout", 120.0)),
        transient_max_attempts=int(ceo_cfg.get("transient_max_attempts", 5)),
        backoff_base=float(ceo_cfg.get("backoff_base", 2.0)),
        backoff_max=float(ceo_cfg.get("backoff_max", 60.0)),
        capture_raw=save_comm,
    )
    # ── Per-chain action-grid + rival-observation specs from the LIVE env ──────
    # Sourced from the env actually used this run (mirrors the --from-run config
    # exactly), so the CEO sees the true grid even though a --graph-states
    # baseline's metadata.json omits m / grid bounds.
    _brand_ct = {str(f.chain): str(f.chain_type) for f in firms}
    grid_specs: dict[str, dict] = {}
    graph_specs: dict[str, dict] = {}
    for _brand, _ct in _brand_ct.items():
        grid_specs[_brand] = env.grid_spec(_ct)
        graph_specs[_brand] = env.graph_degree_spec(_ct)

    # Auto-floor min_delta_p to at least one grid step so envelopes are grid-feasible
    # regardless of the YAML euro value (which can silently undershoot the grid).
    _grid_step_global = float(env.grid_spec(None).get("step", 0.0))
    _min_dp_cfg = float(ceo_cfg.get("min_delta_p", 1.5))
    _min_delta_p_eff = max(_min_dp_cfg, _grid_step_global)
    if _min_delta_p_eff > _min_dp_cfg:
        _log.info(
            "min_delta_p floored to one grid step: cfg=%.3f € -> effective=%.3f € "
            "(grid step=%.3f €).", _min_dp_cfg, _min_delta_p_eff, _grid_step_global,
        )

    ceos = build_chain_ceos(
        firms, client=client, active_divisions=active_divisions,
        division_params=division_params, group_keys=group_keys,
        min_delta_p=_min_delta_p_eff,
        min_delta_e=float(ceo_cfg.get("min_delta_e", 0.1)), T_ceo=T_CEO,
        merge_system=bool(ceo_cfg.get("merge_system_prompt", True)),
        capture_comm=save_comm,
        with_effort=with_effort,
        with_comm=with_comm,
        grid_specs=grid_specs,
        graph_specs=graph_specs,
    )

    # ── Phase 2: strategic game (continues from warmed state, no reset) ────
    from hotelling.simulation.dense_log import DenseLog
    dense_log = DenseLog(
        run_dir=output_dir,
        T=T_game,
        N=len(firms),
        agent_ids=[str(f.id) for f in firms],
        price_grid=env.price_grid,
        effort_grid=env.effort_grid,
        store_price_grids=getattr(env, '_store_price_grids', None),
        store_demand_profit=bool(phase2_cfg.get("store_demand_profit", True)),
        float_dtype=str(phase2_cfg.get("float_dtype", "float32")),
        dense_stride=int(phase2_cfg.get("dense_stride", 1)),
        dense_tail=(int(phase2_cfg["dense_tail"]) if phase2_cfg.get("dense_tail") is not None else None),
        store_effort=bool(phase2_cfg.get(
            "store_effort",
            bool(config.get("with_effort", False)) and bool(phase2_cfg.get("store_demand_profit", True)),
        )),
    )
    phase2 = Phase2StrategicGame(phase2_cfg)
    res = phase2.run(
        env=env, batch_agent=batch_agent, ceos=ceos,
        store_chain=store_chain, store_chain_type=store_chain_type,
        store_group_labels=store_group_labels, group_keys=group_keys, zones=zones,
        T_game=T_game, T_CEO=T_CEO,
        mask_effort=int(config["agents"].get("m_effort", 1)) > 1,
        no_ceo=no_ceo, record_every=record_every,
        dense_log=dense_log,
        store_metadata=metadata,
        enrich_groups=bool(ceo_cfg.get("group_analytics", False)),
        with_effort=with_effort,
        with_comm=with_comm,
        T_measure=(int(T_measure) if T_measure else None),
        p_nash_arr=p_nash_arr,
        p_mono_arr=p_mono_arr,
        strategic_analytics=bool(ceo_cfg.get("strategic_analytics", False)),
        tier_commit=bool(ceo_cfg.get("tier_commit", False)),
    )
    dense_log.flush()
    import json as _json
    with (output_dir / "ceo_decisions.jsonl").open("w") as _f:
        for rec in res.get("decision_log", []):
            _f.write(_json.dumps(rec) + "\n")

    if save_comm:
        import re as _re
        comm_dir = output_dir / "LLM_communication"
        comm_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for _ceo in ceos.values():
            for tr in getattr(_ceo, "transcripts", []):
                safe = _re.sub(r"[^0-9A-Za-z._-]+", "_", str(tr["chain"])).strip("_")
                fpath = comm_dir / f"{safe}_{tr['epoch']}.txt"
                with fpath.open("w") as fh:
                    fh.write("Prompt:\n")
                    fh.write(str(tr["prompt"]) + "\n\n")
                    fh.write("Response:\n")
                    fh.write(str(tr["response"]) + "\n")
                n_written += 1
        _log.info("Saved %d CEO LLM transcripts to %s", n_written, comm_dir)

    # ── CEO call integrity: a run where every CEO call failed yields a Δ that is
    #    indistinguishable from the no-CEO control and MUST NOT be treated as valid.
    ceo_success = int(sum(getattr(c, "n_success", 0) for c in ceos.values()))
    ceo_fail = int(sum(getattr(c, "n_fail", 0) for c in ceos.values()))
    ceo_total = ceo_success + ceo_fail
    ceo_errors = {
        b: getattr(c, "last_error", None)
        for b, c in ceos.items() if getattr(c, "n_fail", 0)
    }
    ceo_success_rate = (ceo_success / ceo_total) if ceo_total else float("nan")
    ceo_all_failed = (not no_ceo) and ceo_total > 0 and ceo_success == 0
    if ceo_all_failed:
        _log.error(
            "ALL %d CEO calls FAILED — Δ is INVALID (equals no-CEO noise). "
            "Example error: %s", ceo_total, next(iter(ceo_errors.values()), None),
        )
    elif (not no_ceo) and ceo_success_rate < 1.0:
        _log.warning("CEO call success rate %.1f%% (%d/%d); some epochs used the "
                     "retained/previous envelope.", 100 * ceo_success_rate,
                     ceo_success, ceo_total)

    # ── Calvano Δ (global + per chain type) from windowed mean prices ───────
    deltas_by_chain: dict = {}
    if p_nash_arr is not None and p_mono_arr is not None:
        N = len(firms)
        fp = res.get("windowed_prices") or res["final_prices"]
        p_learned = np.array([float(fp.get(str(f.id), np.nan)) for f in firms])
        cts = np.array([f.chain_type for f in firms], dtype=object)

        def _delta(m):
            if m.sum() == 0:
                return float("nan")
            pl = float(np.nanmean(p_learned[m]))
            pn = float(p_nash_arr[m].mean()); pm = float(p_mono_arr[m].mean())
            d = pm - pn
            return float("nan") if abs(d) < 1e-9 else float(np.clip((pl - pn) / d, -0.5, 1.5))

        deltas_by_chain = {
            "global": _delta(np.ones(N, bool)),
            "discount": _delta(cts == "discount"),
            "standard": _delta(cts == "standard"),
            "bio": _delta(cts == "bio"),
        }

    # ── Gross profit Δ: (Σ(p−c)·D_realised − π_nash)/(π_mono − π_nash) ─────
    #    Gross variable profit (excludes effort cost). EXACT in price-only mode
    #    (effort=0 everywhere); in --with-effort mode it is a gross index, and
    #    benchmark demand is evaluated at the Bertrand-Nash effort (the same
    #    footing as the effort_fixed monopoly benchmark from Step 4). A full
    #    joint (price, effort) cartel benchmark is a documented follow-up.
    deltas_profit_by_chain: dict = {}
    if p_nash_arr is not None and p_mono_arr is not None:
        try:
            # Evaluate ALL demand vectors on the SAME path the substrate clears
            # on: market_clearing auto-dispatches to the sparse catchment kernel
            # when city.catch_indptr is not None (the simulation path), else
            # dense. Real demand is taken analytically at the windowed mean
            # prices — NOT from the sparse `windowed_demands` record — so real
            # and benchmark profits share identical footing. Mixing the dense
            # benchmark kernel with catchment sim demand was the bug
            # that pinned profit-Δ to the -0.5 clamp. Mirrors run_single_session.
            from hotelling.core.market import market_clearing as _mc_delta
            N = len(firms)
            cts = np.array([f.chain_type for f in firms], dtype=object)
            costs = np.array([f.marginal_cost for f in firms], dtype=np.float64)
            city = comp["city"]
            tc = float(config["env"].get("transport_cost", 0.01))
            e_bench = (np.asarray(e_nash_arr, dtype=np.float64)
                       if (with_effort and e_nash_arr is not None) else np.zeros(N))

            def _demand_at(prices, efforts):
                d, _ = _mc_delta(prices, efforts, city, tc)
                return d

            wp = res.get("windowed_prices", {})
            p_real = np.array([float(wp.get(str(f.id), np.nan)) for f in firms])
            if np.any(np.isnan(p_real)):
                raise ValueError("windowed prices contain NaN; profit-Δ skipped")

            pi_real = (p_real     - costs) * _demand_at(p_real,     e_bench)
            pi_nash = (p_nash_arr - costs) * _demand_at(p_nash_arr, e_bench)
            pi_mono = (p_mono_arr - costs) * _demand_at(p_mono_arr, e_bench)

            def _dpi(m):
                if m.sum() == 0:
                    return float("nan")
                rn = float(np.nansum(pi_real[m]))
                nn = float(pi_nash[m].sum())
                mn = float(pi_mono[m].sum())
                d = mn - nn
                return float("nan") if abs(d) < 1e-9 else float(np.clip((rn - nn) / d, -0.5, 1.5))

            deltas_profit_by_chain = {
                "global": _dpi(np.ones(N, bool)),
                "discount": _dpi(cts == "discount"),
                "standard": _dpi(cts == "standard"),
                "bio": _dpi(cts == "bio"),
            }
        except Exception as _e:  # noqa: BLE001
            _log.warning("Gross profit-Δ computation failed: %s", _e)
            deltas_profit_by_chain = {}

    # ── Outputs ────────────────────────────────────────────────────────────
    pd.DataFrame(res["envelope_log"]).to_parquet(output_dir / "envelopes.parquet", index=False)
    agg = pd.DataFrame({
        "step": res["step_history"], "mean_price": res["price_history"],
        "mean_effort": res["effort_history"],
    })
    for ct in ("discount", "standard", "bio"):
        col = list(res["price_history_by_chain"].get(ct, []))
        col += [float("nan")] * (len(agg) - len(col))
        agg[f"mean_price_{ct}"] = col[: len(agg)]
    agg.to_parquet(output_dir / "aggregate.parquet", index=False)

    meta = {
        "run_id": run_id, "run_name": run_name, "mode": "strategic",
        "no_ceo": no_ceo, "seed": seed, "T_burnin": T_burnin, "T_game": T_game,
        "T_CEO": T_CEO, "n_epochs": res["n_epochs"], "n_firms": len(firms),
        "ceo_model": str(ceo_cfg.get("model")), "active_divisions": active_divisions,
        "group_keys": group_keys, "deltas_by_chain": deltas_by_chain,
        "deltas_profit_by_chain": deltas_profit_by_chain,
        "profit_delta_is_gross": True,
        "state_mode": str(config["agents"].get("state_mode", "neighbors")),
        "graph_k": int(config["agents"].get("graph_k", 2)),
        "graph_n_rival_bins": int(config["agents"].get("graph_n_rival_bins", 10)),
        "graph_own_grid_type": str(comp.get("graph_own_grid_type",
                                             config["agents"].get("graph_own_grid_type", "G"))),
        "graph_rival_match": str(config["agents"].get("graph_rival_match", "A")),
        "graph_candidate_topn": int(config["agents"].get("graph_candidate_topn", 8)),
        "graph_min_edge_weight": float(config["agents"].get("graph_min_edge_weight", 0.0)),
        "chain_specific_grid": bool(config["agents"].get("chain_specific_grid", False)),
        "qtable_init": str(config.get("qtable_init", "zero")),
        "beta_schedule": str(config["agents"].get("beta_schedule", "two_stage")),
        "beta_decay_auto": bool(config["agents"].get("beta_decay_auto", True)),
        "beta1": (float(config["agents"]["beta1"])
                  if config["agents"].get("beta1") is not None else None),
        "beta2": (float(config["agents"]["beta2"])
                  if config["agents"].get("beta2") is not None else None),
        "t0_schedule": int(config["agents"].get("t0_schedule", 0)),
        "epsilon_min": float(config["agents"].get("epsilon_min", 3e-4)),
        "epsilon_transition": float(config["agents"].get("epsilon_transition", 0.10)),
        "explore_fraction": float(config["agents"].get("explore_fraction", 0.65)),
        "mean_epsilon_approx": float(config["agents"].get("mean_epsilon_approx", float("nan"))),
        "with_effort": with_effort, "with_comm": with_comm,
        "T_measure": (int(T_measure) if T_measure else int(T_CEO)),
        "ceo_temperature": float(ceo_cfg.get("temperature", 0)),
        "burnin_epsilon_mean": burn_result.get("epsilon_mean"),
        "epsilon_mean_final": res["epsilon_mean"],
        "ceo_calls_total": ceo_total,
        "ceo_calls_success": ceo_success,
        "ceo_calls_failed": ceo_fail,
        "ceo_success_rate": ceo_success_rate,
        "ceo_all_failed": ceo_all_failed,
        "ceo_errors": ceo_errors,
        "dense_log_meta": str(output_dir / "dense_log_meta.json"),
        "elapsed_s": round(time.time() - t_start, 2),
        "env_config_path": config.get("env_config_path"),
    }
    _grid_meta = env.grid_spec(None)
    _graph_meta_by_chain = {
        ct: env.graph_degree_spec(ct) for ct in ("discount", "standard", "bio")
    }
    meta.update({
        "price_grid_m": int(_grid_meta.get("m", 0)),
        "price_grid_lo": float(_grid_meta.get("lo", 0.0)),
        "price_grid_hi": float(_grid_meta.get("hi", 0.0)),
        "price_grid_step": float(_grid_meta.get("step", 0.0)),
        "price_grid_regime": str(_grid_meta.get("regime", "G")),
        "min_delta_p_effective": float(_min_delta_p_eff),
        "graph_mean_observed_rivals_by_chain": {
            ct: float(d.get("mean_observed", 0.0)) for ct, d in _graph_meta_by_chain.items()
        },
        "graph_n_isolated_by_chain": {
            ct: int(d.get("n_isolated", 0)) for ct, d in _graph_meta_by_chain.items()
        },
    })
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    index_path = out_root / "index.csv"
    row = pd.DataFrame([{
        "run_name": run_name, "run_id": run_id, "no_ceo": no_ceo, "seed": seed,
        "T_game": T_game, "T_CEO": T_CEO, "n_epochs": res["n_epochs"],
        "delta_global": deltas_by_chain.get("global"),
        "ceo_model": str(ceo_cfg.get("model")), "elapsed_s": meta["elapsed_s"],
    }])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(index_path, mode="a", header=not index_path.exists(), index=False)

    return {"run_id": run_id, "output_dir": str(output_dir),
            "deltas_by_chain": deltas_by_chain,
            "deltas_profit_by_chain": deltas_profit_by_chain,
            "ceo_success_rate": ceo_success_rate, "ceo_calls_total": ceo_total,
            "ceo_calls_success": ceo_success, "ceo_all_failed": ceo_all_failed,
            **res}
