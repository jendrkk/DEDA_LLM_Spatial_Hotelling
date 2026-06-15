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

from pathlib import Path
from typing import Any, Dict, List, Optional


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
    )

    use_batch = bool(agent_cfg.get("use_batch", True))
    agents: Dict[str, Any] | None = None
    batch_agent = None

    if use_batch:
        from hotelling.agents.batch_qlearning import BatchQLearningAgent

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

    # --- 4. Run Phase 0 burn-in ---
    T_burnin = int(phase0_cfg.get("T_burnin", 1_000_000))
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

    deltas_by_chain: Dict[str, float] = {}
    chain_price_table: Dict[str, Any] = {}
    realized_outside_share = float("nan")
    realized_chain_shares: Dict[str, float] = {}

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

    phase0_result["deltas_by_chain"] = deltas_by_chain
    phase0_result["chain_price_table"] = chain_price_table
    phase0_result["realized_outside_share"] = realized_outside_share
    phase0_result["realized_chain_shares"] = realized_chain_shares

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
        "deltas_by_chain": deltas_by_chain,
        "chain_price_table": chain_price_table,
        "realized_outside_share": realized_outside_share,
        "realized_chain_shares": realized_chain_shares,
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


def _build_components(config: dict) -> dict:
    """Build City, Firms, env, warmed-ready batch agent, benchmarks, union grid.

    Mirrors run_single_session's construction (kept separate so the Phase-0
    baseline stays untouched). Returns a dict of components for the strategic run.
    """
    from pathlib import Path

    import geopandas as gpd
    import numpy as np

    from hotelling.agents.batch_qlearning import BatchQLearningAgent
    from hotelling.core.equilibrium import bertrand_nash, joint_monopoly
    from hotelling.env.market_env import HotellingMarketEnv
    from hotelling.spatial.loader import load_berlin_city

    env_cfg = config["env"]
    agent_cfg = config["agents"]
    seed = config["phase2"].get("seed", None)

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
    grid_min = agent_cfg.get("min_price", None)
    grid_max = agent_cfg.get("max_price", None)

    if auto_grid and city.dist2_km2 is not None:
        p_nash_arr, _ = bertrand_nash(city, transport_cost=tc, cache_path=bench_cache)
        p_mono_arr, _ = joint_monopoly(city, transport_cost=tc, cache_path=bench_cache)
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
    )
    batch_agent = BatchQLearningAgent(
        n_agents=len(firms), m=int(agent_cfg.get("m", 25)),
        m_effort=int(agent_cfg.get("m_effort", 1)), k=int(agent_cfg.get("k_neighbors", 1)),
        alpha=float(agent_cfg.get("alpha_lr", 0.15)),
        beta_decay=float(agent_cfg.get("beta_decay", 4e-6)),
        delta=float(agent_cfg.get("delta", 0.95)),
        seed=int(seed) if seed is not None else None,
        state_mode=str(agent_cfg.get("state_mode", "neighbors")),
        state_size=env.state_size,
    )
    grid_gdf = gpd.read_parquet(env_cfg.get("grid_path", "data/processed/demand_grid.parquet"))
    return {
        "city": city, "firms": firms, "env": env, "batch_agent": batch_agent,
        "p_nash_arr": p_nash_arr, "p_mono_arr": p_mono_arr,
        "grid_min": grid_min, "grid_max": grid_max, "grid_gdf": grid_gdf,
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

    comp = _build_components(config)
    env, batch_agent, firms = comp["env"], comp["batch_agent"], comp["firms"]
    p_nash_arr, p_mono_arr = comp["p_nash_arr"], comp["p_mono_arr"]

    phase2_cfg = config["phase2"]
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
    ceos = build_chain_ceos(
        firms, client=client, active_divisions=active_divisions,
        division_params=division_params, group_keys=group_keys,
        min_delta_p=float(ceo_cfg.get("min_delta_p", 1.5)),
        min_delta_e=float(ceo_cfg.get("min_delta_e", 0.1)), T_ceo=T_CEO,
        merge_system=bool(ceo_cfg.get("merge_system_prompt", True)),
        capture_comm=save_comm,
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
        store_demand_profit=bool(phase2_cfg.get("store_demand_profit", True)),
        float_dtype=str(phase2_cfg.get("float_dtype", "float32")),
        dense_stride=int(phase2_cfg.get("dense_stride", 1)),
        dense_tail=(int(phase2_cfg["dense_tail"]) if phase2_cfg.get("dense_tail") is not None else None),
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

    # ── Calvano Δ (global + per chain type) from final prices ──────────────
    deltas_by_chain: dict = {}
    if p_nash_arr is not None and p_mono_arr is not None:
        N = len(firms)
        fp = res["final_prices"]
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
            "ceo_success_rate": ceo_success_rate, "ceo_calls_total": ceo_total,
            "ceo_calls_success": ceo_success, "ceo_all_failed": ceo_all_failed,
            **res}
