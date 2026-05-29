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
    )

    # --- 1b. Pre-compute benchmarks and derive Calvano price grid ---
    tc = float(env_cfg.get("transport_cost", 0.01))
    benchmark_cache_pre = (
        Path(env_cfg.get("grid_path", "data/processed/demand_grid.parquet")).parent
        / "benchmarks_cache.npz"
    )
    auto_grid = bool(agent_cfg.get("auto_price_grid", True))
    p_nash_pre = p_mono_pre = None
    if auto_grid:
        p_nash_arr, _ = bertrand_nash(city, transport_cost=tc, cache_path=benchmark_cache_pre)
        p_mono_arr, _ = joint_monopoly(city, transport_cost=tc, cache_path=benchmark_cache_pre)
        p_nash_pre = float(p_nash_arr.mean())
        p_mono_pre = float(p_mono_arr.mean())
        xi = float(agent_cfg.get("price_grid_xi", 0.1))
        span = p_mono_pre - p_nash_pre
        if span > 1e-6:
            mc_min = min(getattr(f, "marginal_cost", 0.0) for f in firms)
            grid_min = max(mc_min, p_nash_pre - xi * span)
            grid_max = p_mono_pre + xi * span
        else:
            grid_min = agent_cfg.get("min_price", None)
            grid_max = agent_cfg.get("max_price", None)
    else:
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

    # Save metadata.json
    metadata = {
        "run_id": run_id,
        "run_name": run_name,
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
        "use_batch": use_batch,
    }
    with (output_dir / "metadata.json").open("w") as _f:
        json.dump(metadata, _f, indent=2)

    # Save aggregate history (step, mean_price, mean_effort) to aggregate.parquet
    agg_df = pd.DataFrame({
        "step": phase0_result.get("step_history", []),
        "mean_price": phase0_result.get("price_history", []),
        "mean_effort": phase0_result.get("effort_history", []),
    })
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
