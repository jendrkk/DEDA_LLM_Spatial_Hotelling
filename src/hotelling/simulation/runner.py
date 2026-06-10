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
    if auto_grid and _can_run_benchmarks:
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
        "use_batch": use_batch,
        "state_mode": agent_cfg.get("state_mode", "neighbors"),
        "local_sum_n": agent_cfg.get("local_sum_n", None),
        "n_price_bins": agent_cfg.get("n_price_bins", 15),
        "summary_stats": agent_cfg.get("summary_stats", ["mean"]),
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
