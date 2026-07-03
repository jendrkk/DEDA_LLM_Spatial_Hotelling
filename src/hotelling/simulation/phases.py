"""Three-phase simulation controller stubs.

Responsibility: orchestrate Phase 0 (burn-in), Phase 1 (entry), and
Phase 2 (strategic game), delegating to stores, CEOs, and the entrant.

Public API: Phase0BurnIn, Phase1Entry, Phase2StrategicGame

Key dependencies: hotelling.agents, hotelling.llm.schemas

References: ADR-006; docs/agent_simulation_technical_report.md §8.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hotelling.agents.entrant_llm import EntrantLLM
from hotelling.llm.schemas import EntrantEntryDecision

logger = logging.getLogger(__name__)


def _fmt_envelope(g, with_effort: bool = True) -> str:
    if with_effort:
        return (f"p={g.p_bar:.2f} dp={g.delta_p:.2f} e={g.e_bar:.2f} "
                f"de={g.delta_e:.2f} eps={g.epsilon:.3f}")
    return f"p={g.p_bar:.2f} dp={g.delta_p:.2f} eps={g.epsilon:.3f}"


def _log_ceo_io(brand: str, ctype: str, epoch: int, st: dict, out,
                failed: bool, with_effort: bool = True) -> None:
    own = st.get("own", {}) or {}
    status = "RETAINED (call failed)" if failed else "OK"
    chain_p = own.get("mean_price_last_T", float("nan"))
    gp = own.get("group_performance", []) or []
    if gp:
        in_groups = "  ".join(f"{g['group_key']}: p={g['mean_price']:.2f}" for g in gp)
    else:
        in_groups = "(single group)"
    lines = [f"CEO {brand} [{ctype}] epoch {epoch} — {status}",
             f"   in : chain mean p={chain_p:.2f} EUR  | groups: {in_groups}"]
    cs = getattr(out, "coordination_signal", None)
    if cs is not None:
        lines.append(
            f"   sig: willing={cs.willing} proposed={cs.proposed_tier_price:.2f} EUR"
        )
    for k, g in out.groups.items():
        lines.append(f"   out: {k} -> {_fmt_envelope(g, with_effort)}")
    logger.info("\n".join(lines))


class Phase0BurnIn:
    """Burn-in phase: incumbent Q-learners converge without CEO or entrant.

    Parameters
    ----------
    config : dict  Phase 0 config slice (T_burnin, convergence threshold).
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(
        self,
        agents: dict | None = None,
        env: object = None,
        city: object = None,
        transport_cost: float = 0.0,
        seed: int | None = None,
        batch_agent: object | None = None,
    ) -> dict:
        """Run Q-learning burn-in until convergence; return statistics.

        Creates a SimulationEngine from agents and env, runs up to T_burnin
        steps, checks for price convergence every check_interval steps, and
        computes the Calvano collusion index Δ against Bertrand-Nash and
        joint-monopoly benchmarks.

        Convergence criterion: rolling standard deviation of sampled mean
        market price over the last convergence_window sample points falls
        below convergence_threshold.

        Parameters
        ----------
        agents : dict mapping agent_id str → QLearningAgent
        env : HotellingMarketEnv instance (already constructed, not yet reset)
        city : City instance (needed for Bertrand-Nash and monopoly benchmarks)
        transport_cost : float transport cost parameter (passed to benchmarks)
        seed : random seed for env.reset()

        Returns
        -------
        dict with keys:
            converged          bool
            n_steps            int
            delta              float  — Calvano collusion index Δ ∈ [0, 1]
            mean_final_price   float  — mean price over last convergence_window samples
            p_nash             float  — mean Bertrand-Nash price
            p_mono             float  — mean joint-monopoly price
            price_history      list   — sampled mean prices (one per record_every steps)
            effort_history     list   — sampled mean efforts
            step_history       list   — step indices for samples
        """
        from hotelling.core.equilibrium import bertrand_nash, joint_monopoly

        cfg = self.config
        T_burnin = int(cfg.get("T_burnin", 1_000_000))
        convergence_window = int(cfg.get("convergence_window", 100))
        convergence_threshold = float(cfg.get("convergence_threshold", 0.01))
        convergence_relative = bool(cfg.get("convergence_relative", True))
        check_interval = int(cfg.get("check_interval", 1_000))
        record_every = int(cfg.get("record_every", check_interval))

        batch_agent = batch_agent or cfg.get("_batch_agent")
        if batch_agent is not None:
            from hotelling.simulation.engine import BatchSimulationEngine

            engine = BatchSimulationEngine(
                env=env,
                batch_agent=batch_agent,
                max_steps=T_burnin,
                record_every=record_every,
                recorder=cfg.get("_recorder", None),
                dense_log=cfg.get("_dense_log"),
            )
        else:
            from hotelling.simulation.engine import SimulationEngine

            if agents is None:
                raise ValueError("agents dict required when batch_agent is not set")
            engine = SimulationEngine(
                env=env,
                agents=agents,
                max_steps=T_burnin,
                record_every=record_every,
                recorder=cfg.get("_recorder", None),
            )

        result = engine.run(seed=seed)

        price_history = result["price_history"]
        converged = False
        if len(price_history) >= convergence_window:
            window_prices = price_history[-convergence_window:]
            std = float(np.std(window_prices))
            mean = float(np.mean(window_prices))
            metric = (
                std / abs(mean)
                if (convergence_relative and abs(mean) > 1e-12)
                else std
            )
            if metric < convergence_threshold:
                converged = True

        mean_final_price = (
            float(np.mean(list(result["final_prices"].values())))
            if result["final_prices"]
            else 0.0
        )

        # --- Compute Bertrand-Nash and joint-monopoly benchmarks ---
        cache_path = cfg.get("benchmark_cache_path", None)
        if cache_path is not None:
            cache_path = Path(cache_path)
        p_nash = cfg.get("p_nash_precomputed", None)
        p_mono = cfg.get("p_mono_precomputed", None)
        if p_nash is None or p_mono is None:
            try:
                p_nash_arr, _ = bertrand_nash(
                    city,
                    transport_cost=transport_cost,
                    cache_path=cache_path,
                )
                p_mono_arr, _ = joint_monopoly(
                    city,
                    transport_cost=transport_cost,
                    cache_path=cache_path,
                )
                p_nash = float(p_nash_arr.mean())
                p_mono = float(p_mono_arr.mean())
            except Exception as exc:
                logger.warning("Could not compute benchmarks: %s", exc)
                p_nash = 0.0
                p_mono = mean_final_price + 1e-9  # avoid division by zero

        # Calvano Δ = (p_mean - p_Nash) / (p_Monopoly - p_Nash)
        denom = p_mono - p_nash
        if abs(denom) < 1e-9:
            delta = 0.0
            logger.warning("Monopoly and Nash prices are equal; Δ set to 0.")
        else:
            delta = float(np.clip((mean_final_price - p_nash) / denom, -0.5, 1.5))

        logger.info(
            "Phase0 complete: converged=%s, n_steps=%d, Δ=%.4f, "
            "p_mean=%.4f, p_nash=%.4f, p_mono=%.4f.",
            converged,
            result["n_steps"],
            delta,
            mean_final_price,
            p_nash,
            p_mono,
        )

        out = {
            "converged": converged,
            "n_steps": result["n_steps"],
            "delta": delta,
            "mean_final_price": mean_final_price,
            "p_nash": p_nash,
            "p_mono": p_mono,
            "price_history": price_history,
            "effort_history": result["effort_history"],
            "step_history": result["step_history"],
            "final_prices": result["final_prices"],
        }
        if "epsilon_mean" in result:
            out["epsilon_mean"] = result["epsilon_mean"]
        out["price_history_by_chain"] = result.get("price_history_by_chain", {})
        return out


class Phase1Entry:
    """Entry phase: entrant LLM makes the one-shot entry decision.

    Parameters
    ----------
    config : dict  Phase 1 config slice.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self, entrant: EntrantLLM, market_state: dict) -> EntrantEntryDecision:
        """Invoke the entrant entry decision and return the validated output."""
        raise NotImplementedError


class Phase2StrategicGame:
    """CEO-only strategic game on a warmed batch agent (no entrant in v1).

    Every T_CEO periods each chain's CEO is called sequentially (ADR-007) to set
    a per-group envelope; the envelopes are compiled into a per-store action mask
    and epsilon vector applied to the batch agent for the next T_CEO periods.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(
        self,
        *,
        env,
        batch_agent,
        ceos: dict,
        store_chain: list,
        store_chain_type: list,
        store_group_labels: list,
        group_keys: list,
        zones: dict,
        T_game: int,
        T_CEO: int,
        mask_effort: bool,
        no_ceo: bool = False,
        record_every: int = 100,
        dense_log=None,
        store_metadata=None,
        enrich_groups: bool = False,
        with_effort: bool = True,
        with_comm: bool = False,
        T_measure: int | None = None,
        p_nash_arr=None,
        p_mono_arr=None,
        strategic_analytics: bool = False,
        tier_commit: bool = False,
    ) -> dict:
        import numpy as np

        from hotelling.envelope.masking import build_action_mask_and_epsilon
        from hotelling.llm.ceo_state import RollingWindow, build_ceo_state

        N = len(env.firms)
        m_effort = int(env.m_effort)
        win_len = int(T_measure) if T_measure else int(T_CEO)
        window = RollingWindow(N, max(1, win_len))

        ceo_history = {b: [] for b in ceos}
        prev_env = {b: None for b in ceos}
        current_signals: dict = {}          # brand -> {"willing", "proposed_tier_price"}
        chain_envelopes: dict = {}
        envelope_log: list[dict] = []
        decision_log: list[dict] = []

        ct_arr = np.array(store_chain_type, dtype=object)
        chain_masks = {ct: (ct_arr == ct) for ct in ("discount", "standard", "bio")}
        price_history, effort_history, step_history = [], [], []
        price_history_by_chain = {ct: [] for ct in ("discount", "standard", "bio")}

        state_signal = env.current_state_signal()
        epoch = 0

        for t in range(int(T_game)):
            actions = batch_agent.act(state_signal)
            _nbr, rewards, demands = env.step_array(actions)
            next_signal = env.current_state_signal()
            states = batch_agent._encode_states(state_signal)
            next_states = batch_agent._encode_states(next_signal)
            batch_agent.update(states, actions, rewards, next_states)
            state_signal = next_signal

            p_idx = actions // m_effort
            e_idx = actions % m_effort
            prices = env.decode_prices(p_idx)
            efforts = env.effort_grid[e_idx]
            window.push(prices, efforts, demands, rewards)

            if dense_log is not None:
                dense_log.write_step(t, p_idx, e_idx, demands, rewards)

            if (t + 1) % record_every == 0:
                price_history.append(float(prices.mean()))
                effort_history.append(float(efforts.mean()))
                step_history.append(t + 1)
                for ct, mask in chain_masks.items():
                    price_history_by_chain[ct].append(
                        float(prices[mask].mean()) if mask.any() else float("nan")
                    )

            if (not no_ceo) and ((t + 1) % T_CEO == 0):
                signals_prev = dict(current_signals)
                analytics_by_brand: dict = {}
                if strategic_analytics and p_nash_arr is not None and p_mono_arr is not None:
                    from hotelling.llm.ceo_analytics import compute_strategic_analytics
                    _win_now = window.arrays()
                    if _win_now["price"].size:
                        _cur_prices = _win_now["price"].mean(axis=0)
                    else:
                        _cur_prices = env.decode_prices(
                            env._current_joint_actions_arr // m_effort
                        )
                    _mc_arr = np.array(
                        [f.marginal_cost for f in env.firms], dtype=np.float64
                    )
                    analytics_by_brand = compute_strategic_analytics(
                        env=env, current_prices=_cur_prices,
                        store_chain=store_chain, store_chain_type=store_chain_type,
                        p_nash_arr=p_nash_arr, p_mono_arr=p_mono_arr,
                        marginal_costs=_mc_arr,
                    )
                for brand, ceo in ceos.items():
                    st = build_ceo_state(
                        window, chain_id=brand, store_chain=store_chain,
                        store_chain_type=store_chain_type,
                        store_group_labels=store_group_labels,
                        group_keys=group_keys, zones=zones,
                        history=ceo_history[brand], epoch=epoch, T_ceo=T_CEO,
                        marginal_cost=ceo.marginal_cost,
                        min_delta_p=ceo.min_delta_p, min_delta_e=ceo.min_delta_e,
                        store_metadata=store_metadata, enrich_groups=enrich_groups,
                        with_effort=with_effort, with_comm=with_comm,
                        signals_last_epoch=signals_prev,
                        own_last_signal=signals_prev.get(brand),
                        strategic_analytics=analytics_by_brand.get(brand),
                    )
                    nf_before = ceo.n_fail
                    out = ceo.decide(st, epoch, prev_env[brand])
                    _log_ceo_io(brand, ceo.chain_type, epoch, st, out,
                                failed=(ceo.n_fail > nf_before), with_effort=with_effort)

                    sig = None
                    if getattr(out, "coordination_signal", None) is not None:
                        sig = {
                            "willing": bool(out.coordination_signal.willing),
                            "proposed_tier_price": float(
                                out.coordination_signal.proposed_tier_price
                            ),
                        }
                    if with_comm and sig is not None:
                        current_signals[brand] = sig

                    decision_log.append({
                        "epoch": epoch, "chain": brand,
                        "n_groups": len(out.groups),
                        "rationale": out.rationale,
                        "coordination_signal": sig,
                    })
                    prev_env[brand] = out
                    chain_envelopes[brand] = out
                    ceo_history[brand].append({
                        "epoch": epoch,
                        "envelopes": {
                            k: {"p_bar": g.p_bar,
                                "dp_minus": g.dp_minus, "dp_plus": g.dp_plus,
                                "delta_p": g.delta_p,
                                "e_bar": g.e_bar, "delta_e": g.delta_e,
                                "epsilon": g.epsilon}
                            for k, g in out.groups.items()
                        },
                        "profit_realized": st["own"]["total_profit_last_T"],
                        "signal": sig,
                    })
                    for k, g in out.groups.items():
                        envelope_log.append({
                            "epoch": epoch, "step": t + 1, "chain": brand, "group": k,
                            "p_bar": g.p_bar,
                            "dp_minus": g.dp_minus, "dp_plus": g.dp_plus,
                            "delta_p": g.delta_p,
                            "e_bar": g.e_bar, "delta_e": g.delta_e, "epsilon": g.epsilon,
                        })
                tier_floor_idx = None
                if tier_commit and with_comm and current_signals:
                    _cur_pidx = env._current_joint_actions_arr // m_effort
                    _cur_prices = env.decode_prices(_cur_pidx)
                    _sg = getattr(env, "_store_price_grids", None)
                    tier_floor_idx = np.zeros(N, dtype=np.int64)
                    for _ct, _ct_mask in chain_masks.items():
                        if not _ct_mask.any():
                            continue
                        _idx_ct = np.nonzero(_ct_mask)[0]
                        _brands_ct = sorted({store_chain[int(j)] for j in _idx_ct})
                        _sigs = [current_signals.get(b) for b in _brands_ct]
                        _cur_type_price = float(_cur_prices[_ct_mask].mean())
                        _all_willing = bool(_sigs) and all(
                            (s is not None and s.get("willing")
                             and float(s.get("proposed_tier_price", -1.0)) >= _cur_type_price)
                            for s in _sigs
                        )
                        if not _all_willing:
                            continue
                        _tier_price = min(float(s["proposed_tier_price"]) for s in _sigs)
                        for j in _idx_ct:
                            _g = _sg[int(j)] if _sg is not None else env.price_grid
                            _below = np.nonzero(np.asarray(_g) <= _tier_price + 1e-9)[0]
                            tier_floor_idx[int(j)] = int(_below.max()) if _below.size else 0
                mask, eps = build_action_mask_and_epsilon(
                    chain_envelopes, store_chain, store_group_labels,
                    env.price_grid, env.effort_grid, m_effort, mask_effort,
                    store_price_grids=getattr(env, '_store_price_grids', None),
                    tier_floor_idx=tier_floor_idx,
                )
                batch_agent.set_action_mask(mask)
                batch_agent.set_epsilon_override(eps)
                epoch += 1

        final_pidx = env._current_joint_actions_arr // m_effort
        final_prices_arr = env.decode_prices(final_pidx)
        final_prices = {
            str(env.firms[i].id): float(final_prices_arr[i])
            for i in range(N)
        }

        # Windowed (post-settling) per-store mean price and demand over the
        # measurement window — used for a low-variance Calvano Δ (Step 5),
        # instead of the single-step final_prices snapshot.
        win = window.arrays()
        wp, wd = win["price"], win["demand"]
        win_price = wp.mean(axis=0) if wp.size else np.full(N, float("nan"))
        win_demand = wd.mean(axis=0) if wd.size else np.zeros(N)
        windowed_prices = {
            str(env.firms[i].id): float(win_price[i]) for i in range(N)
        }
        windowed_demands = {
            str(env.firms[i].id): float(win_demand[i]) for i in range(N)
        }

        return {
            "n_steps": int(T_game),
            "n_epochs": epoch,
            "final_prices": final_prices,
            "windowed_prices": windowed_prices,
            "windowed_demands": windowed_demands,
            "price_history": price_history,
            "effort_history": effort_history,
            "step_history": step_history,
            "price_history_by_chain": price_history_by_chain,
            "envelope_log": envelope_log,
            "decision_log": decision_log,
            "epsilon_mean": float(batch_agent.epsilon_mean),
        }
