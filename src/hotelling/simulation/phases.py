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
            if float(np.std(window_prices)) < convergence_threshold:
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
    """Strategic game phase: CEOs + entrant reassess; stores learn tactically.

    Parameters
    ----------
    config : dict  Phase 2 config slice (T_game, T_CEO, T_entrant).
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(
        self,
        stores: list,
        ceos: list,
        entrant: EntrantLLM,
        env: object,
    ) -> dict:
        """Run the strategic game; return per-period history dict."""
        raise NotImplementedError
