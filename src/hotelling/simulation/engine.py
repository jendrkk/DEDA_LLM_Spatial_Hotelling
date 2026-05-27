"""SimulationEngine: orchestrate reset/step/log for Hotelling competition.

Responsibility: run the main simulation loop over a HotellingMarketEnv,
dispatch observations to agents, collect actions, step the environment,
feed transitions back to agents for learning, and delegate recording to a
SimulationRecorder.

Public API: SimulationEngine

Key dependencies: hotelling.env.market_env, hotelling.agents.base,
    hotelling.simulation.recorder

References:
    Calvano et al. (2020 AER) §III - simulation protocol;
    PettingZoo (Terry et al. 2021) - environment stepping pattern.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hotelling.env.market_env import HotellingMarketEnv
from hotelling.utils.logging import get_logger

logger = get_logger(__name__)


class SimulationEngine:
    """Orchestrates reset/step/log for the Hotelling market environment.

    Parameters
    ----------
    env : HotellingMarketEnv - the multi-agent environment
    agents : dict mapping agent_id -> AgentProtocol instance
    max_steps : maximum number of environment steps per session
    recorder : optional SimulationRecorder for Parquet + MLflow logging
    """

    def __init__(
        self,
        env: Any,
        agents: Dict[str, Any],
        max_steps: int = 1_000_000,
        recorder: Optional[Any] = None,
        record_every: int = 1_000,
    ) -> None:
        self.env = env
        self.agents = agents
        self.max_steps = max_steps
        self.recorder = recorder
        self.record_every = record_every

        # Internal history: list of (step, mean_price, mean_effort)
        # appended every record_every steps, used for convergence detection
        self._price_history: List[float] = []
        self._effort_history: List[float] = []
        self._step_history: List[int] = []

    def run(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Execute a full simulation session.

        Resets the environment and all agents, then steps for max_steps periods
        or until all agents terminate. Sampled price/effort history is recorded
        every record_every steps for convergence monitoring.

        Parameters
        ----------
        seed : random seed forwarded to env.reset()

        Returns
        -------
        dict with keys:
            n_steps         int   — number of steps completed
            final_prices    dict  — final mean price per agent {agent_id: float}
            price_history   list  — mean market price at each sample point
            effort_history  list  — mean market effort at each sample point
            step_history    list  — step indices of each sample point
        """
        # --- Reset ---
        self._price_history.clear()
        self._effort_history.clear()
        self._step_history.clear()

        obs, infos = self.env.reset(seed=seed)

        for agent_id, agent in self.agents.items():
            agent.reset(infos.get(agent_id, {}))

        # --- Main loop ---
        final_obs = obs
        n_steps = 0

        for step in range(self.max_steps):
            next_obs, rewards, done = self._step(obs, step)
            obs = next_obs
            final_obs = obs
            n_steps = step + 1

            # Sample price/effort history at regular intervals
            if (step + 1) % self.record_every == 0:
                # Extract current prices and efforts from env.infos (last step)
                # Fallback: use observation's own_prev_action if infos unavailable
                prices = [
                    self.env.price_grid[
                        self.env._current_joint_actions[aid] // self.env.m_effort
                    ]
                    for aid in self.env.agents
                ]
                efforts = [
                    self.env.effort_grid[
                        self.env._current_joint_actions[aid] % self.env.m_effort
                    ]
                    for aid in self.env.agents
                ]
                self._price_history.append(float(np.mean(prices)))
                self._effort_history.append(float(np.mean(efforts)))
                self._step_history.append(step + 1)

                # Record per-agent data via recorder when one is attached
                if self.recorder is not None:
                    current_step = step + 1
                    for i, firm in enumerate(self.env.firms):
                        aid = str(firm.id)
                        joint_idx = self.env._current_joint_actions.get(aid, 0)
                        price_idx = joint_idx // self.env.m_effort
                        effort_idx = joint_idx % self.env.m_effort
                        p_val = float(self.env.price_grid[price_idx])
                        e_val = float(self.env.effort_grid[effort_idx])
                        # Retrieve demand and profit from last step's infos if available
                        # They are not directly stored; recompute from market_clearing
                        # is expensive per-sample. Instead, tag them as NaN here and let
                        # the caller pass an infos_cache if needed.
                        # For the baseline: only price and effort are recorded per sample;
                        # demand and profit are aggregated at the chain level separately.
                        self.recorder.record_step(
                            period=current_step,
                            agent_id=aid,
                            price=p_val,
                            effort=e_val,
                            demand=float("nan"),   # not recomputed per sample; see note above
                            profit=float("nan"),
                            chain=getattr(firm, "chain", None) or "",
                            chain_type=getattr(firm, "chain_type", None) or "",
                            price_idx=price_idx,
                            effort_idx=effort_idx,
                        )

            if done:
                break

        # --- Final prices per agent ---
        final_prices = {
            aid: float(
                self.env.price_grid[
                    self.env._current_joint_actions.get(aid, 0) // self.env.m_effort
                ]
            )
            for aid in self.env.possible_agents
        }

        if self.recorder is not None:
            self.recorder.flush()

        return {
            "n_steps": n_steps,
            "final_prices": final_prices,
            "price_history": list(self._price_history),
            "effort_history": list(self._effort_history),
            "step_history": list(self._step_history),
        }

    def _step(
        self,
        observations: Dict[str, Any],
        step: int,
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
        """Execute one environment step.

        Collects actions from all agents, steps the environment, builds
        transition dicts, and calls agent.update() for each agent.

        Parameters
        ----------
        observations : dict mapping agent_id → observation dict
        step : current 0-based step index (for logging)

        Returns
        -------
        tuple (next_observations, rewards, done) where done is True when
        all agents are terminated or truncated.
        """
        # 1. Collect actions from all agents
        actions: Dict[str, Any] = {}
        for agent_id, agent in self.agents.items():
            obs = observations.get(agent_id, {})
            actions[agent_id] = agent.act(obs)

        # 2. Step the environment
        next_observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # 3. Build transitions and update agents
        for agent_id, agent in self.agents.items():
            obs = observations.get(agent_id, {})
            next_obs = next_observations.get(agent_id, {})
            transition = {
                "observation": obs,
                "action": actions[agent_id],
                "reward": float(rewards.get(agent_id, 0.0)),
                "next_observation": next_obs,
            }
            agent.update(transition)

        # 4. Done when all agents are terminated or truncated
        done = (
            all(
                terminations.get(a, False) or truncations.get(a, False)
                for a in self.env.agents
            )
            if self.env.agents
            else True
        )

        return next_observations, rewards, done
