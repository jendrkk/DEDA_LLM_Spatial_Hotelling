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
            next_obs, rewards, done, step_infos = self._step(obs, step)
            obs = next_obs
            final_obs = obs
            n_steps = step + 1

            if self.recorder is not None:
                _rec_pidxs = np.array(
                    [
                        self.env._current_joint_actions[str(firm.id)] // self.env.m_effort
                        for firm in self.env.firms
                    ],
                    dtype=np.intp,
                )
                _rec_prices = self.env.decode_prices(_rec_pidxs)
                for i, firm in enumerate(self.env.firms):
                    aid = str(firm.id)
                    joint_idx = self.env._current_joint_actions[aid]
                    p_idx = joint_idx // self.env.m_effort
                    e_idx = joint_idx % self.env.m_effort
                    step_info = step_infos.get(aid, {})
                    self.recorder.record_step(
                        period=step,
                        agent_id=aid,
                        price=float(_rec_prices[i]),
                        effort=float(self.env.effort_grid[e_idx]),
                        demand=step_info.get("demand", float("nan")),
                        profit=float(rewards.get(aid, float("nan"))),
                        price_idx=p_idx,
                        effort_idx=e_idx,
                    )

            # Sample price/effort history at regular intervals (convergence monitoring)
            if (step + 1) % self.record_every == 0:
                _pidxs = np.array([
                    self.env._current_joint_actions[aid] // self.env.m_effort
                    for aid in self.env.agents
                ], dtype=np.intp)
                prices = self.env.decode_prices(_pidxs).tolist()
                efforts = [
                    self.env.effort_grid[
                        self.env._current_joint_actions[aid] % self.env.m_effort
                    ]
                    for aid in self.env.agents
                ]
                self._price_history.append(float(np.mean(prices)))
                self._effort_history.append(float(np.mean(efforts)))
                self._step_history.append(step + 1)

            if done:
                break

        # --- Final prices per agent ---
        _final_pidxs = np.array([
            self.env._current_joint_actions.get(aid, 0) // self.env.m_effort
            for aid in self.env.possible_agents
        ], dtype=np.intp)
        _final_eur = self.env.decode_prices(_final_pidxs)
        final_prices = {
            aid: float(_final_eur[i])
            for i, aid in enumerate(self.env.possible_agents)
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
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """Execute one environment step.

        Collects actions from all agents, steps the environment, builds
        transition dicts, and calls agent.update() for each agent.

        Parameters
        ----------
        observations : dict mapping agent_id → observation dict
        step : current 0-based step index (for logging)

        Returns
        -------
        tuple (next_observations, rewards, done, infos) where done is True when
        all agents are terminated or truncated, and infos maps agent_id to
        per-step metrics (demand, price, effort).
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

        return next_observations, rewards, done, infos


class BatchSimulationEngine:
    """Vectorized simulation engine using a single BatchQLearningAgent.

    Parameters
    ----------
    env : HotellingMarketEnv
    batch_agent : BatchQLearningAgent
    max_steps : maximum environment steps
    recorder : optional SimulationRecorder
    record_every : sparse price-history sampling interval
    dense_log : optional dense step logger (PROMPT 4); None disables
    """

    def __init__(
        self,
        env: Any,
        batch_agent: Any,
        max_steps: int = 1_000_000,
        recorder: Optional[Any] = None,
        record_every: int = 1_000,
        dense_log: Optional[Any] = None,
    ) -> None:
        self.env = env
        self.batch_agent = batch_agent
        self.max_steps = max_steps
        self.recorder = recorder
        self.record_every = record_every
        self.dense_log = dense_log

        self._price_history: List[float] = []
        self._effort_history: List[float] = []
        self._step_history: List[int] = []
        self._chain_masks = {
            ct: np.array(
                [getattr(f, "chain_type", None) == ct for f in self.env.firms],
                dtype=bool,
            )
            for ct in ("discount", "standard", "bio")
        }
        self._price_history_by_chain: Dict[str, List[float]] = {
            ct: [] for ct in ("discount", "standard", "bio")
        }

    def run(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Execute a full batch simulation session."""
        self._price_history.clear()
        self._effort_history.clear()
        self._step_history.clear()
        for ct in self._price_history_by_chain:
            self._price_history_by_chain[ct].clear()

        _obs, _infos = self.env.reset(seed=seed)
        self.batch_agent.reset()
        state_signal = self.env.current_state_signal()

        n_steps = 0
        for step in range(self.max_steps):
            state_signal, _rewards_arr, done = self._batch_step(
                state_signal, step
            )
            n_steps = step + 1
            if done:
                break

        # Read final prices from the array state (dict is stale on the array path)
        final_pidxs = self.env._current_joint_actions_arr // self.env.m_effort
        final_prices_arr = self.env.decode_prices(final_pidxs)
        final_prices = {
            self.env.possible_agents[i]: float(final_prices_arr[i])
            for i in range(len(self.env.possible_agents))
        }

        if self.recorder is not None:
            self.recorder.flush()
        if self.dense_log is not None:
            self.dense_log.flush()

        result: Dict[str, Any] = {
            "n_steps": n_steps,
            "final_prices": final_prices,
            "price_history": list(self._price_history),
            "effort_history": list(self._effort_history),
            "step_history": list(self._step_history),
            "price_history_by_chain": {
                ct: list(v) for ct, v in self._price_history_by_chain.items()
            },
        }
        if hasattr(self.batch_agent, "epsilon_mean"):
            result["epsilon_mean"] = self.batch_agent.epsilon_mean
        return result

    def _obs_dict_to_matrix(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Convert observation dicts to (N, k) neighbor action indices.

        Retained for external callers that still use the dict API; not used
        in the hot loop.  Prefer :meth:`~hotelling.env.market_env.HotellingMarketEnv.get_neighbor_actions_arr`
        for the array path.
        """
        return np.array(
            [
                obs_dict[str(f.id)]["neighbor_prev_actions"]
                for f in self.env.firms
            ],
            dtype=np.int64,
        )

    def _batch_step(
        self,
        state_signal: np.ndarray,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Execute one vectorized environment step via the array hot path.

        Calls ``env.step_array`` directly — no action dict construction and no
        dict reparse of rewards or demands.

        Parameters
        ----------
        state_signal : (N, k) int64 in neighbors mode, or (N,) int64 in local_summary
        step : int
            Current 0-based step index.

        Returns
        -------
        next_signal : same shape/dtype as state_signal
        rewards_arr : (N,) float64
        done : bool — always False (firms never terminate mid-episode).
        """
        actions = self.batch_agent.act(state_signal)

        # Single env call: returns arrays directly, no dict overhead
        _next_neighbor_actions, rewards_arr, demands_arr = self.env.step_array(actions)

        next_signal = self.env.current_state_signal()
        # For hybrid_profit_gap: store current rewards as t−1 profits for the NEXT
        # state signal call.  Must happen AFTER current_state_signal() (which reads
        # _prev_rewards_arr as t−1) and BEFORE the next act() call.
        if self.env.state_mode == "hybrid_profit_gap":
            self.env.update_prev_rewards_for_hybrid(rewards_arr)
        states = self.batch_agent._encode_states(state_signal)
        next_states = self.batch_agent._encode_states(next_signal)
        self.batch_agent.update(states, actions, rewards_arr, next_states)

        price_idxs = actions // self.env.m_effort
        effort_idxs = actions % self.env.m_effort
        prices_arr = self.env.decode_prices(price_idxs)
        efforts_arr = self.env.effort_grid[effort_idxs]

        if self.dense_log is not None:
            self.dense_log.write_step(
                step,
                price_idxs,
                effort_idxs,
                demands_arr,
                rewards_arr,  # profits == rewards
            )

        if self.recorder is not None:
            for i, firm in enumerate(self.env.firms):
                aid = str(firm.id)
                self.recorder.record_step(
                    period=step,
                    agent_id=aid,
                    price=float(prices_arr[i]),
                    effort=float(efforts_arr[i]),
                    demand=float(demands_arr[i]),
                    profit=float(rewards_arr[i]),
                    price_idx=int(price_idxs[i]),
                    effort_idx=int(effort_idxs[i]),
                )

        if (step + 1) % self.record_every == 0:
            self._price_history.append(float(prices_arr.mean()))
            self._effort_history.append(float(efforts_arr.mean()))
            self._step_history.append(step + 1)
            for ct, mask in self._chain_masks.items():
                if mask.any():
                    self._price_history_by_chain[ct].append(
                        float(prices_arr[mask].mean())
                    )
                else:
                    self._price_history_by_chain[ct].append(float("nan"))

        # Firms never terminate mid-episode in this model
        done = len(self.env.agents) == 0

        return next_signal, rewards_arr, done
