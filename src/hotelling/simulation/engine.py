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
        env: HotellingMarketEnv,
        agents: Dict[str, Any],
        max_steps: int = 1_000_000,
        recorder: Optional[Any] = None,
    ) -> None:
        self.env = env
        self.agents = agents
        self.max_steps = max_steps
        self.recorder = recorder

    def run(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Execute a full simulation session.

        Resets the environment and all agents, then steps until max_steps or
        episode termination.  If a recorder is attached, flushes Parquet at
        the end.

        Parameters
        ----------
        seed : random seed forwarded to env.reset()

        Returns
        -------
        dict with keys: n_steps (int), final_prices (dict), summary (dict)
        """
        raise NotImplementedError

    def _step(
        self,
        observations: Dict[str, Any],
        step: int,
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
        """Execute one environment step.

        Parameters
        ----------
        observations : dict mapping agent_id -> observation array
        step : current step index (0-based)

        Returns
        -------
        next_observations, rewards, done
        """
        raise NotImplementedError
