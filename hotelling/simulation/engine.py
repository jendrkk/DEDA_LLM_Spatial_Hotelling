"""
Simulation engine for the 2-D Hotelling model.

The :class:`SimulationEngine` orchestrates the main simulation loop:

1. Initialise city, market, and agents.
2. At each step: agents observe the market state, decide actions, market
   resolves consumer choices, metrics are recorded.
3. Results are returned as a list of step-level metric dictionaries and
   optionally persisted to disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from hotelling.core.city import City
from hotelling.core.firm import Firm
from hotelling.core.market import Market
from hotelling.simulation.config import SimulationConfig
from hotelling.simulation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Runs the Hotelling competition simulation.

    Parameters
    ----------
    config:
        A :class:`~hotelling.simulation.config.SimulationConfig` instance.
    """

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        self.city: Optional[City] = None
        self.market: Optional[Market] = None
        self._results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, firms: List[Firm]) -> None:
        """Initialise the city and market with the provided firms.

        Parameters
        ----------
        firms:
            List of :class:`~hotelling.core.firm.Firm` (or agent) instances
            that will compete in the simulation.
        """
        cfg = self.config
        self.city = City(
            width=cfg.city_width,
            height=cfg.city_height,
            n_consumers=cfg.n_consumers,
            seed=cfg.seed,
        )
        self.market = Market(
            city=self.city,
            firms=list(firms),
            transport_cost=cfg.transport_cost,
        )
        self._results = []
        logger.info("Simulation set up: %s, %d firms.", self.city, len(firms))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> List[Dict[str, Any]]:
        """Execute the simulation and return per-step metrics.

        Returns
        -------
        list of dict
            One dictionary per simulation step containing firm-level metrics.
        """
        if self.city is None or self.market is None:
            raise RuntimeError("Call setup() before run().")

        cfg = self.config

        for step in range(cfg.n_steps):
            self._step(step)

        logger.info("Simulation finished after %d steps.", cfg.n_steps)
        return self._results

    def _step(self, step: int) -> None:
        """Execute a single simulation step."""
        city = self.city
        market = self.market

        # 1. Agents observe and act
        for firm in market.firms:
            if hasattr(firm, "update"):
                firm.update(city, market)

        # 2. Market resolves consumer choices
        market.resolve()

        # 3. Agents receive reward signal (for learning agents)
        for firm in market.firms:
            if hasattr(firm, "observe"):
                state = {
                    "step": step,
                    "reward": firm.profit,
                    "self": {
                        "firm_id": firm.firm_id,
                        "location": firm.location,
                        "price": firm.price,
                        "market_share": firm.market_share,
                    },
                    "firms": [
                        {
                            "firm_id": f.firm_id,
                            "location": f.location,
                            "price": f.price,
                            "market_share": f.market_share,
                        }
                        for f in market.firms
                        if f.firm_id != firm.firm_id
                    ],
                    "city": {"width": city.width, "height": city.height},
                }
                firm.observe(state)

        # 4. Record metrics
        if self.config.log_every > 0 and step % self.config.log_every == 0:
            metrics = compute_metrics(step, market)
            self._results.append(metrics)
            logger.debug("Step %d: %s", step, metrics)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, filename: str = "results.json") -> Path:
        """Write simulation results to a JSON file.

        Parameters
        ----------
        filename:
            Output filename (relative to ``config.output_dir``).

        Returns
        -------
        Path
            The path to the written file.
        """
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        with open(out_path, "w") as fh:
            json.dump(self._results, fh, indent=2)
        logger.info("Results saved to %s.", out_path)
        return out_path

    @property
    def results(self) -> List[Dict[str, Any]]:
        """Read-only access to accumulated step results."""
        return list(self._results)
