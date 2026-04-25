"""
General Hotelling competition experiment.

Runs a symmetric two-firm competition with naive (center-seeking) agents in a
unit-square city and saves the results to the ``results/`` directory.

Usage
-----
    python -m experiments.general_hotelling
    python experiments/general_hotelling.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from hotelling.agents.naive_agent import NaiveAgent
from hotelling.simulation.config import SimulationConfig
from hotelling.simulation.engine import SimulationEngine
from hotelling.simulation.metrics import aggregate_results
from hotelling.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run(
    n_steps: int = 200,
    n_consumers: int = 2000,
    seed: int = 42,
    output_dir: str = "results/general",
) -> None:
    setup_logging(logging.INFO)

    cfg = SimulationConfig(
        n_steps=n_steps,
        n_consumers=n_consumers,
        seed=seed,
        log_every=10,
        output_dir=output_dir,
    )
    engine = SimulationEngine(config=cfg)

    firms = [
        NaiveAgent(firm_id="Firm_A", location=(0.1, 0.5), strategy="center", seed=seed),
        NaiveAgent(firm_id="Firm_B", location=(0.9, 0.5), strategy="center", seed=seed + 1),
    ]

    engine.setup(firms)
    results = engine.run()
    agg = aggregate_results(results)

    out_path = engine.save_results("general_results.json")
    logger.info("Results written to %s", out_path)

    # Print final market shares
    logger.info("Final market state:")
    for firm in engine.market.firms:
        logger.info("  %s: share=%.3f  profit=%.2f", firm.name, firm.market_share, firm.profit)


if __name__ == "__main__":
    run()
