"""
LLM competition experiment.

Runs a Hotelling competition where one firm is controlled by an LLM agent and
the other by a naive (center-seeking) agent.

Requires an OpenAI-compatible API key set via the ``OPENAI_API_KEY``
environment variable or passed directly to :class:`LLMAgent`.

Usage
-----
    python experiments/llm_competition.py
"""

from __future__ import annotations

import logging
import os

from hotelling.agents.llm_agent import LLMAgent
from hotelling.agents.naive_agent import NaiveAgent
from hotelling.simulation.config import SimulationConfig
from hotelling.simulation.engine import SimulationEngine
from hotelling.simulation.metrics import aggregate_results
from hotelling.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run(
    n_steps: int = 50,
    n_consumers: int = 1000,
    seed: int = 42,
    llm_model: str = "gpt-4o-mini",
    output_dir: str = "results/llm",
) -> None:
    setup_logging(logging.INFO)

    cfg = SimulationConfig(
        n_steps=n_steps,
        n_consumers=n_consumers,
        seed=seed,
        log_every=5,
        output_dir=output_dir,
    )
    engine = SimulationEngine(config=cfg)

    api_key = os.environ.get("OPENAI_API_KEY")
    firms = [
        LLMAgent(
            firm_id="LLM_Firm",
            location=(0.3, 0.5),
            price=1.0,
            model=llm_model,
            api_key=api_key,
        ),
        NaiveAgent(
            firm_id="Naive_Firm",
            location=(0.7, 0.5),
            price=1.0,
            strategy="center",
            seed=seed,
        ),
    ]

    engine.setup(firms)
    results = engine.run()
    agg = aggregate_results(results)

    out_path = engine.save_results("llm_results.json")
    logger.info("Results written to %s", out_path)

    logger.info("Aggregated %d steps. HHI range: %.3f – %.3f",
                len(agg.get("steps", [])),
                min(agg.get("hhi", [0])),
                max(agg.get("hhi", [0])))
    logger.info("Results written to %s", out_path)

    logger.info("Final market state:")
    for firm in engine.market.firms:
        logger.info("  %s: share=%.3f  profit=%.2f", firm.name, firm.market_share, firm.profit)


if __name__ == "__main__":
    run()
