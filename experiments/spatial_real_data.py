"""
Spatial real-data experiment.

Demonstrates how to load real population and business-location data into the
Hotelling simulation.  When the data files are absent, the experiment falls
back to synthetic data so it can still be run end-to-end.

Usage
-----
    python experiments/spatial_real_data.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hotelling.agents.naive_agent import NaiveAgent
from hotelling.agents.qlearning_agent import QLearningAgent
from hotelling.core.city import City
from hotelling.core.market import Market
from hotelling.simulation.config import SimulationConfig
from hotelling.simulation.engine import SimulationEngine
from hotelling.simulation.metrics import aggregate_results
from hotelling.spatial.business_locations import BusinessLocations
from hotelling.spatial.map_loader import MapLoader
from hotelling.spatial.population import PopulationGrid
from hotelling.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"


def run(
    n_steps: int = 100,
    n_consumers: int = 2000,
    seed: int = 42,
    output_dir: str = "results/spatial",
) -> None:
    setup_logging(logging.INFO)

    # ---- Optional: load real spatial data --------------------------------
    map_loader = MapLoader(
        filepath=_DATA_DIR / "raw" / "boundary.geojson" if (
            _DATA_DIR / "raw" / "boundary.geojson"
        ).exists() else None
    )

    pop_grid = PopulationGrid(
        raster_path=_DATA_DIR / "raw" / "population.tif" if (
            _DATA_DIR / "raw" / "population.tif"
        ).exists() else None,
        map_loader=map_loader,
    )

    biz_locs = BusinessLocations(
        filepath=_DATA_DIR / "raw" / "businesses.csv" if (
            _DATA_DIR / "raw" / "businesses.csv"
        ).exists() else None,
        map_loader=map_loader,
    )

    # ---- Build city with population-weighted consumers -------------------
    consumer_locations = pop_grid.sample_locations(n=n_consumers, seed=seed)
    city = City(width=1.0, height=1.0, n_consumers=n_consumers, seed=seed)
    # Override with population-sampled locations if available
    city.consumer_locations = consumer_locations

    # ---- Initialise firms ------------------------------------------------
    sampled_biz = biz_locs.sample(n=2, seed=seed)
    rng = np.random.default_rng(seed)

    def _loc(idx: int) -> tuple:
        if idx < len(sampled_biz):
            return sampled_biz[idx]
        return (float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)))

    firms = [
        QLearningAgent(
            firm_id="QL_Firm",
            location=_loc(0),
            price=1.0,
            seed=seed,
        ),
        NaiveAgent(
            firm_id="Naive_Firm",
            location=_loc(1),
            price=1.0,
            strategy="random_walk",
            seed=seed + 1,
        ),
    ]

    cfg = SimulationConfig(
        n_steps=n_steps,
        n_consumers=n_consumers,
        seed=seed,
        log_every=10,
        output_dir=output_dir,
    )
    engine = SimulationEngine(config=cfg)
    engine.city = city
    from hotelling.core.market import Market  # noqa: PLC0415

    engine.market = Market(city=city, firms=firms)
    engine._results = []

    results = engine.run()
    agg = aggregate_results(results)

    out_path = engine.save_results("spatial_results.json")
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
