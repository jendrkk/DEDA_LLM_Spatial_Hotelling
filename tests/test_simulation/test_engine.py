"""Tests for the simulation engine and metrics."""

import pytest

from hotelling.agents.naive_agent import NaiveAgent
from hotelling.core.firm import Firm
from hotelling.simulation.config import SimulationConfig
from hotelling.simulation.engine import SimulationEngine
from hotelling.simulation.metrics import aggregate_results, compute_metrics
from hotelling.core.market import Market
from hotelling.core.city import City


# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------


class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.n_steps == 100
        assert cfg.seed == 42

    def test_to_from_dict_roundtrip(self):
        cfg = SimulationConfig(n_steps=50, seed=7)
        d = cfg.to_dict()
        cfg2 = SimulationConfig.from_dict(d)
        assert cfg2.n_steps == 50
        assert cfg2.seed == 7


# ---------------------------------------------------------------------------
# SimulationEngine
# ---------------------------------------------------------------------------


class TestSimulationEngine:
    def _make_engine(self, n_steps=20, n_consumers=100):
        cfg = SimulationConfig(n_steps=n_steps, n_consumers=n_consumers, seed=0, log_every=5)
        return SimulationEngine(config=cfg)

    def test_run_without_setup_raises(self):
        engine = self._make_engine()
        with pytest.raises(RuntimeError):
            engine.run()

    def test_run_produces_results(self):
        engine = self._make_engine(n_steps=20)
        firms = [
            Firm(firm_id="A", location=(0.25, 0.5), price=1.0),
            Firm(firm_id="B", location=(0.75, 0.5), price=1.0),
        ]
        engine.setup(firms)
        results = engine.run()
        # With n_steps=20 and log_every=5, expect 4 recorded steps (0,5,10,15)
        assert len(results) == 4

    def test_run_with_naive_agents(self):
        engine = self._make_engine(n_steps=10)
        firms = [
            NaiveAgent(firm_id="A", location=(0.1, 0.5), strategy="center", seed=0),
            NaiveAgent(firm_id="B", location=(0.9, 0.5), strategy="center", seed=1),
        ]
        engine.setup(firms)
        results = engine.run()
        assert len(results) > 0
        # Agents should have moved towards the center
        for firm in engine.market.firms:
            x = firm.location[0]
            assert 0.0 <= x <= 1.0

    def test_save_results(self, tmp_path):
        cfg = SimulationConfig(n_steps=10, output_dir=str(tmp_path), log_every=5, seed=0)
        engine = SimulationEngine(config=cfg)
        firms = [Firm(firm_id="A"), Firm(firm_id="B", location=(0.8, 0.5))]
        engine.setup(firms)
        engine.run()
        out = engine.save_results("test_results.json")
        assert out.exists()

    def test_results_property(self):
        engine = self._make_engine(n_steps=5)
        engine.setup([Firm(firm_id="A"), Firm(firm_id="B", location=(0.8, 0.5))])
        engine.run()
        assert isinstance(engine.results, list)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def _make_resolved_market(self):
        city = City(n_consumers=500, seed=0)
        firms = [
            Firm(firm_id="A", location=(0.25, 0.5), price=1.0),
            Firm(firm_id="B", location=(0.75, 0.5), price=1.0),
        ]
        market = Market(city=city, firms=firms)
        market.resolve()
        return market

    def test_compute_metrics_keys(self):
        market = self._make_resolved_market()
        m = compute_metrics(step=5, market=market)
        for key in ("step", "firms", "hhi", "avg_price", "price_dispersion"):
            assert key in m

    def test_compute_metrics_step(self):
        market = self._make_resolved_market()
        m = compute_metrics(step=7, market=market)
        assert m["step"] == 7

    def test_hhi_in_valid_range(self):
        market = self._make_resolved_market()
        m = compute_metrics(step=0, market=market)
        # HHI is between 0 and 1 for market shares that sum to 1
        assert 0.0 <= m["hhi"] <= 1.0

    def test_aggregate_results_shape(self):
        market = self._make_resolved_market()
        results = [compute_metrics(step=i, market=market) for i in range(5)]
        agg = aggregate_results(results)
        assert agg["steps"] == list(range(5))
        assert len(agg["hhi"]) == 5
        assert "A" in agg["firms"]
        assert "B" in agg["firms"]

    def test_aggregate_empty(self):
        agg = aggregate_results([])
        assert agg == {}
