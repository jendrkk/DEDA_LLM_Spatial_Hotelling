"""Unit tests for HotellingMarketEnv.grid_spec() and .graph_degree_spec()."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.core.city import City
from hotelling.core.firm import Firm
from hotelling.env.market_env import HotellingMarketEnv


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_city() -> City:
    return City(
        boundary=(0.0, 0.0, 1.0, 1.0),
        population_grid=None,
        firms=[],
        dist2_km2=np.zeros((2, 2), dtype=float),
        cell_pop=np.array([100.0, 150.0], dtype=float),
        lambda_phi=np.array([10.0, 20.0], dtype=float),
        pi_H=np.array([0.4, 0.6], dtype=float),
        pi_H_lambda_phi=np.array([0.4, 0.6], dtype=float),
        alpha=np.array([0.0, 0.0], dtype=float),
        beta=0.0,
        mu=0.25,
        a0=0.0,
    )


def _make_firm(idx: int, chain_type: str = "standard", x: float = 0.5) -> Firm:
    return Firm(
        id=f"f{idx}",
        location=(x, 0.5),
        marginal_cost=1.0,
        quality=1.0,
        kappa0=1.0,
        size=1.0,
        rent=0.0,
        chain_type=chain_type,
    )


def _global_env(m: int = 10, n: int = 3) -> HotellingMarketEnv:
    city = _make_city()
    firms = [_make_firm(i, x=float(i)) for i in range(n)]
    city.firms = firms
    return HotellingMarketEnv(
        city=city,
        firms=firms,
        m=m,
        min_price=1.0,
        max_price=5.0,
        m_effort=1,
        k_neighbors=1,
        state_mode="neighbors",
    )


def _cs_env(m: int = 8) -> HotellingMarketEnv:
    """Chain-specific grid env: discount [1,3], standard [2,5], bio [3,7]."""
    city = _make_city()
    firms = [
        _make_firm(0, "discount", x=0.0),
        _make_firm(1, "standard", x=1.0),
        _make_firm(2, "bio",      x=2.0),
        _make_firm(3, "discount", x=3.0),
    ]
    city.firms = firms
    chain_type_grids = {
        "discount": np.linspace(1.0, 3.0, m),
        "standard": np.linspace(2.0, 5.0, m),
        "bio":      np.linspace(3.0, 7.0, m),
    }
    return HotellingMarketEnv(
        city=city,
        firms=firms,
        m=m,
        min_price=1.0,
        max_price=7.0,
        m_effort=1,
        k_neighbors=1,
        state_mode="neighbors",
        chain_type_grids=chain_type_grids,
    )


def _graph_env(graph_rivals: np.ndarray, graph_k: int = 2) -> HotellingMarketEnv:
    city = _make_city()
    n = graph_rivals.shape[0]
    firms = [_make_firm(i, x=float(i)) for i in range(n)]
    city.firms = firms
    return HotellingMarketEnv(
        city=city,
        firms=firms,
        m=8,
        min_price=1.0,
        max_price=5.0,
        m_effort=1,
        k_neighbors=1,
        state_mode="graph_states",
        graph_rivals=graph_rivals,
        graph_k=graph_k,
        graph_n_rival_bins=5,
        graph_rival_match="A",
    )


# ---------------------------------------------------------------------------
# grid_spec — global regime
# ---------------------------------------------------------------------------

class TestGridSpecGlobal:
    def test_regime_is_G(self):
        env = _global_env()
        assert env.grid_spec()["regime"] == "G"

    def test_m_matches_env(self):
        env = _global_env(m=10)
        assert env.grid_spec()["m"] == env.m

    def test_step_consistency(self):
        env = _global_env(m=10)
        spec = env.grid_spec()
        expected_step = (spec["hi"] - spec["lo"]) / (env.m - 1)
        assert abs(spec["step"] - expected_step) < 1e-9

    def test_lo_hi_match_price_grid(self):
        env = _global_env(m=10)
        spec = env.grid_spec()
        assert abs(spec["lo"] - float(env.price_grid.min())) < 1e-9
        assert abs(spec["hi"] - float(env.price_grid.max())) < 1e-9

    def test_grid_list_length(self):
        env = _global_env(m=10)
        assert len(env.grid_spec()["grid"]) == 10

    def test_other_chain_grids_empty_in_G(self):
        env = _global_env()
        assert env.grid_spec()["other_chain_grids"] == {}

    def test_chain_type_arg_ignored_in_G(self):
        env = _global_env()
        assert env.grid_spec("discount") == env.grid_spec()


# ---------------------------------------------------------------------------
# grid_spec — chain-specific regime
# ---------------------------------------------------------------------------

class TestGridSpecChainSpecific:
    def test_discount_regime_is_CS(self):
        env = _cs_env()
        assert env.grid_spec("discount")["regime"] == "CS"

    def test_discount_lo_hi(self):
        env = _cs_env(m=8)
        spec = env.grid_spec("discount")
        assert abs(spec["lo"] - 1.0) < 1e-9
        assert abs(spec["hi"] - 3.0) < 1e-9

    def test_bio_lo_hi(self):
        env = _cs_env(m=8)
        spec = env.grid_spec("bio")
        assert abs(spec["lo"] - 3.0) < 1e-9
        assert abs(spec["hi"] - 7.0) < 1e-9

    def test_step_consistency_cs(self):
        env = _cs_env(m=8)
        spec = env.grid_spec("standard")
        expected = (spec["hi"] - spec["lo"]) / (spec["m"] - 1)
        assert abs(spec["step"] - expected) < 1e-9

    def test_other_chain_grids_contains_standard_and_bio(self):
        env = _cs_env()
        others = env.grid_spec("discount")["other_chain_grids"]
        assert "standard" in others
        assert "bio" in others
        assert "discount" not in others

    def test_other_chain_grids_keys(self):
        env = _cs_env()
        for ct in ("standard", "bio"):
            spec = env.grid_spec("discount")["other_chain_grids"][ct]
            assert "lo" in spec and "hi" in spec and "step" in spec

    def test_none_chain_type_returns_global_grid(self):
        env = _cs_env(m=8)
        spec = env.grid_spec(None)
        # regime is still CS (chain-specific grids are active)
        assert spec["regime"] == "CS"
        # grid matches env.price_grid
        assert abs(spec["lo"] - float(env.price_grid.min())) < 1e-9
        assert abs(spec["hi"] - float(env.price_grid.max())) < 1e-9

    def test_unknown_chain_type_falls_back_to_global(self):
        env = _cs_env(m=8)
        spec = env.grid_spec("unknown_ct")
        assert spec["lo"] == env.grid_spec(None)["lo"]


# ---------------------------------------------------------------------------
# graph_degree_spec — graph_states mode
# ---------------------------------------------------------------------------

class TestGraphDegreeSpecGraphStates:
    def _rivals_array(self) -> np.ndarray:
        """5 stores, graph_k=2; stores 0 and 2 are fully isolated."""
        # row 0: [-1, -1]  → isolated
        # row 1: [0, 2]    → degree 2
        # row 2: [-1, -1]  → isolated
        # row 3: [1, 4]    → degree 2
        # row 4: [3, -1]   → degree 1
        return np.array([
            [-1, -1],
            [0,  2],
            [-1, -1],
            [1,  4],
            [3,  -1],
        ], dtype=np.int64)

    def test_k_equals_graph_k(self):
        rivals = self._rivals_array()
        env = _graph_env(rivals, graph_k=2)
        spec = env.graph_degree_spec()
        assert spec["k"] == 2

    def test_n_stores_all(self):
        rivals = self._rivals_array()
        env = _graph_env(rivals, graph_k=2)
        assert env.graph_degree_spec()["n_stores"] == 5

    def test_n_isolated_two(self):
        rivals = self._rivals_array()
        env = _graph_env(rivals, graph_k=2)
        # rows 0 and 2 are fully isolated
        assert env.graph_degree_spec()["n_isolated"] == 2

    def test_mean_observed(self):
        rivals = self._rivals_array()
        env = _graph_env(rivals, graph_k=2)
        spec = env.graph_degree_spec()
        # degrees: 0, 2, 0, 2, 1 → mean = 1.0
        assert abs(spec["mean_observed"] - 1.0) < 1e-9

    def test_max_observed(self):
        rivals = self._rivals_array()
        env = _graph_env(rivals, graph_k=2)
        assert env.graph_degree_spec()["max_observed"] == 2

    def test_mode_field(self):
        rivals = self._rivals_array()
        env = _graph_env(rivals, graph_k=2)
        assert env.graph_degree_spec()["mode"] == "graph_states"

    def test_match_field(self):
        rivals = self._rivals_array()
        env = _graph_env(rivals, graph_k=2)
        assert env.graph_degree_spec()["match"] == "A"

    def test_grid_regime_G(self):
        rivals = self._rivals_array()
        env = _graph_env(rivals, graph_k=2)
        assert env.graph_degree_spec()["grid_regime"] == "G"

    def test_all_isolated_when_all_minus_one(self):
        rivals = np.full((4, 2), -1, dtype=np.int64)
        env = _graph_env(rivals, graph_k=2)
        spec = env.graph_degree_spec()
        assert spec["n_isolated"] == 4
        assert spec["mean_observed"] == 0.0


# ---------------------------------------------------------------------------
# graph_degree_spec — non-graph mode (neighbors)
# ---------------------------------------------------------------------------

class TestGraphDegreeSpecNeighbors:
    def test_mode_neighbors(self):
        env = _global_env(m=8, n=4)
        spec = env.graph_degree_spec()
        assert spec["mode"] == "neighbors"

    def test_match_na(self):
        env = _global_env(m=8, n=4)
        assert env.graph_degree_spec()["match"] == "n/a"

    def test_k_equals_k_neighbors(self):
        env = _global_env(m=8, n=4)
        assert env.graph_degree_spec()["k"] == env.k_neighbors

    def test_n_isolated_zero_in_neighbors_mode(self):
        env = _global_env(m=8, n=4)
        assert env.graph_degree_spec()["n_isolated"] == 0


# ---------------------------------------------------------------------------
# Import sanity
# ---------------------------------------------------------------------------

def test_import_clean():
    import hotelling.env.market_env  # noqa: F401
