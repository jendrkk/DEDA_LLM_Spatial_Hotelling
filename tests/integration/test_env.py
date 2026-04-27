"""Integration tests for env.HotellingMarketEnv construction."""
from __future__ import annotations

import pytest

from hotelling.core.city import City
from hotelling.core.firm import Firm
from hotelling.env.market_env import HotellingMarketEnv


@pytest.fixture
def duopoly_env():
    city = City(boundary=(0.0, 0.0, 1.0, 1.0))
    firms = [
        Firm(id="firm_0", location=(0.25, 0.5), marginal_cost=1.0),
        Firm(id="firm_1", location=(0.75, 0.5), marginal_cost=1.0),
    ]
    city.firms = firms
    return HotellingMarketEnv(city=city, firms=firms, m=15)


class TestHotellingMarketEnvConstruction:
    def test_possible_agents(self, duopoly_env):
        env = duopoly_env
        assert set(env.possible_agents) == {"firm_0", "firm_1"}

    def test_price_grid_length(self, duopoly_env):
        assert len(duopoly_env.price_grid) == 15

    def test_price_grid_bounds(self, duopoly_env):
        import numpy as np

        env = duopoly_env
        assert env.price_grid[0] == pytest.approx(1.0)
        assert env.price_grid[-1] == pytest.approx(2.0)

    def test_initial_price_indices_in_range(self, duopoly_env):
        env = duopoly_env
        for agent, idx in env._current_prices.items():
            assert 0 <= idx < env.m
