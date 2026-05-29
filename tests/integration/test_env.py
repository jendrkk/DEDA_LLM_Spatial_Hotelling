"""Integration tests for env.HotellingMarketEnv construction."""
from __future__ import annotations

import pytest

from hotelling.core.city import City
from hotelling.core.firm import Firm
from hotelling.env.market_env import HotellingMarketEnv


@pytest.fixture
def duopoly_env():
    import numpy as np

    firms = [
        Firm(id="firm_0", location=(0.25, 0.5), marginal_cost=1.0,
             quality=0.0, kappa0=1.0, size=600.0, rent=0.0),
        Firm(id="firm_1", location=(0.75, 0.5), marginal_cost=1.0,
             quality=0.0, kappa0=1.0, size=600.0, rent=0.0),
    ]
    city = City(
        boundary=(0.0, 0.0, 1.0, 1.0),
        population_grid=None,
        firms=firms,
        dist2_km2=np.ones((1, 2)),
        cell_pop=np.array([1.0]),
        lambda_phi=np.zeros(1),
        pi_H=np.array([0.5]),
        pi_H_lambda_phi=np.array([0.5]),
        alpha=np.array([0.5, 1.5]),
        beta=0.001,
    )
    return HotellingMarketEnv(
        city=city,
        firms=firms,
        m=15,
        transport_cost=0.0,
        min_price=1.0,
        max_price=2.0,
    )


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
        obs, _ = env.reset(seed=0)
        action_space_size = env.m * env.m_effort
        for agent_obs in obs.values():
            assert 0 <= agent_obs["own_prev_action"] < action_space_size
