"""Tests for agent implementations."""

import pytest

from hotelling.agents.naive_agent import NaiveAgent
from hotelling.agents.qlearning_agent import QLearningAgent
from hotelling.core.city import City
from hotelling.core.market import Market


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(city_w=1.0, city_h=1.0, step=0):
    return {
        "step": step,
        "self": {
            "firm_id": "A",
            "location": (0.5, 0.5),
            "price": 1.0,
            "market_share": 0.5,
        },
        "firms": [
            {
                "firm_id": "B",
                "location": (0.75, 0.5),
                "price": 1.0,
                "market_share": 0.5,
            }
        ],
        "city": {"width": city_w, "height": city_h},
        "reward": 0.5,
    }


# ---------------------------------------------------------------------------
# NaiveAgent
# ---------------------------------------------------------------------------


class TestNaiveAgent:
    def test_stay_strategy_does_not_move(self):
        agent = NaiveAgent(firm_id="A", location=(0.3, 0.7), strategy="stay")
        state = _make_state()
        action = agent.act(state)
        assert action["location"] == (0.3, 0.7)

    def test_center_strategy_moves_towards_center(self):
        agent = NaiveAgent(firm_id="A", location=(0.0, 0.0), strategy="center", step_size=1.0)
        state = _make_state()
        action = agent.act(state)
        x, y = action["location"]
        # Should move towards (0.5, 0.5)
        assert x > 0.0
        assert y > 0.0

    def test_random_walk_stays_in_bounds(self):
        agent = NaiveAgent(firm_id="A", location=(0.5, 0.5), strategy="random_walk", seed=42)
        state = _make_state()
        for _ in range(100):
            action = agent.act(state)
            x, y = action["location"]
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            NaiveAgent(firm_id="A", strategy="invalid")

    def test_history_recorded(self):
        agent = NaiveAgent(firm_id="A", strategy="stay")
        state = _make_state()
        agent.observe(state)
        assert len(agent.history) == 1

    def test_update_changes_state(self):
        city = City(n_consumers=50, seed=0)
        agent = NaiveAgent(firm_id="A", location=(0.0, 0.0), strategy="center")
        market = Market(city=city, firms=[agent])
        market.resolve()
        old_loc = agent.location
        agent.update(city, market)
        # With center strategy from (0,0), location should change
        assert agent.location != old_loc


# ---------------------------------------------------------------------------
# QLearningAgent
# ---------------------------------------------------------------------------


class TestQLearningAgent:
    def test_act_returns_valid_location_and_price(self):
        agent = QLearningAgent(firm_id="A", location=(0.5, 0.5), price=1.0, seed=0)
        state = _make_state()
        action = agent.act(state)
        x, y = action["location"]
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert action["price"] > 0.0

    def test_observe_updates_epsilon(self):
        agent = QLearningAgent(firm_id="A", seed=0, epsilon=1.0, epsilon_decay=0.9)
        state = _make_state()
        _action = agent.act(state)  # noqa: F841 – result intentionally unused
        initial_eps = agent.epsilon
        agent.observe(state)
        assert agent.epsilon < initial_eps

    def test_q_table_populated_after_steps(self):
        agent = QLearningAgent(firm_id="A", seed=42)
        state = _make_state()
        for _ in range(10):
            agent.act(state)
            agent.observe(state)
        assert len(agent._q) > 0

    def test_save_load_roundtrip(self, tmp_path):
        agent = QLearningAgent(firm_id="A", seed=0)
        state = _make_state()
        for _ in range(5):
            agent.act(state)
            agent.observe(state)

        path = tmp_path / "qtable.json"
        agent.save(path)

        agent2 = QLearningAgent(firm_id="A", seed=0)
        agent2.load(path)
        # Both agents should have the same number of states in Q-table
        assert len(agent2._q) == len(agent._q)

    def test_epsilon_floor(self):
        agent = QLearningAgent(firm_id="A", seed=0, epsilon=0.01, epsilon_min=0.05)
        state = _make_state()
        agent.act(state)
        agent.observe(state)
        assert agent.epsilon >= agent.epsilon_min
