"""Unit tests for hotelling.agents – protocol conformance and QLearningAgent."""
from __future__ import annotations

import math

import pytest

from hotelling.agents.base import AgentProtocol
from hotelling.agents.myopic import MyopicAgent
from hotelling.agents.qlearning import QLearningAgent
from hotelling.agents.random_agent import RandomAgent


class TestAgentProtocol:
    def test_qlearning_satisfies_protocol(self):
        agent = QLearningAgent(firm_id="A")
        assert isinstance(agent, AgentProtocol)

    def test_myopic_satisfies_protocol(self):
        agent = MyopicAgent(firm_id="A")
        assert isinstance(agent, AgentProtocol)

    def test_random_satisfies_protocol(self):
        agent = RandomAgent(firm_id="A")
        assert isinstance(agent, AgentProtocol)


class TestQLearningAgent:
    def test_init_defaults(self):
        agent = QLearningAgent(firm_id="A")
        assert agent.firm_id == "A"
        assert agent.m == 15
        assert agent.alpha == 0.10
        assert agent.delta == 0.95

    def test_epsilon_at_zero_steps(self):
        agent = QLearningAgent(firm_id="A", beta=1e-5, seed=0)
        # At t=0, epsilon = exp(0) = 1.0
        assert agent.epsilon == pytest.approx(1.0)

    def test_epsilon_decays(self):
        agent = QLearningAgent(firm_id="A", beta=2e-5, seed=0)
        agent._t = 100_000
        expected = math.exp(-2e-5 * 100_000)
        assert agent.epsilon == pytest.approx(expected, rel=1e-6)

