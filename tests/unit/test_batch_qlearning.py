"""Unit tests for BatchQLearningAgent."""
from __future__ import annotations

import numpy as np

from hotelling.agents.batch_qlearning import BatchQLearningAgent
from hotelling.agents.qlearning import QLearningAgent


def test_encode_state_matches_single_agent_k1() -> None:
    m, m_effort, k = 15, 5, 1
    n = 4
    batch = BatchQLearningAgent(n, m, m_effort, k, 0.1, 4e-6, 0.95, seed=0)
    single = QLearningAgent("0", m=m, m_effort=m_effort, k=k, seed=0)

    neighbors = np.array([[3], [7], [0], [14]], dtype=np.int64)
    batch_states = batch._encode_states(neighbors)
    for i in range(n):
        assert batch_states[i] == single._encode_state(neighbors[i].tolist())


def test_act_shape_and_epsilon_decay() -> None:
    batch = BatchQLearningAgent(3, 15, 5, 1, 0.1, 4e-6, 0.95, seed=42)
    neighbors = np.zeros((3, 1), dtype=np.int64)
    for _ in range(100):
        actions = batch.act(neighbors)
        assert actions.shape == (3,)
    expected = float(np.exp(-4e-6 * 100))
    assert batch.epsilon_mean == np.exp(-4e-6 * batch._t).mean()
    assert abs(batch.epsilon_mean - expected) < 1e-9
