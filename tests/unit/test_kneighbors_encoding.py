"""Unit tests for k>1 mixed-radix Q-table state encoding."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.agents.batch_qlearning import BatchQLearningAgent


def test_mixed_radix_encoding_k3() -> None:
    agent = BatchQLearningAgent(
        n_agents=4,
        m=4,
        m_effort=1,
        k=3,
        alpha=0.15,
        beta_decay=4e-6,
        delta=0.95,
        seed=0,
    )
    assert agent.action_size == 4
    assert agent.state_size == 64

    neighbor_actions = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [3, 2, 1],
            [2, 0, 1],
        ],
        dtype=np.int64,
    )
    states = agent._encode_states(neighbor_actions)
    expected = (
        neighbor_actions[:, 0]
        + neighbor_actions[:, 1] * 4
        + neighbor_actions[:, 2] * 16
    )
    np.testing.assert_array_equal(states, expected)
    assert np.all(states >= 0)
    assert np.all(states < 64)
    assert states[0] == 0


def test_distinct_neighbor_tuples_map_to_distinct_states() -> None:
    agent = BatchQLearningAgent(
        n_agents=1,
        m=4,
        m_effort=1,
        k=3,
        alpha=0.15,
        beta_decay=4e-6,
        delta=0.95,
    )
    tuples = [
        (a0, a1, a2)
        for a0 in range(4)
        for a1 in range(4)
        for a2 in range(4)
    ]
    neighbor_actions = np.array(tuples, dtype=np.int64)
    states = agent._encode_states(neighbor_actions)
    assert len(np.unique(states)) == len(tuples)


def test_qtable_memory_guard_raises() -> None:
    with pytest.raises(MemoryError, match="max_qtable_gib"):
        BatchQLearningAgent(
            n_agents=494,
            m=15,
            m_effort=5,
            k=3,
            alpha=0.15,
            beta_decay=4e-6,
            delta=0.95,
            max_qtable_gib=8.0,
        )
