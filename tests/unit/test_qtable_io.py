import numpy as np
import pytest

from hotelling.agents.batch_qlearning import BatchQLearningAgent


def _agent(seed=0, m=5):
    return BatchQLearningAgent(n_agents=4, m=m, m_effort=1, k=1, alpha=0.15,
                               beta_decay=4e-6, delta=0.95, seed=seed,
                               state_mode="neighbors")


def test_qtable_roundtrip(tmp_path):
    a = _agent(seed=1)
    a._q[:] = np.random.default_rng(0).random(a._q.shape)
    a._t[:] = 12345
    a.save_qtable(tmp_path / "q.npz")
    b = _agent(seed=2)
    b.load_qtable(tmp_path / "q.npz")
    np.testing.assert_array_equal(a._q, b._q)
    np.testing.assert_array_equal(a._t, b._t)


def test_qtable_shape_mismatch_raises(tmp_path):
    a = _agent(seed=1, m=5)
    a.save_qtable(tmp_path / "q.npz")
    b = _agent(seed=1, m=7)  # different action_size
    with pytest.raises(ValueError):
        b.load_qtable(tmp_path / "q.npz")
