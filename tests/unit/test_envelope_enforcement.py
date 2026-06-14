import numpy as np
import pytest

from hotelling.agents.batch_qlearning import BatchQLearningAgent


def _agent(seed=0):
    return BatchQLearningAgent(
        n_agents=4, m=5, m_effort=1, k=1, alpha=0.15,
        beta_decay=4e-6, delta=0.95, seed=seed, state_mode="neighbors",
    )


def test_no_mask_is_identical_baseline():
    a = _agent(seed=1)
    b = _agent(seed=1)
    sig = np.zeros((4, 1), dtype=np.int64)
    for _ in range(50):
        np.testing.assert_array_equal(a.act(sig.copy()), b.act(sig.copy()))


def test_mask_confines_actions():
    a = _agent(seed=2)
    A = a.action_size  # 5
    mask = np.zeros((4, A), dtype=bool)
    mask[:, 1:3] = True  # only actions {1,2} allowed
    a.set_action_mask(mask)
    a.set_epsilon_override(np.full(4, 0.3))
    sig = np.zeros((4, 1), dtype=np.int64)
    for _ in range(200):
        acts = a.act(sig)
        assert np.all((acts >= 1) & (acts <= 2)), acts


def test_epsilon_override_reported():
    a = _agent(seed=3)
    a.set_epsilon_override(np.full(4, 0.07))
    assert abs(a.epsilon_mean - 0.07) < 1e-12
    a.set_epsilon_override(None)
    assert a.epsilon_mean <= 1.0


def test_mask_shape_validation():
    a = _agent(seed=4)
    with pytest.raises(ValueError):
        a.set_action_mask(np.ones((3, a.action_size), dtype=bool))
    with pytest.raises(ValueError):
        a.set_epsilon_override(np.ones(3))
