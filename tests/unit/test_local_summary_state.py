"""Unit tests for local_summary Q-learning state representation."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hotelling.agents.batch_qlearning import BatchQLearningAgent
from hotelling.core.city import City
from hotelling.core.firm import Firm
from hotelling.env.market_env import HotellingMarketEnv, _local_price_summary


def test_local_price_summary_hand_csr() -> None:
    """_local_price_summary on a hand-built CSR returns correct mean & min."""
    # 3 stores; store 0 competes with 1 and 2; store 1 with 0; store 2 isolated
    indptr = np.array([0, 2, 3, 3], dtype=np.int64)
    indices = np.array([1, 2, 0], dtype=np.int64)
    prices = np.array([10.0, 4.0, 6.0], dtype=np.float64)

    mean_c, min_c = _local_price_summary(prices, indptr, indices)

    np.testing.assert_allclose(mean_c, [5.0, 10.0, 6.0])
    np.testing.assert_allclose(min_c, [4.0, 10.0, 6.0])


def test_price_binning_min_max() -> None:
    """Competitor mean at grid min -> bin 0; at grid max -> B-1."""
    firms = [
        Firm(id="0", location=(0.0, 0.0), marginal_cost=0.0, quality=1.0,
             kappa0=1.0, size=1.0, rent=0.0),
        Firm(id="1", location=(1.0, 0.0), marginal_cost=0.0, quality=1.0,
             kappa0=1.0, size=1.0, rent=0.0),
    ]
    city = City(
        boundary=(0, 0, 2, 1),
        population_grid=None,
        firms=firms,
        dist2_km2=np.ones((1, 2)),
        cell_pop=np.array([100.0]),
        lambda_phi=np.array([10.0]),
        pi_H=np.array([0.5]),
        pi_H_lambda_phi=np.array([0.5]),
        alpha=np.array([0.0, 0.0]),
        beta=0.0,
        mu=0.25,
        a0=0.0,
        catch_indptr=np.array([0, 2], dtype=np.int64),
        catch_indices=np.array([0, 1], dtype=np.int32),
        catch_tt=np.array([1.0, 1.0]),
    )
    B = 5
    env = HotellingMarketEnv(
        city=city,
        firms=firms,
        m=B,
        m_effort=1,
        k_neighbors=1,
        min_price=0.0,
        max_price=4.0,
        state_mode="local_summary",
        n_price_bins=B,
        summary_stats=("mean",),
    )

    # Store 0's only competitor is store 1 (shared catchment cell).
    # Competitor at grid max -> bin B-1
    env._current_joint_actions_arr[:] = np.array([0, B - 1], dtype=np.int64)
    assert env.current_state_signal()[0] == B - 1

    # Competitor at grid min -> bin 0
    env._current_joint_actions_arr[:] = np.array([B - 1, 0], dtype=np.int64)
    assert env.current_state_signal()[0] == 0


def test_state_size_and_two_stat_encoding() -> None:
    """state_size == B^len(stats); mean+min encoding is bin_mean + bin_min*B."""
    firms = [
        Firm(id=str(i), location=(float(i), 0.0), marginal_cost=0.0, quality=1.0,
             kappa0=1.0, size=1.0, rent=0.0)
        for i in range(3)
    ]
    city = City(
        boundary=(0, 0, 3, 1),
        population_grid=None,
        firms=firms,
        dist2_km2=np.ones((1, 3)),
        cell_pop=np.array([100.0]),
        lambda_phi=np.array([10.0]),
        pi_H=np.array([0.5]),
        pi_H_lambda_phi=np.array([0.5]),
        alpha=np.array([0.0, 0.0]),
        beta=0.0,
        mu=0.25,
        a0=0.0,
        catch_indptr=np.array([0, 3], dtype=np.int64),
        catch_indices=np.array([0, 1, 2], dtype=np.int32),
        catch_tt=np.array([1.0, 1.0, 1.0]),
    )
    B = 4
    env = HotellingMarketEnv(
        city=city,
        firms=firms,
        m=B,
        m_effort=1,
        state_mode="local_summary",
        n_price_bins=B,
        summary_stats=("mean", "min"),
        min_price=0.0,
        max_price=3.0,
    )
    assert env.state_size == B ** 2

    pair_to_index: dict[tuple[int, int], int] = {}
    for p1 in range(B):
        for p2 in range(B):
            env._current_joint_actions_arr[:] = np.array([0, p1, p2], dtype=np.int64)
            sig = int(env.current_state_signal()[0])
            prices = env.price_grid[[p1, p2]]
            mean_bin = int(np.clip(
                np.digitize(prices.mean(), env._price_bin_edges) - 1, 0, B - 1
            ))
            min_bin = int(np.clip(
                np.digitize(prices.min(), env._price_bin_edges) - 1, 0, B - 1
            ))
            expected = mean_bin + min_bin * B
            assert sig == expected
            key = (mean_bin, min_bin)
            if key in pair_to_index:
                assert pair_to_index[key] == sig
            else:
                pair_to_index[key] = sig

    assert len(pair_to_index) == len(set(pair_to_index.values()))


def test_demand_overlap_csr_symmetric_self_excluded() -> None:
    """Demand-overlap CSR on a 3-cell toy catchment excludes self and is symmetric."""
    firms = [
        Firm(id=str(i), location=(float(i), 0.0), marginal_cost=0.0, quality=1.0,
             kappa0=1.0, size=1.0, rent=0.0)
        for i in range(4)
    ]
    # Cell 0: stores 0,1; cell 1: stores 1,2; cell 2: store 3 alone
    catch_indptr = np.array([0, 2, 4, 5], dtype=np.int64)
    catch_indices = np.array([0, 1, 1, 2, 3], dtype=np.int32)
    city = City(
        boundary=(0, 0, 4, 1),
        population_grid=None,
        firms=firms,
        dist2_km2=None,
        cell_pop=np.ones(3),
        lambda_phi=np.ones(3),
        pi_H=np.full(3, 0.5),
        pi_H_lambda_phi=np.full(3, 0.5),
        alpha=np.array([0.0, 0.0]),
        beta=0.0,
        mu=0.25,
        a0=0.0,
        catch_indptr=catch_indptr,
        catch_indices=catch_indices,
        catch_tt=np.ones(5),
    )
    env = HotellingMarketEnv(
        city=city,
        firms=firms,
        m=5,
        m_effort=1,
        state_mode="local_summary",
        local_sum_n=None,
    )
    N = 4
    adj = np.zeros((N, N), dtype=bool)
    for j in range(N):
        for idx in env._comp_indices[env._comp_indptr[j]: env._comp_indptr[j + 1]]:
            adj[j, idx] = True

    assert not np.any(np.diag(adj))
    assert np.allclose(adj, adj.T)
    # store 0 <-> 1; store 1 <-> 0,2; store 2 <-> 1; store 3 isolated
    assert adj[0, 1] and adj[1, 0]
    assert adj[1, 2] and adj[2, 1]
    assert not adj[3].any() and not adj[:, 3].any()


def test_agent_local_summary_encode_passthrough() -> None:
    """local_summary agent allocates (n, K, action_size) and passes signal through."""
    n, K, action_size = 3, 15, 75
    agent = BatchQLearningAgent(
        n_agents=n,
        m=15,
        m_effort=5,
        k=1,
        alpha=0.15,
        beta_decay=4e-6,
        delta=0.95,
        state_mode="local_summary",
        state_size=K,
    )
    assert agent._q.shape == (n, K, action_size)
    signal = np.array([0, 7, 14], dtype=np.int64)
    np.testing.assert_array_equal(agent._encode_states(signal), signal)
