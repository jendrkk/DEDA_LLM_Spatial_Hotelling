"""Unit tests for detailed local-summary state (total + same-type channels)."""
from __future__ import annotations

import numpy as np

from hotelling.core.city import City
from hotelling.core.firm import Firm
from hotelling.env.market_env import HotellingMarketEnv, _local_price_summary


def _six_store_city() -> tuple[list[Firm], City]:
    """6 stores, 2 catchment cells; mixed chain types in each cell."""
    firms = [
        Firm(id="0", location=(0.0, 0.0), marginal_cost=0.0, quality=0.8,
             kappa0=1.0, size=1.0, rent=0.0, chain_type="discount"),
        Firm(id="1", location=(0.1, 0.0), marginal_cost=0.0, quality=0.8,
             kappa0=1.0, size=1.0, rent=0.0, chain_type="discount"),
        Firm(id="2", location=(0.2, 0.0), marginal_cost=0.0, quality=1.0,
             kappa0=1.0, size=1.0, rent=0.0, chain_type="standard"),
        Firm(id="3", location=(1.0, 0.0), marginal_cost=0.0, quality=1.0,
             kappa0=1.0, size=1.0, rent=0.0, chain_type="standard"),
        Firm(id="4", location=(1.1, 0.0), marginal_cost=0.0, quality=1.2,
             kappa0=1.0, size=1.0, rent=0.0, chain_type="bio"),
        Firm(id="5", location=(2.0, 0.0), marginal_cost=0.0, quality=1.2,
             kappa0=1.0, size=1.0, rent=0.0, chain_type="bio"),
    ]
    # Cell 0: 0,1 (discount), 2 (standard); cell 1: 3 (standard), 4,5 (bio)
    catch_indptr = np.array([0, 3, 6], dtype=np.int64)
    catch_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    city = City(
        boundary=(0, 0, 3, 1),
        population_grid=None,
        firms=firms,
        dist2_km2=None,
        cell_pop=np.ones(2),
        lambda_phi=np.ones(2),
        pi_H=np.full(2, 0.5),
        pi_H_lambda_phi=np.full(2, 0.5),
        alpha=np.array([0.0, 0.0]),
        beta=0.0,
        mu=0.25,
        a0=0.0,
        catch_indptr=catch_indptr,
        catch_indices=catch_indices,
        catch_tt=np.ones(6),
    )
    return firms, city


def test_same_type_csr_keeps_only_same_type_edges() -> None:
    """same_type CSR on toy layout keeps only same-type edges, excludes self."""
    firms, city = _six_store_city()
    env = HotellingMarketEnv(
        city=city,
        firms=firms,
        m=5,
        m_effort=1,
        state_mode="local_summary",
        local_summary_detailed=True,
        min_price=0.0,
        max_price=4.0,
    )
    N = 6
    ct = np.array([f.chain_type for f in firms], dtype=object)

    # Rebuild expected same-type adjacency from "all"
    all_adj = np.zeros((N, N), dtype=bool)
    for j in range(N):
        for idx in env._comp_indices[env._comp_indptr[j]: env._comp_indptr[j + 1]]:
            all_adj[j, idx] = True

    same_adj = np.zeros((N, N), dtype=bool)
    for j in range(N):
        for idx in env._comp_indices_same[
            env._comp_indptr_same[j]: env._comp_indptr_same[j + 1]
        ]:
            same_adj[j, idx] = True

    assert not np.any(np.diag(same_adj))
    for j in range(N):
        for k in range(N):
            if same_adj[j, k]:
                assert ct[j] == ct[k]
                assert all_adj[j, k]
    # store 0 (discount): same-type rival is store 1 only (not standard store 2)
    assert same_adj[0, 1] and not same_adj[0, 2]


def test_detailed_state_size_is_b_squared() -> None:
    """env.state_size == n_price_bins**2 when detailed."""
    firms, city = _six_store_city()
    B = 15
    env = HotellingMarketEnv(
        city=city,
        firms=firms,
        m=B,
        m_effort=1,
        state_mode="local_summary",
        local_summary_detailed=True,
        n_price_bins=B,
        min_price=0.0,
        max_price=4.0,
    )
    assert env.state_size == B ** 2
    assert env._ls_channels == [("all", "mean"), ("same_type", "mean")]


def test_mixed_radix_index_all_plus_same_times_b() -> None:
    """Mixed-radix index == bin_all + bin_same*B; distinct pairs -> distinct states."""
    firms, city = _six_store_city()
    B = 4
    env = HotellingMarketEnv(
        city=city,
        firms=firms,
        m=B,
        m_effort=1,
        state_mode="local_summary",
        local_summary_detailed=True,
        n_price_bins=B,
        min_price=0.0,
        max_price=3.0,
    )

    def _expected_for_store0(pidx: np.ndarray) -> int:
        prices = env.price_grid[pidx].astype(np.float64)
        mean_all, _ = _local_price_summary(
            prices, env._comp_indptr, env._comp_indices
        )
        mean_same, _ = _local_price_summary(
            prices, env._comp_indptr_same, env._comp_indices_same
        )
        bin_all = int(np.clip(
            np.digitize(mean_all[0], env._price_bin_edges) - 1, 0, B - 1
        ))
        bin_same = int(np.clip(
            np.digitize(mean_same[0], env._price_bin_edges) - 1, 0, B - 1
        ))
        return bin_all + bin_same * B

    pair_to_index: dict[tuple[int, int], int] = {}
    for p0 in range(B):
        for p1 in range(B):
            for p2 in range(B):
                pidx = np.array([p0, p1, p2, 0, 0, 0], dtype=np.int64)
                env._current_joint_actions_arr[:] = pidx
                sig = int(env.current_state_signal()[0])
                expected = _expected_for_store0(pidx)
                assert sig == expected
                bin_all = expected % B
                bin_same = expected // B
                pair_to_index[(bin_all, bin_same)] = sig

    assert len(pair_to_index) == len(set(pair_to_index.values()))


def test_no_same_type_rival_uses_own_price_bin() -> None:
    """Store with no same-type local rival bins same-type channel to own price."""
    firms, city = _six_store_city()
    B = 5
    env = HotellingMarketEnv(
        city=city,
        firms=firms,
        m=B,
        m_effort=1,
        state_mode="local_summary",
        local_summary_detailed=True,
        n_price_bins=B,
        min_price=0.0,
        max_price=4.0,
    )
    # store 5 is bio in cell 1 with bio store 4 -> same-type rival exists;
    # store 2 is standard, sole standard in cell 0 -> no same-type rival
    pidx = 2
    env._current_joint_actions_arr[:] = np.array(
        [0, 0, pidx, 0, 0, 0], dtype=np.int64
    )
    sig = env.current_state_signal()
    own_price = float(env.price_grid[pidx])
    own_bin = int(np.clip(
        np.digitize(own_price, env._price_bin_edges) - 1, 0, B - 1
    ))
    prices = env.price_grid[env._current_joint_actions_arr // env.m_effort].astype(
        np.float64
    )
    mean_all, _ = _local_price_summary(
        prices, env._comp_indptr, env._comp_indices
    )
    bin_all = int(np.clip(
        np.digitize(mean_all[2], env._price_bin_edges) - 1, 0, B - 1
    ))
    assert int(sig[2]) == bin_all + own_bin * B

    # store 5: bio with same-type rival 4 in cell 1
    pidx = 3
    env._current_joint_actions_arr[:] = np.array(
        [0, 0, 0, 0, 0, pidx], dtype=np.int64
    )
    sig = env.current_state_signal()
    prices = env.price_grid[env._current_joint_actions_arr // env.m_effort].astype(
        np.float64
    )
    mean_all, _ = _local_price_summary(
        prices, env._comp_indptr, env._comp_indices
    )
    mean_same, _ = _local_price_summary(
        prices, env._comp_indptr_same, env._comp_indices_same
    )
    bin_all = int(np.clip(
        np.digitize(mean_all[5], env._price_bin_edges) - 1, 0, B - 1
    ))
    bin_same = int(np.clip(
        np.digitize(mean_same[5], env._price_bin_edges) - 1, 0, B - 1
    ))
    assert int(sig[5]) == bin_all + bin_same * B
