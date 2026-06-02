"""Parity and timing test: dict-based step() vs array-based step_array().

Verifies that refactoring the env↔engine path in Prompt-3 produces
bit-for-bit identical demand/profit/price/effort trajectories.

Test outline
------------
1.  Build a small 3-firm city with a non-trivial dist2_km2.
2.  Generate a fixed sequence of random actions (200 steps × N firms).
3.  Run the dict path: ``env.step({agent_id: action})`` per step.
4.  Reset env, run the array path: ``env.step_array(actions_arr)`` per step.
5.  Assert ``np.allclose(…, atol=0, rtol=0)`` on prices, efforts, demands,
    profits at every step.
6.  Log a simple timing comparison (dict time / array time ratio).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from hotelling.core.city import City
from hotelling.core.firm import Firm
from hotelling.env.market_env import HotellingMarketEnv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

N_FIRMS = 3
M_PRICE = 5
M_EFFORT = 3
K_NEIGHBORS = 1
TRANSPORT_COST = 0.02
N_STEPS = 200


def _make_firms() -> list[Firm]:
    return [
        Firm(
            id=str(i),
            location=(0.2 * (i + 1), 0.5),
            marginal_cost=1.0 + 0.05 * i,
            quality=float(i) * 0.1,
            kappa0=1.0,
            size=1.0,
            rent=0.0,
        )
        for i in range(N_FIRMS)
    ]


def _make_city(firms: list[Firm]) -> City:
    N = len(firms)
    M_CELLS = 4
    rng = np.random.default_rng(0)
    dist2 = rng.uniform(0.01, 4.0, size=(M_CELLS, N))

    return City(
        boundary=(0.0, 0.0, 1.0, 1.0),
        population_grid=None,
        firms=firms,
        dist2_km2=dist2,
        cell_pop=rng.uniform(50.0, 200.0, size=M_CELLS),
        lambda_phi=rng.uniform(5.0, 30.0, size=M_CELLS),
        pi_H=rng.uniform(0.2, 0.8, size=M_CELLS),
        pi_H_lambda_phi=rng.uniform(0.2, 0.8, size=M_CELLS),
        alpha=np.array([0.0, 0.5]),
        beta=0.1,
        mu=0.25,
        a0=-5.0,
    )


def _make_env(firms: list[Firm], city: City) -> HotellingMarketEnv:
    return HotellingMarketEnv(
        city=city,
        firms=firms,
        m=M_PRICE,
        m_effort=M_EFFORT,
        k_neighbors=K_NEIGHBORS,
        transport_cost=TRANSPORT_COST,
        min_price=1.0,
        max_price=3.0,
    )


def _random_actions(env: HotellingMarketEnv, n_steps: int, seed: int = 42) -> np.ndarray:
    """Pre-generate (n_steps, N) array of random joint action indices."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, env._action_size, size=(n_steps, N_FIRMS), dtype=np.int64)


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------

class TestStepArrayParity:
    """Dict path and array path must produce identical trajectories."""

    def _run_dict_path(
        self,
        env: HotellingMarketEnv,
        action_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run N_STEPS via env.step(dict); return (prices, efforts, demands, profits)."""
        firms = env.firms
        T, N = action_seq.shape
        prices_out = np.empty((T, N), dtype=np.float64)
        efforts_out = np.empty((T, N), dtype=np.float64)
        demands_out = np.empty((T, N), dtype=np.float64)
        profits_out = np.empty((T, N), dtype=np.float64)

        for t in range(T):
            actions_dict = {str(firms[i].id): int(action_seq[t, i]) for i in range(N)}
            _obs, rewards_d, _term, _trunc, infos = env.step(actions_dict)
            for i, firm in enumerate(firms):
                aid = str(firm.id)
                prices_out[t, i] = infos[aid]["price"]
                efforts_out[t, i] = infos[aid]["effort"]
                demands_out[t, i] = infos[aid]["demand"]
                profits_out[t, i] = rewards_d[aid]

        return prices_out, efforts_out, demands_out, profits_out

    def _run_array_path(
        self,
        env: HotellingMarketEnv,
        action_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run N_STEPS via env.step_array(arr); return (prices, efforts, demands, profits)."""
        T, N = action_seq.shape
        prices_out = np.empty((T, N), dtype=np.float64)
        efforts_out = np.empty((T, N), dtype=np.float64)
        demands_out = np.empty((T, N), dtype=np.float64)
        profits_out = np.empty((T, N), dtype=np.float64)

        m_effort = env.m_effort

        for t in range(T):
            _neighbor_actions, profits, demands = env.step_array(action_seq[t])
            # Decode prices / efforts from the updated joint-action array
            joint = env._current_joint_actions_arr
            pidx = joint // m_effort
            eidx = joint % m_effort
            prices_out[t] = env.price_grid[pidx]
            efforts_out[t] = env.effort_grid[eidx]
            demands_out[t] = demands
            profits_out[t] = profits

        return prices_out, efforts_out, demands_out, profits_out

    def test_identical_trajectories(self) -> None:
        """200 steps via dict path == 200 steps via array path, bit-for-bit."""
        firms = _make_firms()
        city = _make_city(firms)
        env = _make_env(firms, city)
        action_seq = _random_actions(env, N_STEPS)

        # --- Dict path ---
        env.reset()
        t0 = time.perf_counter()
        d_prices, d_efforts, d_demands, d_profits = self._run_dict_path(env, action_seq)
        t_dict = time.perf_counter() - t0

        # --- Array path ---
        env.reset()
        t1 = time.perf_counter()
        a_prices, a_efforts, a_demands, a_profits = self._run_array_path(env, action_seq)
        t_arr = time.perf_counter() - t1

        logger.info(
            "step timing — dict: %.4f s  array: %.4f s  ratio dict/array: %.2f×",
            t_dict, t_arr, t_dict / max(t_arr, 1e-9),
        )

        assert np.allclose(d_prices,  a_prices,  atol=0, rtol=0), "prices differ"
        assert np.allclose(d_efforts, a_efforts, atol=0, rtol=0), "efforts differ"
        assert np.allclose(d_demands, a_demands, atol=0, rtol=0), "demands differ"
        assert np.allclose(d_profits, a_profits, atol=0, rtol=0), "profits differ"

    def test_neighbor_actions_consistent_with_dict_obs(self) -> None:
        """step_array neighbor_actions must match what _build_observation returns."""
        firms = _make_firms()
        city = _make_city(firms)
        env = _make_env(firms, city)
        env.reset()

        # Use a fixed single step
        rng = np.random.default_rng(7)
        actions_arr = rng.integers(0, env._action_size, size=N_FIRMS, dtype=np.int64)

        # Array path
        neighbor_arr, _, _ = env.step_array(actions_arr)
        # Sync dict (step_array intentionally does not sync it; use step() for that)
        for i, firm in enumerate(firms):
            env._current_joint_actions[str(firm.id)] = int(
                env._current_joint_actions_arr[i]
            )

        # Dict path: build observations the slow way
        for i, firm in enumerate(firms):
            obs = env._build_observation(str(firm.id))
            expected = obs["neighbor_prev_actions"]
            actual = neighbor_arr[i].tolist()
            assert actual == expected, (
                f"firm {firm.id}: step_array neighbor {actual} != dict obs {expected}"
            )

    def test_reset_initialises_arr(self) -> None:
        """After reset(), _current_joint_actions_arr matches the dict state."""
        firms = _make_firms()
        city = _make_city(firms)
        env = _make_env(firms, city)

        # Run some steps to dirty the state
        env.reset()
        actions_arr = np.zeros(N_FIRMS, dtype=np.int64)
        env.step_array(actions_arr)

        # Reset and check arr == dict
        env.reset()
        for i, firm in enumerate(firms):
            aid = str(firm.id)
            assert int(env._current_joint_actions_arr[i]) == env._current_joint_actions[aid]

    def test_get_neighbor_actions_arr_matches_dict_obs(self) -> None:
        """get_neighbor_actions_arr() must replicate _build_observation padding."""
        firms = _make_firms()
        city = _make_city(firms)
        env = _make_env(firms, city)
        env.reset()

        arr = env.get_neighbor_actions_arr()  # (N, k)
        for i, firm in enumerate(firms):
            obs = env._build_observation(str(firm.id))
            assert arr[i].tolist() == obs["neighbor_prev_actions"]
