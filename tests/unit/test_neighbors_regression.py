"""Regression: default k-neighbors path unchanged after current_state_signal routing."""
from __future__ import annotations

import numpy as np

from hotelling.agents.batch_qlearning import BatchQLearningAgent
from hotelling.core.city import City
from hotelling.core.firm import Firm
from hotelling.core.equilibrium import bertrand_nash, joint_monopoly
from hotelling.env.market_env import HotellingMarketEnv
from hotelling.simulation.engine import BatchSimulationEngine


N_FIRMS = 3
M_PRICE = 5
M_EFFORT = 1
K_NEIGHBORS = 1
TRANSPORT_COST = 0.02
N_STEPS = 3000
SEED = 42


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


def _make_env_and_agent(
    firms: list[Firm], city: City
) -> tuple[HotellingMarketEnv, BatchQLearningAgent]:
    env = HotellingMarketEnv(
        city=city,
        firms=firms,
        m=M_PRICE,
        m_effort=M_EFFORT,
        k_neighbors=K_NEIGHBORS,
        transport_cost=TRANSPORT_COST,
        min_price=1.0,
        max_price=3.0,
        state_mode="neighbors",
    )
    agent = BatchQLearningAgent(
        n_agents=len(firms),
        m=M_PRICE,
        m_effort=M_EFFORT,
        k=K_NEIGHBORS,
        alpha=0.15,
        beta_decay=4e-6,
        delta=0.95,
        seed=SEED,
        state_mode="neighbors",
        state_size=env.state_size,
    )
    return env, agent


def _compute_delta(
    env: HotellingMarketEnv, city: City, final_prices: dict[str, float]
) -> float:
    p_nash_arr, _ = bertrand_nash(city, transport_cost=TRANSPORT_COST)
    p_mono_arr, _ = joint_monopoly(city, transport_cost=TRANSPORT_COST)
    p_nash = float(p_nash_arr.mean())
    p_mono = float(p_mono_arr.mean())
    mean_final = float(np.mean(list(final_prices.values())))
    denom = p_mono - p_nash
    if abs(denom) < 1e-9:
        return 0.0
    return float(np.clip((mean_final - p_nash) / denom, -0.5, 1.5))


def test_current_state_signal_equals_neighbor_actions() -> None:
    """In neighbors mode current_state_signal() == get_neighbor_actions_arr()."""
    firms = _make_firms()
    city = _make_city(firms)
    env, agent = _make_env_and_agent(firms, city)
    env.reset(seed=SEED)

    rng = np.random.default_rng(SEED)
    for _ in range(50):
        signal = env.current_state_signal()
        neighbors = env.get_neighbor_actions_arr()
        np.testing.assert_array_equal(signal, neighbors)

        actions = agent.act(signal)
        env.step_array(actions)
        post_step = env.current_state_signal()
        post_neighbors = env.get_neighbor_actions_arr()
        np.testing.assert_array_equal(post_step, post_neighbors)


def test_seeded_run_reproducible_delta() -> None:
    """Two identical seeded runs produce the same final delta (neighbors, k=1)."""
    firms = _make_firms()
    city = _make_city(firms)

    def _run_once() -> tuple[float, np.ndarray]:
        env, agent = _make_env_and_agent(firms, city)
        engine = BatchSimulationEngine(
            env=env,
            batch_agent=agent,
            max_steps=N_STEPS,
            record_every=1000,
        )
        result = engine.run(seed=SEED)
        delta = _compute_delta(env, city, result["final_prices"])
        q_snapshot = agent._q.copy()
        return delta, q_snapshot

    delta1, q1 = _run_once()
    delta2, q2 = _run_once()

    assert delta1 == delta2
    np.testing.assert_array_equal(q1, q2)
