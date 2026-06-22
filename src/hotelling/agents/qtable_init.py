"""Q-table initialization strategies for BatchQLearningAgent.

Four modes:
  zero         — Q_0 = 0 everywhere (current default)
  nash_anchor  — Q_{i,0}(s,a) = π_i(a; p^N_{-i}) / (1−δ)
  solve        — Calvano eq.(8) adapted: average over nearest neighbor's actions
  optimistic   — Q_{i,0}(s,a) = π_i^mono / (1−δ)  for all (s,a)

All modes produce a state-independent initialization: Q[i, s, a] is the same
for all states s (the prior over opponents doesn't condition on state). The
returned tensor shape is (N, state_size, action_size); broadcasting across the
state axis is done internally.

References:
    Calvano et al. (2020 AER) eq. (8);
    ADR-010 (entrant Q-table initialisation strategies).
"""
from __future__ import annotations

import enum
import logging
import time
from typing import Any

import numpy as np

from hotelling.core.market import FirmArrays, market_clearing_arrays

logger = logging.getLogger(__name__)


class QtableInitMode(enum.Enum):
    """Q-table initialization strategy selector."""

    ZERO = "zero"
    NASH_ANCHOR = "nash_anchor"
    SOLVE = "solve"
    OPTIMISTIC = "optimistic"

    @classmethod
    def from_cli(cls, s: str) -> "QtableInitMode":
        """Parse a CLI string like 'nash-anchor' into the enum value."""
        return cls(s.replace("-", "_"))


def compute_q_init(
    mode: QtableInitMode,
    *,
    env: Any,
    city: Any,
    n_agents: int,
    state_size: int,
    action_size: int,
    m: int,
    m_effort: int,
    delta: float,
    p_nash_arr: np.ndarray,
    p_mono_arr: np.ndarray | None = None,
    transport_cost: float = 0.01,
) -> np.ndarray:
    """Compute the initial Q-table tensor.

    Parameters
    ----------
    mode : QtableInitMode
        Which initialization strategy to use.
    env : HotellingMarketEnv
        Fully constructed environment (needed for price grids, firm arrays,
        neighbor indices).
    city : City
        Spatial container (needed for demand computation).
    n_agents : int
        Number of stores (= N).
    state_size : int
        Number of Q-table state slots per store.
    action_size : int
        Number of Q-table action slots per store (= m * m_effort).
    m : int
        Number of discrete price levels.
    m_effort : int
        Number of discrete effort levels.
    delta : float
        Discount factor δ ∈ (0, 1).
    p_nash_arr : (N,) float64
        Pre-computed Bertrand-Nash equilibrium prices.
    p_mono_arr : (N,) float64 or None
        Pre-computed joint-monopoly prices (required for 'optimistic' mode).
    transport_cost : float
        Transport disutility coefficient.

    Returns
    -------
    (N, state_size, action_size) float64 array.
    """
    if mode == QtableInitMode.ZERO:
        logger.info("Q-table init: ZERO (all zeros).")
        return np.zeros((n_agents, state_size, action_size), dtype=np.float64)

    firm_arrays: FirmArrays = env._firm_arrays
    t0 = time.time()

    if mode == QtableInitMode.NASH_ANCHOR:
        q = _init_nash_anchor(
            env=env, city=city, N=n_agents, state_size=state_size,
            action_size=action_size, m=m, m_effort=m_effort,
            delta=delta, p_nash_arr=p_nash_arr,
            transport_cost=transport_cost, firm_arrays=firm_arrays,
        )
    elif mode == QtableInitMode.SOLVE:
        q = _init_solve(
            env=env, city=city, N=n_agents, state_size=state_size,
            action_size=action_size, m=m, m_effort=m_effort,
            delta=delta, p_nash_arr=p_nash_arr,
            transport_cost=transport_cost, firm_arrays=firm_arrays,
        )
    elif mode == QtableInitMode.OPTIMISTIC:
        if p_mono_arr is None:
            raise ValueError(
                "optimistic Q-table init requires p_mono_arr (joint-monopoly "
                "prices), but None was provided."
            )
        q = _init_optimistic(
            env=env, city=city, N=n_agents, state_size=state_size,
            action_size=action_size, delta=delta, p_mono_arr=p_mono_arr,
            transport_cost=transport_cost, firm_arrays=firm_arrays,
        )
    else:
        raise ValueError(f"Unknown Q-table init mode: {mode}")

    elapsed = time.time() - t0
    logger.info(
        "Q-table init %s: shape=%s, mean=%.6f, std=%.6f, "
        "min=%.6f, max=%.6f, elapsed=%.1fs",
        mode.value, q.shape,
        float(q.mean()), float(q.std()),
        float(q.min()), float(q.max()),
        elapsed,
    )
    return q


def _get_store_price(env: Any, store_idx: int, price_idx: int) -> float:
    """Decode a single store's price from its action index.

    Handles chain-specific grids when active.
    """
    if env._store_price_grids is not None:
        return float(env._store_price_grids[store_idx, price_idx])
    return float(env.price_grid[price_idx])


def _init_nash_anchor(
    *,
    env: Any,
    city: Any,
    N: int,
    state_size: int,
    action_size: int,
    m: int,
    m_effort: int,
    delta: float,
    p_nash_arr: np.ndarray,
    transport_cost: float,
    firm_arrays: FirmArrays,
) -> np.ndarray:
    r"""Nash-anchor initialization.

    .. math::
        Q_{i,0}(s, a_i) = \frac{\pi_i(a_i;\; \bar{p}^N_{-i})}{1 - \delta}
        \quad \forall s

    For each store *i* and each own action *a_i*, hold all other stores at
    their Bertrand-Nash prices and compute store *i*'s profit when playing
    action *a_i*. The resulting Q-value is the discounted present value of
    this per-period profit (under the fiction that the situation persists
    forever at discount δ).

    Computational cost: N × m × m_effort calls to market_clearing_arrays.
    With N=494, m=15, m_effort=1: 7,410 calls ≈ 20–30 seconds.
    """
    q_action = np.zeros((N, action_size), dtype=np.float64)
    inv_one_minus_delta = 1.0 / (1.0 - delta)

    total_iters = N * m * m_effort
    log_interval = max(1, total_iters // 10)
    count = 0

    for p_idx in range(m):
        for e_idx in range(m_effort):
            a = p_idx * m_effort + e_idx
            for i in range(N):
                prices = p_nash_arr.copy()
                prices[i] = _get_store_price(env, i, p_idx)
                efforts = np.zeros(N, dtype=np.float64)
                efforts[i] = env.effort_grid[e_idx]

                _, profits = market_clearing_arrays(
                    prices, efforts, city, transport_cost, firm_arrays,
                )
                q_action[i, a] = profits[i] * inv_one_minus_delta

                count += 1
                if count % log_interval == 0:
                    logger.info(
                        "  nash_anchor: %d/%d (%.0f%%)",
                        count, total_iters, 100.0 * count / total_iters,
                    )

    # Broadcast across all states (state-independent initialization)
    return np.broadcast_to(
        q_action[:, np.newaxis, :], (N, state_size, action_size)
    ).copy()


def _init_solve(
    *,
    env: Any,
    city: Any,
    N: int,
    state_size: int,
    action_size: int,
    m: int,
    m_effort: int,
    delta: float,
    p_nash_arr: np.ndarray,
    transport_cost: float,
    firm_arrays: FirmArrays,
) -> np.ndarray:
    r"""Calvano eq. (8) adapted for the spatial heterogeneous setting.

    .. math::
        Q_{i,0}(s, a_i) = \frac{1}{(1-\delta)\, m_j}
            \sum_{a_j=0}^{m_j - 1} \pi_i(a_i, a_j;\; \bar{p}^N_{-\{i,j\}})
            \quad \forall s

    where *j* is store *i*'s nearest spatial neighbor (from
    ``env._neighbor_idx[:, 0]``), and all stores outside the *(i, j)* pair
    are held at their Bertrand-Nash prices. Each action *a_j* is a
    *price-only* index on neighbor *j*'s grid (effort fixed at zero for
    the neighbor).

    Stores with no valid neighbor (sentinel index = N) fall back to the
    :func:`_init_nash_anchor` formula for that store.

    Computational cost: N × m × m_effort × m_j calls to market_clearing.
    With N=494, m=m_j=15, m_effort=1: ~111K calls ≈ 5–8 minutes.
    """
    q_action = np.zeros((N, action_size), dtype=np.float64)
    inv_one_minus_delta = 1.0 / (1.0 - delta)

    # Nearest neighbor for each store
    nbr_idx = env._neighbor_idx  # (N, k) int32
    k = nbr_idx.shape[1]

    total_stores = N
    log_interval = max(1, N // 20)

    for i in range(N):
        j = int(nbr_idx[i, 0]) if k >= 1 else N  # sentinel N = no neighbor

        if j >= N:
            # No valid neighbor: fall back to nash-anchor for this store
            for p_idx in range(m):
                for e_idx in range(m_effort):
                    a = p_idx * m_effort + e_idx
                    prices = p_nash_arr.copy()
                    prices[i] = _get_store_price(env, i, p_idx)
                    efforts = np.zeros(N, dtype=np.float64)
                    efforts[i] = env.effort_grid[e_idx]
                    _, profits = market_clearing_arrays(
                        prices, efforts, city, transport_cost, firm_arrays,
                    )
                    q_action[i, a] = profits[i] * inv_one_minus_delta
        else:
            # Average over neighbor j's action space
            m_j_price = m  # neighbor uses same m (grid size is uniform)

            for p_idx in range(m):
                for e_idx in range(m_effort):
                    a = p_idx * m_effort + e_idx
                    profit_sum = 0.0

                    for p_j in range(m_j_price):
                        # Neighbor takes each price action with uniform prob;
                        # neighbor effort fixed at 0 (price-only averaging)
                        prices = p_nash_arr.copy()
                        prices[i] = _get_store_price(env, i, p_idx)
                        prices[j] = _get_store_price(env, j, p_j)
                        efforts = np.zeros(N, dtype=np.float64)
                        efforts[i] = env.effort_grid[e_idx]

                        _, profits = market_clearing_arrays(
                            prices, efforts, city, transport_cost, firm_arrays,
                        )
                        profit_sum += profits[i]

                    q_action[i, a] = (profit_sum / m_j_price) * inv_one_minus_delta

        if (i + 1) % log_interval == 0:
            logger.info(
                "  solve: store %d/%d (%.0f%%)",
                i + 1, N, 100.0 * (i + 1) / N,
            )

    return np.broadcast_to(
        q_action[:, np.newaxis, :], (N, state_size, action_size)
    ).copy()


def _init_optimistic(
    *,
    env: Any,
    city: Any,
    N: int,
    state_size: int,
    action_size: int,
    delta: float,
    p_mono_arr: np.ndarray,
    transport_cost: float,
    firm_arrays: FirmArrays,
) -> np.ndarray:
    r"""Optimistic initialization: all Q-values set to monopoly profit level.

    .. math::
        Q_{i,0}(s, a) = \frac{\pi_i^{\text{mono}}}{1 - \delta}
        \quad \forall s, a

    Every (state, action) cell for store *i* is set to the discounted
    present value of store *i*'s joint-monopoly profit. This creates an
    optimistic prior that encourages exploration: the agent keeps trying
    actions whose Q-values haven't been corrected downward by TD updates.
    Because the true reward for most actions is below monopoly, Q-values
    decrease with experience, and seldom-visited cells remain at the high
    initial level — guaranteeing they will be explored.

    In a multi-agent setting, simultaneous optimistic initialization biases
    all agents toward high prices in early rounds (before ε has decayed),
    which may bootstrap supra-competitive coordination.

    Computational cost: 1 call to market_clearing_arrays.
    """
    inv_one_minus_delta = 1.0 / (1.0 - delta)
    efforts_zero = np.zeros(N, dtype=np.float64)

    _, pi_mono = market_clearing_arrays(
        np.ascontiguousarray(p_mono_arr, dtype=np.float64),
        efforts_zero,
        city,
        transport_cost,
        firm_arrays,
    )

    # q_action[i, :] = pi_mono[i] / (1 - delta) for all actions
    q_action = np.outer(pi_mono * inv_one_minus_delta, np.ones(action_size))

    logger.info(
        "  optimistic: π_mono mean=%.6f, Q_init mean=%.6f",
        float(pi_mono.mean()),
        float(q_action.mean()),
    )

    return np.broadcast_to(
        q_action[:, np.newaxis, :], (N, state_size, action_size)
    ).copy()
