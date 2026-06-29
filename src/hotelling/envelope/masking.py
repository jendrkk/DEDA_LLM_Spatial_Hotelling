"""Translate CEO strategy envelopes into per-store action masks + epsilon.

The mask constrains the absolute (nominal-price) joint action grid of the
BatchQLearningAgent to the CEO's asymmetric price band
[p_bar - dp_minus, p_bar + dp_plus] (and effort band when effort is active).
Q-values over the fixed grid stay valid across epochs; only the feasible set
changes. See ADR-009 and the CEO layer design.

Hard feasibility guarantee
--------------------------
Every store is given at least TWO contiguous allowed price-grid points, even if
the CEO's euro band is narrower than one grid step. This makes a sub-step p_bar
change impossible to turn into a no-op and always leaves the store room to
explore within the envelope. See _allowed_grid_indices_asym.

Public API: build_action_mask_and_epsilon
"""
from __future__ import annotations

import numpy as np

from hotelling.llm.schemas import ChainEnvelopeOutput

# Defensive epsilon clamp (matches GroupEnvelope's (0, 0.25) bound; guards against
# any malformed value reaching the agent and destabilising the substrate).
_EPS_LO = 1.0e-3
_EPS_HI = 0.25


def _allowed_grid_indices_asym(
    grid: np.ndarray, centre: float, dp_minus: float, dp_plus: float
) -> np.ndarray:
    """Indices of grid points within [centre - dp_minus, centre + dp_plus].

    Hard feasibility guarantee: ALWAYS returns at least two contiguous indices.
    If the euro band contains < 2 grid points, the window is widened around the
    nearest index ``i*`` to include ``i*`` and one neighbour (preferring the side
    with more headroom inside the grid). This guarantees that (a) the store can
    always explore at least 2 prices within the envelope, and (b) a one-step
    change in ``centre`` shifts ``i*`` and therefore the feasible window.
    """
    m = int(grid.shape[0])
    lo, hi = centre - dp_minus, centre + dp_plus
    idx = np.nonzero((grid >= lo) & (grid <= hi))[0].astype(np.int64)
    if idx.size >= 2:
        return idx

    # Fewer than 2 points inside the band → snap-and-widen to guarantee 2.
    i_star = int(np.argmin(np.abs(grid - centre)))
    if m == 1:
        return np.array([0], dtype=np.int64)  # degenerate grid; nothing else possible
    if i_star == 0:
        pair = (0, 1)
    elif i_star == m - 1:
        pair = (m - 2, m - 1)
    else:
        # Choose the neighbour on the side of ``centre`` with the larger gap,
        # so the pair brackets the target rather than sitting entirely to one side.
        left_gap = centre - grid[i_star - 1]
        right_gap = grid[i_star + 1] - centre
        if right_gap >= left_gap:
            pair = (i_star, i_star + 1)
        else:
            pair = (i_star - 1, i_star)
    return np.array(pair, dtype=np.int64)


def build_action_mask_and_epsilon(
    chain_envelopes: dict[str, ChainEnvelopeOutput],
    store_chain: list[str],
    store_group_labels: list[str],
    price_grid: np.ndarray,
    effort_grid: np.ndarray,
    m_effort: int,
    mask_effort: bool,
    store_price_grids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (N, m*m_effort) action mask and (N,) epsilon vector.

    Joint action encoding (matches HotellingMarketEnv): a = price_idx * m_effort + effort_idx.

    Uses the asymmetric price band [p_bar - dp_minus, p_bar + dp_plus] from each
    group envelope (GroupEnvelope guarantees dp_minus/dp_plus are populated even
    for legacy delta_p-only envelopes). Effort, when active, still uses a
    symmetric [e_bar +/- delta_e] band. Epsilon is clamped to [_EPS_LO, _EPS_HI].

    Parameters
    ----------
    store_price_grids : optional (N, m) per-store price grids. When provided each
        store's envelope EUR band is matched against its OWN chain-specific grid;
        None uses ``price_grid`` for all stores.

    Returns
    -------
    mask : (N, m*m_effort) bool — True = allowed (every row has >= 2 True price
        columns by construction).
    eps  : (N,) float — per-store exploration rate, clamped to [_EPS_LO, _EPS_HI].
    """
    N = len(store_chain)
    m = int(len(price_grid))
    action_size = m * m_effort
    mask = np.zeros((N, action_size), dtype=bool)
    eps = np.empty(N, dtype=np.float64)
    all_effort = np.arange(m_effort, dtype=np.int64)

    for i in range(N):
        brand = store_chain[i]
        if brand not in chain_envelopes:
            raise KeyError(
                f"No envelope for chain {brand!r} at store index {i}; "
                "ensure every CEO is called before building the mask."
            )
        groups = chain_envelopes[brand].groups
        label = store_group_labels[i]
        env = groups.get(label) or next(iter(groups.values()))  # fallback: first group
        _grid_i = store_price_grids[i] if store_price_grids is not None else price_grid
        p_idx = _allowed_grid_indices_asym(
            _grid_i, float(env.p_bar), float(env.dp_minus), float(env.dp_plus)
        )
        if mask_effort and m_effort > 1:
            # Effort stays symmetric; reuse the asymmetric helper with equal widths.
            e_idx = _allowed_grid_indices_asym(
                effort_grid, float(env.e_bar), float(env.delta_e), float(env.delta_e)
            )
        else:
            e_idx = all_effort
        joint = (p_idx[:, None] * m_effort + e_idx[None, :]).ravel()
        mask[i, joint] = True
        eps[i] = float(np.clip(env.epsilon, _EPS_LO, _EPS_HI))

    return mask, eps
