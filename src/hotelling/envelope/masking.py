"""Translate CEO strategy envelopes into per-store action masks + epsilon.

The mask constrains the absolute (nominal-price) joint action grid of the
BatchQLearningAgent to the CEO's [p_bar +/- delta_p] (and effort band when
effort is active). Q-values over the fixed grid stay valid across epochs;
only the feasible set changes. See ADR-009 and the CEO layer design.

Public API: build_action_mask_and_epsilon
"""
from __future__ import annotations

import numpy as np

from hotelling.llm.schemas import ChainEnvelopeOutput


def _allowed_grid_indices(
    grid: np.ndarray, centre: float, half_width: float
) -> np.ndarray:
    """Indices of grid points within [centre - hw, centre + hw].

    Guarantees at least one index: if the band contains no grid point, returns
    the single nearest index to ``centre`` (snap-to-nearest). This makes the
    mask robust to bands narrower than the grid step at euro scale.
    """
    lo, hi = centre - half_width, centre + half_width
    idx = np.nonzero((grid >= lo) & (grid <= hi))[0]
    if idx.size == 0:
        idx = np.array([int(np.argmin(np.abs(grid - centre)))], dtype=np.int64)
    return idx.astype(np.int64)


def build_action_mask_and_epsilon(
    chain_envelopes: dict[str, ChainEnvelopeOutput],
    store_chain: list[str],
    store_group_labels: list[str],
    price_grid: np.ndarray,
    effort_grid: np.ndarray,
    m_effort: int,
    mask_effort: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (N, m*m_effort) action mask and (N,) epsilon vector.

    Joint action encoding (matches HotellingMarketEnv): a = price_idx * m_effort + effort_idx.

    Parameters
    ----------
    chain_envelopes : brand -> validated ChainEnvelopeOutput for the current epoch.
        Every brand present in ``store_chain`` must have an entry.
    store_chain : (N,) brand per store, canonical order (firm.chain).
    store_group_labels : (N,) composite group label per store (from assign_groups).
    price_grid : (m,) nominal price levels.
    effort_grid : (m_effort,) effort levels.
    m_effort : number of effort levels (1 in the price-only baseline).
    mask_effort : if True (m_effort > 1) also constrain effort to [e_bar +/- delta_e];
        if False, all effort indices are allowed (effort dimension inert).

    Returns
    -------
    mask : (N, m*m_effort) bool — True = allowed.
    eps  : (N,) float — per-store exploration rate from the group envelope.
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
        p_idx = _allowed_grid_indices(price_grid, env.p_bar, env.delta_p)
        if mask_effort and m_effort > 1:
            e_idx = _allowed_grid_indices(effort_grid, env.e_bar, env.delta_e)
        else:
            e_idx = all_effort
        joint = (p_idx[:, None] * m_effort + e_idx[None, :]).ravel()
        mask[i, joint] = True
        eps[i] = float(env.epsilon)

    return mask, eps
