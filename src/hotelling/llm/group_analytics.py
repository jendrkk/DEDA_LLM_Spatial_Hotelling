"""Per-group competitive analytics for the LLM-CEO prompt (baseline 2.0).

Pure-Python digests computed once per CEO epoch from the rolling window and
static store metadata — no extra LLM calls. Each brief summarises one store
group within a chain: dispersion, margin, demand-vs-store-share index, local
competition intensity, same-tier rival price gap, position label, price trend.

Public API: compute_group_briefs
"""
from __future__ import annotations

import numpy as np


def compute_group_briefs(
    window_arrays: dict,
    *,
    chain_id: str,
    store_chain: list,
    store_chain_type: list,
    store_group_labels: list,
    store_metadata: list,
    group_keys: list,
    marginal_cost: float,
    position_tol: float = 0.5,
) -> dict[str, dict]:
    """Return ``{group_key: brief}`` for every non-empty group of ``chain_id``.

    Parameters
    ----------
    window_arrays : output of RollingWindow.arrays() — dict of (W, N) arrays
        with keys ``price``, ``effort``, ``demand``, ``profit`` (decoded values).
    store_metadata : list of per-store dicts (canonical order) from
        build_store_metadata; must contain ``n_rivals_within_R``.
    position_tol : € band around the same-tier mean within which a group is
        labelled "at_market" rather than "undercut"/"premium".

    Brief fields
    ------------
    price_std, margin_eur, demand_index, store_share_pct, mean_local_competition,
    same_tier_rival_mean_price, price_gap_vs_same_tier, position_label, price_trend_pct.
    """
    price = np.asarray(window_arrays["price"], dtype=np.float64)      # (W, N)
    demand = np.asarray(window_arrays["demand"], dtype=np.float64)    # (W, N)
    chain = np.array(store_chain, dtype=object)
    ctype = np.array(store_chain_type, dtype=object)
    labels = np.array(store_group_labels, dtype=object)
    n_rivals = np.array(
        [float(m.get("n_rivals_within_R", 0)) for m in store_metadata], dtype=np.float64
    )

    own_mask = chain == chain_id
    if not own_mask.any():
        return {}
    own_type = ctype[own_mask][0]
    own_demand_total = float(demand[:, own_mask].sum())
    own_store_count = int(own_mask.sum())
    # Same-tier rivals: OTHER chains of the same chain_type (direct competitors).
    same_tier = (~own_mask) & (ctype == own_type)
    same_tier_price = float(price[:, same_tier].mean()) if same_tier.any() else float("nan")

    briefs: dict[str, dict] = {}
    for key in group_keys:
        gm = own_mask & (labels == key)
        if not gm.any():
            continue
        gprice = price[:, gm]
        mean_price = float(gprice.mean())
        g_demand = float(demand[:, gm].sum())
        demand_share = (100.0 * g_demand / own_demand_total) if own_demand_total > 0 else 0.0
        store_share = (100.0 * int(gm.sum()) / own_store_count) if own_store_count > 0 else 0.0
        demand_index = (demand_share / store_share) if store_share > 0 else 0.0
        tier_price = same_tier_price if np.isfinite(same_tier_price) else mean_price
        gap = mean_price - tier_price
        if gap < -position_tol:
            position = "undercut"
        elif gap > position_tol:
            position = "premium"
        else:
            position = "at_market"
        W = gprice.shape[0]
        if W >= 2:
            half = W // 2
            first = float(gprice[:half].mean())
            second = float(gprice[half:].mean())
            trend = (100.0 * (second - first) / first) if first else 0.0
        else:
            trend = 0.0
        briefs[key] = {
            "price_std": float(gprice.std()),
            "margin_eur": mean_price - float(marginal_cost),
            "demand_index": float(demand_index),
            "store_share_pct": float(store_share),
            "mean_local_competition": float(n_rivals[gm].mean()),
            "same_tier_rival_mean_price": float(tier_price),
            "price_gap_vs_same_tier": float(gap),
            "position_label": position,
            "price_trend_pct": float(trend),
        }
    return briefs
