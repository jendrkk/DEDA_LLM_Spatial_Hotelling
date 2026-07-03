"""Strategic analytics for the LLM-CEO prompt: monopoly headroom + coordinated
tier counterfactual.

Pure-analytic per-epoch digests (no extra LLM calls). Two blocks per chain
(brand):

``headroom``
    Where the brand's chain type sits between its Bertrand-Nash floor and its
    joint-monopoly ceiling (from hotelling.core.equilibrium). Gives the CEO the
    ceiling reference it otherwise lacks.

``tier_counterfactual``
    The brand's gross variable profit Σ(p−c)·D under three discrete moves,
    relative to holding: (1) the whole chain TYPE steps up one grid step
    together (reciprocal), (2) two steps together, (3) the brand alone steps up
    one step (unilateral). The unilateral-vs-reciprocal contrast is the point:
    at the near-Nash attractor the unilateral move is ≈0 / negative (which is
    why the tier never rises on its own), while the coordinated move is
    positive — the signal the CEO needs to justify a coordinated ratchet.

Demand is evaluated with hotelling.core.market.market_clearing, which
auto-dispatches to the same (catchment) kernel the substrate optimises against,
so the counterfactual is faithful to the discrete action space and the realised
demand model.

Public API: compute_strategic_analytics
"""
from __future__ import annotations

import numpy as np

_CLIP_LO, _CLIP_HI = -0.5, 1.5


def _store_grid(env, i: int) -> np.ndarray:
    """Return store i's own price lattice (chain-specific grid when active)."""
    sg = getattr(env, "_store_price_grids", None)
    return sg[i] if sg is not None else env.price_grid


def _raise_prices_by_steps(
    env, prices: np.ndarray, mask: np.ndarray, k: int
) -> np.ndarray:
    """Snap the masked stores to grid and move them up ``k`` grid steps (clipped).

    Off-grid ``prices`` (windowed means) are first snapped to the nearest grid
    index of each store's own lattice, then advanced by ``k`` and clipped to the
    top of the grid, so the counterfactual price vector is action-space-feasible.
    """
    p = np.asarray(prices, dtype=np.float64).copy()
    m = int(env.m)
    idxs = np.nonzero(mask)[0]
    for i in idxs:
        g = _store_grid(env, int(i))
        j = int(np.argmin(np.abs(g - prices[i])))
        j = min(j + k, m - 1)
        p[i] = float(g[j])
    return p


def _chain_gross_profit(
    prices: np.ndarray, demand: np.ndarray, costs: np.ndarray, brand_mask: np.ndarray
) -> float:
    """Σ_{i∈brand} (p_i − c_i) · D_i for one brand at a given price/demand vector."""
    return float(((prices[brand_mask] - costs[brand_mask]) * demand[brand_mask]).sum())


def compute_strategic_analytics(
    *,
    env,
    current_prices: np.ndarray,          # (N,) EUR, windowed mean per store
    store_chain: list[str],
    store_chain_type: list[str],
    p_nash_arr: np.ndarray,              # (N,)
    p_mono_arr: np.ndarray,              # (N,)
    marginal_costs: np.ndarray,          # (N,)
    efforts: np.ndarray | None = None,   # (N,) or None -> zeros (price-only)
) -> dict[str, dict]:
    """Return ``{brand: {"headroom": {...}, "tier_counterfactual": {...}}}``.

    Computed once per CEO epoch. Demand at every scenario uses
    :func:`hotelling.core.market.market_clearing` (auto-dispatch, same path as
    the simulation). ``current_prices`` are the windowed mean per-store prices.
    """
    from hotelling.core.market import market_clearing

    N = len(store_chain)
    chain = np.asarray(store_chain, dtype=object)
    ctype = np.asarray(store_chain_type, dtype=object)
    costs = np.asarray(marginal_costs, dtype=np.float64)
    p_now = np.asarray(current_prices, dtype=np.float64)
    p_nash = np.asarray(p_nash_arr, dtype=np.float64)
    p_mono = np.asarray(p_mono_arr, dtype=np.float64)
    e0 = np.zeros(N, dtype=np.float64) if efforts is None else np.asarray(efforts, np.float64)

    city = env.city
    tc = float(env.transport_cost)

    # Per-chain-type step size (chain-specific grid when active, else global).
    type_step = {}
    for ct in ("discount", "standard", "bio"):
        try:
            type_step[ct] = float(env.grid_spec(ct).get("step", 0.0))
        except Exception:  # noqa: BLE001
            type_step[ct] = float(env.grid_spec(None).get("step", 0.0))

    type_masks = {ct: (ctype == ct) for ct in ("discount", "standard", "bio")}

    # --- Demand scenarios shared across brands of the same type ---------------
    d_hold, _ = market_clearing(p_now, e0, city, tc)
    d_up_recip: dict[tuple[str, int], np.ndarray] = {}
    for ct, mask_ct in type_masks.items():
        if not mask_ct.any():
            continue
        for k in (1, 2):
            p_k = _raise_prices_by_steps(env, p_now, mask_ct, k)
            d_k, _ = market_clearing(p_k, e0, city, tc)
            d_up_recip[(ct, k)] = d_k

    out: dict[str, dict] = {}
    for brand in sorted({str(c) for c in store_chain}):
        b_mask = chain == brand
        if not b_mask.any():
            continue
        ct = str(ctype[b_mask][0])
        ct_mask = type_masks.get(ct, np.zeros(N, dtype=bool))

        # ---- headroom (per type) --------------------------------------------
        nash_ct = float(p_nash[ct_mask].mean()) if ct_mask.any() else float("nan")
        mono_ct = float(p_mono[ct_mask].mean()) if ct_mask.any() else float("nan")
        cur_own = float(p_now[b_mask].mean())
        step_ct = type_step.get(ct, 0.0) or float("nan")
        span = mono_ct - nash_ct
        surplus = (float(np.clip((cur_own - nash_ct) / span, _CLIP_LO, _CLIP_HI))
                   if abs(span) > 1e-9 else float("nan"))
        room_eur = mono_ct - cur_own
        room_steps = (room_eur / step_ct) if step_ct and step_ct == step_ct else float("nan")
        headroom = {
            "chain_type": ct,
            "current_price": cur_own,
            "nash_price": nash_ct,
            "mono_price": mono_ct,
            "surplus_captured_pct": 100.0 * surplus if surplus == surplus else float("nan"),
            "room_to_mono_eur": float(room_eur),
            "room_to_mono_steps": float(room_steps),
            "grid_step_eur": float(step_ct) if step_ct == step_ct else float("nan"),
        }

        # ---- tier counterfactual (per brand) --------------------------------
        pi_hold = _chain_gross_profit(p_now, d_hold, costs, b_mask)

        def _pct(pi_new: float) -> float:
            return (100.0 * (pi_new - pi_hold) / pi_hold) if abs(pi_hold) > 1e-9 else float("nan")

        p_up1 = _raise_prices_by_steps(env, p_now, ct_mask, 1)
        p_up2 = _raise_prices_by_steps(env, p_now, ct_mask, 2)
        pi_up1_recip = _chain_gross_profit(p_up1, d_up_recip.get((ct, 1), d_hold), costs, b_mask)
        pi_up2_recip = _chain_gross_profit(p_up2, d_up_recip.get((ct, 2), d_hold), costs, b_mask)

        p_uni = _raise_prices_by_steps(env, p_now, b_mask, 1)
        d_uni, _ = market_clearing(p_uni, e0, city, tc)
        pi_uni1 = _chain_gross_profit(p_uni, d_uni, costs, b_mask)

        tier_counterfactual = {
            "profit_hold_gross": float(pi_hold),
            "up1_reciprocal_pct": _pct(pi_up1_recip),
            "up2_reciprocal_pct": _pct(pi_up2_recip),
            "up1_unilateral_pct": _pct(pi_uni1),
            "grid_step_eur": float(step_ct) if step_ct == step_ct else float("nan"),
        }

        out[brand] = {"headroom": headroom, "tier_counterfactual": tier_counterfactual}
    return out
