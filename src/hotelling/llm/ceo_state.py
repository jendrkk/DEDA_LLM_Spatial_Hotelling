"""CEO state assembly: rolling-window aggregates + static consumer zones.

Produces the dict consumed by llm/prompts/state_ceo.jinja. Information is
chain-private aggregates + PUBLIC rival prices + zone demographics only
(ADR-007 information segregation).

Public API: RollingWindow, build_consumer_zones, division_context, build_ceo_state
"""
from __future__ import annotations

import numpy as np

_CT_CODE = {"discount": "D", "standard": "S", "bio": "B"}
_CT_LABEL = {"discount": "Discount", "standard": "Standard", "bio": "Bio"}


def ct_code(chain_type: str) -> str:
    return _CT_CODE.get(str(chain_type).lower(), "S")


def ct_label(chain_type: str) -> str:
    return _CT_LABEL.get(str(chain_type).lower(), "Standard")


class RollingWindow:
    """Fixed-size ring buffer of the last ``W`` periods of (price, effort, demand, profit).

    Stores DECODED values (euro prices, effort levels, demand units, profit euros),
    shape (W, N). ``push`` overwrites the oldest row once full.
    """

    def __init__(self, n_stores: int, window: int) -> None:
        self.N = int(n_stores)
        self.W = int(window)
        self.price = np.zeros((self.W, self.N), dtype=np.float64)
        self.effort = np.zeros((self.W, self.N), dtype=np.float64)
        self.demand = np.zeros((self.W, self.N), dtype=np.float64)
        self.profit = np.zeros((self.W, self.N), dtype=np.float64)
        self._pos = 0
        self._filled = 0

    def push(self, prices, efforts, demands, profits) -> None:
        r = self._pos
        self.price[r] = prices
        self.effort[r] = efforts
        self.demand[r] = demands
        self.profit[r] = profits
        self._pos = (r + 1) % self.W
        self._filled = min(self._filled + 1, self.W)

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the filled window rows (chronological order not required)."""
        k = self._filled
        if k == self.W:
            return {"price": self.price, "effort": self.effort,
                    "demand": self.demand, "profit": self.profit}
        return {"price": self.price[:k], "effort": self.effort[:k],
                "demand": self.demand[:k], "profit": self.profit[:k]}


def division_context(active_divisions: list[str], division_params: dict | None) -> list[dict]:
    """Build the active-division descriptors for the CEO system prompt."""
    if not active_divisions:
        return []
    from hotelling.envelope.groups import _instantiate_divisions

    out = []
    for d in _instantiate_divisions(active_divisions, division_params):
        out.append({
            "name": d.name,
            "description": d.description(),
            "category_a": d.categories[0],
            "category_b": d.categories[1],
        })
    return out


def build_consumer_zones(grid_gdf, firms, n_side: int = 3) -> dict:
    """Static zone summary: an ``n_side`` x ``n_side`` spatial grid over the extent.

    Each zone: population (sum Einwohner), high-status share (pop-weighted mean of
    the social column * 100), dominant rival chain type (most stores in zone).
    Returns the ``consumers`` dict for the CEO state. Computed once per session.
    """
    social_col = next(
        (c for c in ("pi_H_res", "esix_normalized", "si_normalized")
         if grid_gdf is not None and c in grid_gdf.columns),
        None,
    )
    # Cell coordinates (centroids) + population + social share
    if grid_gdf is not None and "Einwohner" in grid_gdf.columns:
        cent = grid_gdf.geometry.centroid
        cx = cent.x.to_numpy()
        cy = cent.y.to_numpy()
        pop = grid_gdf["Einwohner"].fillna(0.0).to_numpy(dtype=np.float64)
        soc = (grid_gdf[social_col].fillna(0.5).to_numpy(dtype=np.float64)
               if social_col else np.full(len(grid_gdf), 0.5))
    else:
        cx = cy = pop = soc = np.array([])

    sx = np.array([f.location[0] for f in firms], dtype=np.float64)
    sy = np.array([f.location[1] for f in firms], dtype=np.float64)
    sct = np.array([ct_code(f.chain_type) for f in firms], dtype=object)

    xs = np.concatenate([cx, sx]) if cx.size else sx
    ys = np.concatenate([cy, sy]) if cy.size else sy
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    ex = (xmax - xmin) / n_side or 1.0
    ey = (ymax - ymin) / n_side or 1.0

    def zidx(x, y):
        ix = min(int((x - xmin) // ex), n_side - 1)
        iy = min(int((y - ymin) // ey), n_side - 1)
        return iy * n_side + ix

    compass = {0: "SW", 1: "S", 2: "SE", 3: "W", 4: "Central",
               5: "E", 6: "NW", 7: "N", 8: "NE"}
    zones = []
    total_pop = float(pop.sum()) if pop.size else 0.0
    total_hs = 0.0
    for z in range(n_side * n_side):
        if pop.size:
            cmask = np.array([zidx(cx[i], cy[i]) == z for i in range(len(cx))])
            zpop = float(pop[cmask].sum())
            zhs = float((pop[cmask] * soc[cmask]).sum() / zpop * 100.0) if zpop > 0 else 0.0
        else:
            zpop, zhs = 0.0, 0.0
        smask = np.array([zidx(sx[i], sy[i]) == z for i in range(len(sx))])
        if smask.any():
            vals, counts = np.unique(sct[smask], return_counts=True)
            dom = str(vals[int(np.argmax(counts))])
        else:
            dom = "-"
        total_hs += zhs * zpop
        zones.append({
            "label": f"{compass.get(z, str(z))}",
            "population": zpop,
            "high_status_pct": zhs,
            "dominant_rival_type": dom,
            "notes": "",
        })
    high_status_share_pct = (total_hs / total_pop) if total_pop > 0 else 0.0
    return {
        "total_population": total_pop,
        "high_status_share_pct": float(high_status_share_pct),
        "zones": [z for z in zones if z["population"] > 0 or z["dominant_rival_type"] != "-"],
    }


def build_ceo_state(
    window: RollingWindow,
    *,
    chain_id: str,
    store_chain: list[str],
    store_chain_type: list[str],
    store_group_labels: list[str],
    group_keys: list[str],
    zones: dict,
    history: list[dict],
    epoch: int,
    T_ceo: int,
    marginal_cost: float,
    min_delta_p: float,
    min_delta_e: float,
    store_metadata: list | None = None,
    enrich_groups: bool = False,
    with_effort: bool = True,
    with_comm: bool = False,
    signals_last_epoch: dict | None = None,
    own_last_signal: dict | None = None,
    strategic_analytics: dict | None = None,
) -> dict:
    """Assemble the per-epoch CEO state dict for state_ceo.jinja."""
    a = window.arrays()
    price, effort, demand, profit = a["price"], a["effort"], a["demand"], a["profit"]
    chain = np.array(store_chain, dtype=object)
    labels = np.array(store_group_labels, dtype=object)
    own_mask = chain == chain_id

    def _mean(arr, m):
        sub = arr[:, m]
        return float(sub.mean()) if sub.size else 0.0

    def _sum(arr, m):
        sub = arr[:, m]
        return float(sub.sum()) if sub.size else 0.0

    own_profit = _sum(profit, own_mask)
    own_demand = _sum(demand, own_mask)
    prev_profit = history[-1]["profit_realized"] if history else None
    trend = (100.0 * (own_profit - prev_profit) / abs(prev_profit)
             if prev_profit not in (None, 0.0) else 0.0)

    group_perf = []
    for key in group_keys:
        gm = own_mask & (labels == key)
        if not gm.any():
            continue
        g_dem = _sum(demand, gm)
        g_prof = _sum(profit, gm)
        group_perf.append({
            "group_key": key,
            "mean_price": _mean(price, gm),
            "mean_effort": _mean(effort, gm),
            "demand_share_pct": (100.0 * g_dem / own_demand) if own_demand > 0 else 0.0,
            "profit_share_pct": (100.0 * g_prof / own_profit) if own_profit != 0 else 0.0,
        })

    if enrich_groups and store_metadata is not None:
        from hotelling.llm.group_analytics import compute_group_briefs
        briefs = compute_group_briefs(
            a, chain_id=chain_id, store_chain=store_chain,
            store_chain_type=store_chain_type, store_group_labels=store_group_labels,
            store_metadata=store_metadata, group_keys=group_keys,
            marginal_cost=marginal_cost,
        )
        for gp in group_perf:
            b = briefs.get(gp["group_key"])
            if b:
                gp.update(b)

    # Public rival info: per OTHER brand, mean published price + within-window trend.
    rivals = []
    for brand in sorted({c for c in store_chain if c != chain_id}):
        rm = chain == brand
        rprice = price[:, rm]
        last_price = float(rprice.mean()) if rprice.size else 0.0
        if rprice.shape[0] >= 2:
            half = rprice.shape[0] // 2
            first = float(rprice[:half].mean())
            second = float(rprice[half:].mean())
            rtrend = 100.0 * (second - first) / first if first else 0.0
        else:
            rtrend = 0.0
        # chain_type code of this rival brand (all its stores share it)
        rtype = ct_code(np.array(store_chain_type, dtype=object)[rm][0]) if rm.any() else "S"
        rivals.append({
            "id": brand, "type": rtype, "n_stores": int(rm.sum()),
            "last_published_price": last_price, "price_trend_pct": rtrend,
            "last_signal": (signals_last_epoch or {}).get(brand),
        })

    n_own = int(own_mask.sum())
    W_win = int(price.shape[0]) if price.size else 0
    own_mean_price = _mean(price, own_mask)
    margin_per_unit = own_mean_price - float(marginal_cost)
    profit_per_store_per_period = (
        own_profit / (n_own * W_win) if (n_own > 0 and W_win > 0) else 0.0
    )

    return {
        "epoch": int(epoch),
        "T_ceo": int(T_ceo),
        "own": {
            "n_stores": int(own_mask.sum()),
            "mean_price_last_T": _mean(price, own_mask),
            "mean_effort_last_T": _mean(effort, own_mask),
            "total_demand_last_T": own_demand,
            "total_profit_last_T": own_profit,
            "profit_trend_pct": float(trend),
            "margin_per_unit": float(margin_per_unit),
            "profit_per_store_per_period": float(profit_per_store_per_period),
            "group_performance": group_perf,
            **({"headroom": strategic_analytics["headroom"]}
               if strategic_analytics and "headroom" in strategic_analytics else {}),
            **({"tier_counterfactual": strategic_analytics["tier_counterfactual"]}
               if strategic_analytics and "tier_counterfactual" in strategic_analytics else {}),
        },
        "history": history[-3:],
        "rivals": rivals,
        "consumers": zones,
        "marginal_cost": float(marginal_cost),
        "min_delta_p": float(min_delta_p),
        "min_delta_e": float(min_delta_e),
        "with_effort": bool(with_effort),
        "with_comm": bool(with_comm),
        "own_last_signal": own_last_signal,
    }
