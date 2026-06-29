"""Metric helpers for the run-report visualisation pipeline (self-contained).

These are intentionally local to the pipeline so it runs without depending on
the (partially-stubbed) ``hotelling.analysis.metrics`` module.  All Calvano Δ
definitions follow the same convention as ``runner.py``:

    price  Δ_tau = (p_bar_tau - p^N_tau) / (p^M_tau - p^N_tau)
    profit Δ_tau = (Σ π_tau - Σ π^N_tau) / (Σ π^M_tau - Σ π^N_tau)

with the gross variable profit π_j = (p_j - c_j) · D_j (effort cost excluded,
exact in the price-only Phase-0 regime).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CHAIN_TYPES = ("discount", "standard", "bio")
_VARIANTS = ("global", "discount", "standard", "bio")


# ── moving average ───────────────────────────────────────────────────────────

def moving_average(y: np.ndarray, window: int, kind: str = "trailing") -> np.ndarray:
    """Moving average of a 1-D series; returns an array of the same length.

    ``window`` <= 1 returns *y* unchanged.  NaNs are ignored within each window.
    """
    y = np.asarray(y, dtype=np.float64)
    if window <= 1 or y.size == 0:
        return y.copy()
    window = int(min(window, y.size))
    import pandas as pd
    s = pd.Series(y)
    if kind == "centered":
        out = s.rolling(window, min_periods=1, center=True).mean()
    else:
        out = s.rolling(window, min_periods=1).mean()
    return out.to_numpy()


def steps_to_points(window_steps: int, step_spacing: int) -> int:
    """Convert a moving-average window expressed in sim steps to recorded points."""
    if window_steps <= 0:
        return 1
    return max(1, int(round(window_steps / max(step_spacing, 1))))


# ── aggregate chain trajectories ─────────────────────────────────────────────

def chain_mean_series(
    bundle, kind: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Per-analysis-row means of price / profit / demand, global + per chain type.

    Reads the precomputed, cached analysis matrices on *bundle*
    (``analysis_prices`` and the lazily-reconstructed
    ``get_analysis_profits()`` / ``get_analysis_demands()``), so the expensive
    spatial market-clearing reconstruction runs at most once per run regardless
    of how many plots call this.  Returns ``(analysis_steps, {variant: (A,)})``.
    """
    if kind == "price":
        vals = bundle.analysis_prices
    elif kind == "profit":
        vals = bundle.get_analysis_profits()
    elif kind == "demand":
        vals = bundle.get_analysis_demands()
    else:
        raise ValueError(kind)

    steps = bundle.analysis_steps
    out: Dict[str, np.ndarray] = {}
    for v in _VARIANTS:
        m = bundle.type_masks[v]
        out[v] = vals[:, m].mean(axis=1) if m.any() else np.full(len(steps), np.nan)
    return steps, out


def chain_benchmark_levels(bundle, kind: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-variant Nash and monopoly reference levels (means over the variant)."""
    nash, mono = {}, {}
    for v in _VARIANTS:
        m = bundle.type_masks[v]
        if not m.any():
            nash[v] = mono[v] = np.nan
            continue
        if kind == "price":
            nash[v] = float(bundle.p_nash[m].mean())
            mono[v] = float(bundle.p_mono[m].mean())
        else:
            nash[v] = float(bundle.profit_nash[m].mean())
            mono[v] = float(bundle.profit_mono[m].mean())
    return nash, mono


def delta_series_from_means(
    series: Dict[str, np.ndarray], bundle, kind: str, clip: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    """Calvano Δ(t) per variant from mean trajectories.

    Price Δ uses chain means directly.  Profit Δ uses the *sum* over the
    variant's stores (mean·n is monotone in the mean, so a mean-based Δ with
    equal denominators is identical up to the n factor that cancels)."""
    out: Dict[str, np.ndarray] = {}
    lo, hi = clip
    for v in _VARIANTS:
        m = bundle.type_masks[v]
        if not m.any():
            out[v] = np.full_like(series[v], np.nan)
            continue
        if kind == "price":
            pn = float(bundle.p_nash[m].mean())
            pm = float(bundle.p_mono[m].mean())
        else:
            pn = float(bundle.profit_nash[m].mean())
            pm = float(bundle.profit_mono[m].mean())
        denom = pm - pn
        if abs(denom) < 1e-9:
            out[v] = np.full_like(series[v], np.nan)
        else:
            out[v] = np.clip((series[v] - pn) / denom, lo, hi)
    return out


# ── per-cell local deltas ────────────────────────────────────────────────────

class CellDeltaBench:
    """Static per-cell, per-variant benchmark expected prices & profit densities.

    Built once from the Nash / monopoly inside-mass matrices; per frame only the
    learned inside-mass and learned prices are needed to evaluate the local Δ.
    """

    def __init__(self, bundle, variants: List[str], min_cell_demand: float):
        self.bundle = bundle
        self.variants = variants
        self.min_demand = float(min_cell_demand)
        b = bundle
        inside_N = b.inside_mass_static(b.p_nash, b.e_bench)     # (M, N)
        inside_M = b.inside_mass_static(b.p_mono, b.e_bench)
        self.pbar_N: Dict[str, np.ndarray] = {}
        self.pbar_M: Dict[str, np.ndarray] = {}
        self.pi_N: Dict[str, np.ndarray] = {}
        self.pi_M: Dict[str, np.ndarray] = {}
        self.denom_p: Dict[str, np.ndarray] = {}
        self.denom_pi: Dict[str, np.ndarray] = {}
        self.valid_p: Dict[str, np.ndarray] = {}
        self.valid_pi: Dict[str, np.ndarray] = {}
        costs = b.marginal_costs
        for v in variants:
            mask = b.type_masks[v]
            pN, piN, servedN = self._cell_stats(inside_N, b.p_nash, costs, mask)
            pM, piM, servedM = self._cell_stats(inside_M, b.p_mono, costs, mask)
            self.pbar_N[v], self.pbar_M[v] = pN, pM
            self.pi_N[v], self.pi_M[v] = piN, piM
            self.denom_p[v] = pM - pN
            self.denom_pi[v] = piM - piN
            served_ok = (servedN > self.min_demand) & (servedM > self.min_demand)
            self.valid_p[v] = served_ok & (np.abs(self.denom_p[v]) > 1e-9)
            self.valid_pi[v] = served_ok & (np.abs(self.denom_pi[v]) > 1e-9)

    @staticmethod
    def _cell_stats(inside, prices, costs, mask):
        sub = inside[:, mask]                       # (M, n)
        served = sub.sum(axis=1)                    # (M,)
        with np.errstate(invalid="ignore", divide="ignore"):
            pbar = (sub @ prices[mask]) / served
        pi = sub @ (prices[mask] - costs[mask])
        return pbar, pi, served

    def local_price_delta(self, v: str, inside_t: np.ndarray, prices_t: np.ndarray,
                          vmin: float, vmax: float) -> np.ndarray:
        mask = self.bundle.type_masks[v]
        sub = inside_t[:, mask]
        served = sub.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            pbar = (sub @ prices_t[mask]) / served
            d = (pbar - self.pbar_N[v]) / self.denom_p[v]
        ok = self.valid_p[v] & (served > self.min_demand)
        out = np.where(ok, np.clip(d, vmin, vmax), np.nan)
        return out

    def local_profit_delta(self, v: str, inside_t: np.ndarray, prices_t: np.ndarray,
                           vmin: float, vmax: float) -> np.ndarray:
        mask = self.bundle.type_masks[v]
        costs = self.bundle.marginal_costs
        sub = inside_t[:, mask]
        served = sub.sum(axis=1)
        pi = sub @ (prices_t[mask] - costs[mask])
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (pi - self.pi_N[v]) / self.denom_pi[v]
        ok = self.valid_pi[v] & (served > self.min_demand)
        out = np.where(ok, np.clip(d, vmin, vmax), np.nan)
        return out


# ── local HHI ────────────────────────────────────────────────────────────────

def build_cell_adjacency(centroids: np.ndarray, radius_m: float):
    """Sparse (M, M) CSR boolean adjacency: cells within ``radius_m`` (incl. self)."""
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=radius_m, output_type="ndarray")  # (P, 2) i<j
    M = centroids.shape[0]
    rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(M)])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(M)])
    data = np.ones(rows.shape[0], dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(M, M))


def group_inside(inside: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """Aggregate (M, N) store inside-mass into (M, n_groups) by group id per store."""
    M = inside.shape[0]
    out = np.zeros((M, n_groups), dtype=np.float64)
    for g in range(n_groups):
        cols = group_ids == g
        if cols.any():
            out[:, g] = inside[:, cols].sum(axis=1)
    return out


def local_hhi(inside_t: np.ndarray, adj, group_ids: np.ndarray, n_groups: int,
              outside_t: Optional[np.ndarray], include_outside: bool,
              normalised: bool) -> np.ndarray:
    """Per-cell local HHI over the neighbourhood defined by ``adj``."""
    g_inside = group_inside(inside_t, group_ids, n_groups)        # (M, G)
    nb = adj @ g_inside                                           # (M, G)
    total = nb.sum(axis=1)
    if include_outside and outside_t is not None:
        nb_out = adj @ outside_t.reshape(-1, 1)
        total = total + nb_out[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = nb / total[:, None]
        hhi = np.nansum(shares ** 2, axis=1)
        if include_outside and outside_t is not None:
            hhi = hhi + (nb_out[:, 0] / total) ** 2
    hhi = np.where(total > 0, hhi, np.nan)
    return hhi if normalised else hhi * 10000.0


def global_hhi(demands: np.ndarray, group_ids: np.ndarray, n_groups: int,
               normalised: bool = True) -> float:
    """Market-wide HHI from store demands aggregated by group."""
    tot = demands.sum()
    if tot <= 0:
        return np.nan
    g = np.array([demands[group_ids == k].sum() for k in range(n_groups)])
    shares = g / tot
    hhi = float(np.sum(shares ** 2))
    return hhi if normalised else hhi * 10000.0


# ── welfare ──────────────────────────────────────────────────────────────────

def total_consumer_surplus(bundle, prices: np.ndarray, efforts: np.ndarray) -> float:
    """Total logit inclusive-value welfare proxy (EUR). Dense path only.

    Returns NaN when the dense distance matrix is unavailable (pure sparse run).
    """
    if bundle.city.dist2_km2 is None:
        return float("nan")
    from hotelling.core.market import cell_metrics
    cs = cell_metrics(prices, efforts, bundle.city, bundle.transport_cost,
                      metric="consumer_surplus")              # per-consumer, per cell
    total_w = bundle.city.cell_pop + bundle.city.lambda_phi
    return float(np.nansum(cs * total_w))


# ── group-id helper ──────────────────────────────────────────────────────────

def group_ids_from_labels(labels: np.ndarray) -> Tuple[np.ndarray, List[str], int]:
    """Map an (N,) array of string labels to contiguous integer ids."""
    uniq = list(dict.fromkeys(labels.tolist()))
    lut = {name: i for i, name in enumerate(uniq)}
    ids = np.array([lut[x] for x in labels], dtype=np.int64)
    return ids, uniq, len(uniq)
