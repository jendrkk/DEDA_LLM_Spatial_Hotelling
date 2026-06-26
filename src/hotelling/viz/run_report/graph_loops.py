"""Graph-state loop delta visualisation for the run-report pipeline (#9).

Only meaningful for ``state_mode == "graph_states"`` runs, where each store
observes a small set of reciprocal rivals selected by max-weight b-matching.
The undirected rival graph decomposes into connected components; its *cycles*
(closed loops) are the structures where mutual observation can sustain tacit
coordination.  This module extracts those cycles and renders, on an OSM
basemap, each closed loop coloured by the mean Calvano Δ of its member stores
(price and/or profit), with the loop edges drawn and the Δ annotated.

Cycle detection uses :func:`networkx.cycle_basis` when available; absent
networkx it falls back to a DFS cycle search that is exact for the degree-≤2
b-matching graphs (the common ``--graph-states K 2`` case) and best-effort
otherwise.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import animation_io as _aio
from .style import latex_or_plain

logger = logging.getLogger(__name__)


# ── graph construction ───────────────────────────────────────────────────────

def _edges_from_rivals(graph_rivals: np.ndarray) -> List[Tuple[int, int]]:
    edges = set()
    N, k = graph_rivals.shape
    for i in range(N):
        for c in range(k):
            r = int(graph_rivals[i, c])
            if r >= 0:
                edges.add((min(i, r), max(i, r)))
    return sorted(edges)


def _find_cycles(edges: List[Tuple[int, int]], n_nodes: int, min_size: int) -> List[List[int]]:
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges)
        cycles = [c for c in nx.cycle_basis(G) if len(c) >= min_size]
        return cycles
    except Exception as exc:  # noqa: BLE001
        logger.info("networkx unavailable (%s); using DFS cycle fallback.", exc)
        return _dfs_cycles(edges, n_nodes, min_size)


def _dfs_cycles(edges, n_nodes, min_size) -> List[List[int]]:
    """Exact for components that are simple paths/cycles (degree ≤ 2)."""
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    seen = set()
    cycles: List[List[int]] = []
    for start in range(n_nodes):
        if start in seen or start not in adj:
            continue
        comp, stack = [], [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u)
            for w in adj[u]:
                if w not in seen:
                    stack.append(w)
        n_edges = sum(len(adj[u]) for u in comp) // 2
        if n_edges >= len(comp) >= min_size and all(len(adj[u]) == 2 for u in comp):
            cycles.append(_order_cycle(comp, adj))
    return cycles


def _order_cycle(nodes, adj) -> List[int]:
    nodes = list(nodes)
    start = nodes[0]
    order, prev, cur = [start], None, start
    while True:
        nxts = [w for w in adj[cur] if w != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        order.append(nxt); prev, cur = cur, nxt
        if len(order) > len(nodes):
            break
    return order


# ── per-store deltas ─────────────────────────────────────────────────────────

def _store_delta(bundle, kind: str) -> np.ndarray:
    if kind == "price":
        num = bundle.learned_prices - bundle.p_nash
        den = bundle.p_mono - bundle.p_nash
    else:
        num = bundle.learned_profits - bundle.profit_nash
        den = bundle.profit_mono - bundle.profit_nash
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(np.abs(den) > 1e-9, num / den, np.nan)
    return np.clip(d, 0.0, 1.0)


def _store_delta_series(bundle, kind: str, rows: np.ndarray) -> np.ndarray:
    """(len(rows), N) per-store Δ over time (for the per-loop time series)."""
    P = bundle.decode_prices_rows(rows)                       # (R, N)
    if kind == "price":
        den = bundle.p_mono - bundle.p_nash
        num = P - bundle.p_nash[None, :]
    else:
        prof = np.empty_like(P)
        for i, t in enumerate(rows):
            prof[i] = bundle.demands_profits_at(int(t))[1]
        den = bundle.profit_mono - bundle.profit_nash
        num = prof - bundle.profit_nash[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(np.abs(den)[None, :] > 1e-9, num / den[None, :], np.nan)
    return np.clip(d, 0.0, 1.0)


# ── rendering ────────────────────────────────────────────────────────────────

def _render_loop_map(bundle, cfg, out_dir, edges, cycles, kind: str) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import Normalize

    sx = bundle.stores_gdf_3857.geometry.x.values
    sy = bundle.stores_gdf_3857.geometry.y.values
    extent = tuple(bundle.grid_gdf_3857.total_bounds)
    delta = _store_delta(bundle, kind)
    norm = Normalize(0.0, 1.0)
    cmap = plt.get_cmap(cfg.graph_loops.loop_cmap)

    fig = plt.figure(figsize=(11.0, 10.0))
    ax = fig.add_axes([0.02, 0.05, 0.74, 0.90])
    _aio.add_osm_basemap(ax, extent, cfg.basemap.provider, alpha=1.0, zoom=cfg.basemap.zoom)

    # all matched edges (faint context)
    seg = [[(sx[a], sy[a]), (sx[b], sy[b])] for a, b in edges]
    ax.add_collection(LineCollection(seg, colors="0.5", linewidths=0.4, alpha=0.35, zorder=3))

    # cycles: filled translucent polygon + bold edge, coloured by mean Δ
    loop_deltas = []
    polys, polys_c, loop_segments = [], [], []
    for cyc in cycles:
        dmean = float(np.nanmean(delta[cyc]))
        loop_deltas.append(dmean)
        coords = np.column_stack([sx[cyc], sy[cyc]])
        polys.append(coords)
        polys_c.append(cmap(norm(dmean)) if np.isfinite(dmean) else (0, 0, 0, 0))
        ring = list(cyc) + [cyc[0]]
        loop_segments.extend([[(sx[ring[i]], sy[ring[i]]), (sx[ring[i + 1]], sy[ring[i + 1]])]
                              for i in range(len(ring) - 1)])
    if polys:
        ax.add_collection(PolyCollection(polys, facecolors=polys_c, edgecolors="none",
                                         alpha=0.45, zorder=4))
        edge_c = [polys_c[i] for i, cyc in enumerate(cycles) for _ in range(len(cyc))]
        ax.add_collection(LineCollection(loop_segments, colors=edge_c, linewidths=1.8, zorder=5))

    ax.scatter(sx, sy, s=9, c="0.2", zorder=6, linewidths=0)

    if cfg.graph_loops.annotate:
        for cyc, dmean in zip(cycles, loop_deltas):
            if not np.isfinite(dmean):
                continue
            cx, cy = float(sx[cyc].mean()), float(sy[cyc].mean())
            ax.annotate(f"{dmean:.2f}", (cx, cy), fontsize=7, ha="center", va="center",
                        color="black", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    cax = fig.add_axes([0.80, 0.18, 0.024, 0.64])
    sm_ = ScalarMappable(norm=norm, cmap=cmap); sm_.set_array([])
    fig.colorbar(sm_, cax=cax,
                 label=latex_or_plain(rf"Loop mean $\Delta$ ({kind})", f"Loop mean Delta ({kind})"))
    ax.set_title(latex_or_plain(
        rf"Closed rival loops --- {kind} $\Delta$ ({len(cycles)} loops)",
        f"Closed rival loops — {kind} Delta ({len(cycles)} loops)"), fontsize=11)

    out = Path(out_dir) / f"09_graph_loop_delta_{kind}.png"
    fig.savefig(out, dpi=cfg.global_.dpi, bbox_inches="tight",
                transparent=cfg.global_.transparent, pad_inches=0.05)
    plt.close(fig)
    return out


def _render_loop_timeseries(bundle, cfg, out_dir, cycles, kind: str) -> Optional[Path]:
    import matplotlib.pyplot as plt
    if not cycles:
        return None
    rows = bundle.frame_rows
    steps = bundle.recorded_steps[rows]
    dts = _store_delta_series(bundle, kind, rows)             # (R, N)
    n = len(cycles)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow), squeeze=False)
    cmap = plt.get_cmap(cfg.graph_loops.loop_cmap)
    for idx, cyc in enumerate(cycles):
        ax = axes[idx // ncol][idx % ncol]
        series = np.nanmean(dts[:, cyc], axis=1)
        ax.plot(steps, series, color=cmap(0.8), lw=1.4)
        ax.axhline(0, color="grey", lw=0.6); ax.axhline(1, color="grey", lw=0.6, ls="--")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(latex_or_plain(rf"Loop {idx} ($|C|={len(cyc)}$)", f"Loop {idx} (|C|={len(cyc)})"),
                     fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(latex_or_plain(rf"Per-loop {kind} $\Delta(t)$", f"Per-loop {kind} Delta(t)"))
    fig.supxlabel(r"Simulation step $t$", fontsize=9)
    fig.tight_layout()
    out = Path(out_dir) / f"09b_graph_loop_timeseries_{kind}.png"
    fig.savefig(out, dpi=cfg.global_.dpi, bbox_inches="tight",
                transparent=cfg.global_.transparent, pad_inches=0.05)
    plt.close(fig)
    return out


def render_graph_loops(bundle, cfg, out_dir) -> List[Path]:
    """Entry point: emit loop maps (and optional time series) for #9."""
    if bundle.graph_rivals is None:
        logger.info("graph_loop_deltas skipped: no graph_rivals.npy (not a graph_states run).")
        return []
    edges = _edges_from_rivals(bundle.graph_rivals)
    cycles = _find_cycles(edges, bundle.N, cfg.graph_loops.min_loop_size)
    if not cycles:
        logger.info("graph_loop_deltas: no closed loops of size >= %d found.",
                    cfg.graph_loops.min_loop_size)
    kinds = {"price": ["price"], "profit": ["profit"], "both": ["price", "profit"]}[cfg.graph_loops.which_delta]
    paths: List[Path] = []
    for kind in kinds:
        paths.append(_render_loop_map(bundle, cfg, out_dir, edges, cycles, kind))
        if cfg.graph_loops.per_loop_timeseries:
            ts = _render_loop_timeseries(bundle, cfg, out_dir, cycles, kind)
            if ts is not None:
                paths.append(ts)
    return paths
