"""Store-marker animations for the run-report pipeline (#4 price, #5 profit).

Each store is a chain-type-shaped marker (▼/●/■) coloured by its current price
(or gross per-store profit) on a *per-chain-type* sequential colormap
(D = winter, S = autumn, B = summer) with a **fixed** scale across all
timestamps, on an OSM-Mapnik basemap with **no** demand grid drawn.  Marker
sizes are static.  Three colorbars (one per chain type) and the marker-shape
legend sit outside the map so they never occlude data.

Colour-scale interval per chain type τ (price):
    action_grid:  [max(0, p^N_τ − ξ·span_τ), p^M_τ + ξ·span_τ],  span_τ = p^M_τ − p^N_τ
        mirroring the chain-specific Q-learning action grid — and computed the
        same way whether or not the run actually used --chs-grid.
    learned:      observed [min, max] of learned prices for that chain type.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import animation_io as _aio
from .style import CHAIN_TYPES, CHAIN_MARKERS, CHAIN_LABELS, latex_or_plain

logger = logging.getLogger(__name__)


def _price_scale(bundle, cfg, ct: str, observed: np.ndarray) -> Tuple[float, float]:
    sm = cfg.store_markers
    if sm.price_scale_mode == "learned":
        return float(np.nanmin(observed)), float(np.nanmax(observed))
    xi = sm.price_scale_xi if sm.price_scale_xi is not None else bundle.price_grid_xi
    m = bundle.type_masks[ct]
    if sm.price_scale_anchor == "store_minmax":
        pn = float(bundle.p_nash[m].min()); pm = float(bundle.p_mono[m].max())
    else:
        pn = float(bundle.p_nash[m].mean()); pm = float(bundle.p_mono[m].mean())
    span = max(pm - pn, 1e-6)
    return max(0.0, pn - xi * span), pm + xi * span


def _profit_scale(bundle, cfg, ct: str, observed: np.ndarray) -> Tuple[float, float]:
    sm = cfg.store_markers
    m = bundle.type_masks[ct]
    if sm.profit_scale_mode == "learned":
        lo, hi = float(np.nanmin(observed)), float(np.nanmax(observed))
    else:
        lo = float(bundle.profit_nash[m].min()); hi = float(bundle.profit_mono[m].max())
    span = max(hi - lo, 1e-6)
    pad = sm.profit_scale_margin * span
    return lo - pad, hi + pad


def animate_store_metric(bundle, cfg, out_dir: Path, metric: str) -> Path:
    """Render the price (#4) or profit (#5) store-marker animation."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    assert metric in ("price", "profit")
    frame_rows = bundle.frame_rows
    F = len(frame_rows)
    N = bundle.N

    # ── precompute (F, N) value matrix in one pass ──────────────────────────
    vals = np.empty((F, N), dtype=np.float64)
    for fi, t in enumerate(frame_rows):
        if metric == "price":
            vals[fi] = bundle.prices_at(int(t))
        else:
            vals[fi] = bundle.demands_profits_at(int(t))[1]

    # ── per-chain colour scales (fixed across all frames) ───────────────────
    norms: Dict[str, Normalize] = {}
    cmaps: Dict[str, str] = {}
    for ct in CHAIN_TYPES:
        m = bundle.type_masks[ct]
        if not m.any():
            continue
        obs = vals[:, m]
        lo, hi = (_price_scale if metric == "price" else _profit_scale)(bundle, cfg, ct, obs)
        norms[ct] = Normalize(vmin=lo, vmax=hi)
        cmaps[ct] = cfg.chain_cmaps.for_type(ct)

    sx = bundle.stores_gdf_3857.geometry.x.values
    sy = bundle.stores_gdf_3857.geometry.y.values
    extent = tuple(bundle.grid_gdf_3857.total_bounds)

    # ── static scene ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11.0, 10.0))
    ax = fig.add_axes([0.02, 0.05, 0.72, 0.90])
    _aio.add_osm_basemap(ax, extent, cfg.basemap.provider,
                         alpha=cfg.store_markers.basemap_alpha, zoom=cfg.basemap.zoom)

    scatters: Dict[str, object] = {}
    present = [ct for ct in CHAIN_TYPES if bundle.type_masks[ct].any()]
    for ct in present:
        m = bundle.type_masks[ct]
        sc = ax.scatter(sx[m], sy[m], c=vals[0, m], cmap=plt.get_cmap(cmaps[ct]),
                        norm=norms[ct], s=cfg.store_markers.marker_size,
                        marker=CHAIN_MARKERS[ct], edgecolors=cfg.store_markers.edge_colour,
                        linewidths=cfg.store_markers.edge_width, zorder=5)
        scatters[ct] = sc

    # per-chain colorbars stacked on the right (outside the map)
    cbar_label = ("price (EUR)" if metric == "price" else "profit (EUR)")
    cbar_slots = {"discount": 0.69, "standard": 0.40, "bio": 0.11}
    for ct in present:
        cax = fig.add_axes([0.80, cbar_slots[ct], 0.022, 0.24])
        sm_ = ScalarMappable(norm=norms[ct], cmap=plt.get_cmap(cmaps[ct])); sm_.set_array([])
        cb = fig.colorbar(sm_, cax=cax)
        cb.set_label(latex_or_plain(rf"{ct.capitalize()} {cbar_label}", f"{ct} {cbar_label}"), fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # marker-shape legend (outside, top-right above the colorbars)
    shape_handles = [Line2D([0], [0], marker=CHAIN_MARKERS[ct], color="none",
                            markerfacecolor="0.6", markeredgecolor="black",
                            markersize=9, label=CHAIN_LABELS[ct]) for ct in present]
    fig.legend(handles=shape_handles, loc="upper left", bbox_to_anchor=(0.77, 0.99),
               fontsize=cfg.legend.fontsize, title="Chain type",
               title_fontsize=cfg.legend.title_fontsize, framealpha=0.85)

    title = ax.set_title("", fontsize=11)

    # ── frame loop ──────────────────────────────────────────────────────────
    frames: List[np.ndarray] = []
    for fi, t in enumerate(frame_rows):
        for ct in present:
            scatters[ct].set_array(vals[fi, bundle.type_masks[ct]])
        step = int(bundle.recorded_steps[int(t)])
        title.set_text(latex_or_plain(rf"Store {metric}s --- step $t={step}$", f"Store {metric}s — step t={step}"))
        frames.append(_aio.fig_to_rgba(fig, cfg.animation_dpi(), cfg.global_.transparent))
    plt.close(fig)

    name = "04_store_price_animation" if metric == "price" else "05_store_profit_animation"
    return _aio.save_animation(
        frames, Path(out_dir) / name, cfg.animation.fps, cfg.animation_format(),
        cfg.global_.transparent, cfg.animation.loop, cfg.animation.lossless_webp,
        mov_codec=cfg.animation.mov_codec,
        mov_bits_per_mb=cfg.animation.mov_bits_per_mb,
        ffmpeg_path=cfg.animation.ffmpeg_path,
    )


def animate_store_price(bundle, cfg, out_dir) -> Path:
    return animate_store_metric(bundle, cfg, out_dir, "price")


def animate_store_profit(bundle, cfg, out_dir) -> Path:
    return animate_store_metric(bundle, cfg, out_dir, "profit")
