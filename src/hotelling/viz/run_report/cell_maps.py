"""Per-cell choropleth animations for the run-report pipeline.

    6  cell_price_delta   local Calvano price Δ per cell (D / S / B / all), cmap [0,1]
    7  cell_profit_delta  local Calvano profit-density Δ per cell (D / S / B / all)
    8  local_hhi          neighbourhood HHI per cell over time

The demand grid is drawn as a translucent (alpha=0.5) choropleth over an OSM
Mapnik basemap, with no grid lines.  All four delta maps, both metrics, and the
HHI map are produced in a SINGLE pass over the frames (the per-cell inside-mass
— the expensive term — is evaluated once per frame and reused).  Cells with
insufficient local demand are masked (fully transparent → basemap shows
through).  The colourbar sits outside the map.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import animation_io as _aio
from . import metrics as _m
from .style import latex_or_plain

logger = logging.getLogger(__name__)

# user-facing variant token -> internal type-mask key
_VKEY = {"discount": "discount", "standard": "standard", "bio": "bio", "all": "global"}
_VLABEL = {"discount": "Discount", "standard": "Standard", "bio": "Bio", "all": "All chains"}


def _grid_verts(grid_gdf) -> List[np.ndarray]:
    verts = []
    for g in grid_gdf.geometry:
        if g.geom_type == "Polygon":
            verts.append(np.asarray(g.exterior.coords, dtype=np.float64)[:, :2])
        elif g.geom_type == "MultiPolygon":
            largest = max(g.geoms, key=lambda gg: gg.area)
            verts.append(np.asarray(largest.exterior.coords, dtype=np.float64)[:, :2])
        else:
            verts.append(np.zeros((3, 2)))
    return verts


def _masked_cmap(name: str):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad((0, 0, 0, 0))   # NaN -> transparent
    return cmap


def _inside_outside(bundle, prices, efforts) -> Tuple[np.ndarray, np.ndarray]:
    from hotelling.core.market import catchment_cell_mass, cell_choice_mass
    if bundle.city.catch_indptr is not None:
        return catchment_cell_mass(bundle.city, prices, efforts, bundle.transport_cost)
    return cell_choice_mass(
        prices=prices, efforts=efforts, dist2_km2=bundle.city.dist2_km2,
        cell_pop=bundle.city.cell_pop, lambda_phi=bundle.city.lambda_phi,
        pi_H=bundle.city.pi_H, pi_H_lambda_phi=bundle.city.pi_H_lambda_phi,
        alpha=bundle.city.alpha, quality=bundle._qual, beta=bundle.city.beta,
        transport_cost=bundle.transport_cost, mu=bundle.city.mu, a0=bundle.city.a0,
        transport_exponent=getattr(bundle.city, "transport_exponent", 1.0),
    )


def compute_cell_frames(bundle, cfg, want_price: bool, want_profit: bool, want_hhi: bool) -> Dict:
    """Single pass over frames → per-frame (M,) arrays for every requested map."""
    variants = [v for v in cfg.cells.variants if v in _VKEY]
    bench = _m.CellDeltaBench(bundle, [_VKEY[v] for v in variants], cfg.cells.min_cell_demand) \
        if (want_price or want_profit) else None

    adj = group_ids = n_groups = None
    if want_hhi:
        adj = _m.build_cell_adjacency(bundle.cell_centroids_3035, cfg.hhi.radius_m)
        labels = bundle.chains if cfg.hhi.group_by == "chain" else bundle.chain_types
        group_ids, _, n_groups = _m.group_ids_from_labels(labels)

    F = len(bundle.frame_rows)
    M = bundle.M
    price_frames = {v: np.full((F, M), np.nan, np.float32) for v in variants} if want_price else {}
    profit_frames = {v: np.full((F, M), np.nan, np.float32) for v in variants} if want_profit else {}
    hhi_frames = np.full((F, M), np.nan, np.float32) if want_hhi else None

    need_outside = want_hhi and cfg.hhi.include_outside
    for fi, t in enumerate(bundle.frame_rows):
        prices_t = bundle.prices_at(int(t))
        efforts_t = bundle.efforts_at(int(t))
        if need_outside:
            inside_t, outside_t = _inside_outside(bundle, prices_t, efforts_t)
        else:
            inside_t = bundle.inside_mass_static(prices_t, efforts_t)
            outside_t = None
        for v in variants:
            key = _VKEY[v]
            if want_price:
                price_frames[v][fi] = bench.local_price_delta(
                    key, inside_t, prices_t, cfg.cells.delta_vmin, cfg.cells.delta_vmax)
            if want_profit:
                profit_frames[v][fi] = bench.local_profit_delta(
                    key, inside_t, prices_t, cfg.cells.delta_vmin, cfg.cells.delta_vmax)
        if want_hhi:
            hhi_frames[fi] = _m.local_hhi(inside_t, adj, group_ids, n_groups,
                                          outside_t, cfg.hhi.include_outside, cfg.hhi.normalised)

    return {"price": price_frames, "profit": profit_frames, "hhi": hhi_frames}


def _render_choropleth(bundle, cfg, out_dir, name, frame_vals, cmap_name,
                       vmin, vmax, cbar_label, title_prefix) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    verts = _grid_verts(bundle.grid_gdf_3857)
    extent = tuple(bundle.grid_gdf_3857.total_bounds)
    cmap = _masked_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(11.0, 10.0))
    ax = fig.add_axes([0.02, 0.05, 0.74, 0.90])
    _aio.add_osm_basemap(ax, extent, cfg.basemap.provider, alpha=1.0, zoom=cfg.basemap.zoom)

    poly = PolyCollection(verts, array=np.ma.masked_invalid(frame_vals[0]),
                          cmap=cmap, norm=norm, linewidths=0, edgecolors="none",
                          alpha=cfg.cells.alpha, zorder=3)
    ax.add_collection(poly)

    cax = fig.add_axes([0.80, 0.18, 0.024, 0.64])
    sm_ = ScalarMappable(norm=norm, cmap=cmap); sm_.set_array([])
    fig.colorbar(sm_, cax=cax, label=cbar_label)

    title = ax.set_title("", fontsize=11)
    frames: List[np.ndarray] = []
    for fi, t in enumerate(bundle.frame_rows):
        poly.set_array(np.ma.masked_invalid(frame_vals[fi]))
        step = int(bundle.recorded_steps[int(t)])
        title.set_text(latex_or_plain(rf"{title_prefix} --- step $t={step}$", f"{title_prefix} — step t={step}"))
        frames.append(_aio.fig_to_rgba(fig, cfg.animation_dpi(), cfg.global_.transparent))
    plt.close(fig)
    return _aio.save_animation(
        frames, Path(out_dir) / name, cfg.animation.fps,
        cfg.animation_format(), cfg.global_.transparent,
        cfg.animation.loop, cfg.animation.lossless_webp,
        mov_codec=cfg.animation.mov_codec,
        mov_bits_per_mb=cfg.animation.mov_bits_per_mb,
        ffmpeg_path=cfg.animation.ffmpeg_path,
    )


def render_cell_animations(bundle, cfg, out_dir, want_price, want_profit, want_hhi) -> List[Path]:
    """Compute the shared per-frame fields once, then render every enabled map."""
    data = compute_cell_frames(bundle, cfg, want_price, want_profit, want_hhi)
    paths: List[Path] = []

    for v in [vv for vv in cfg.cells.variants if vv in _VKEY]:
        if want_price:
            paths.append(_render_choropleth(
                bundle, cfg, out_dir, f"06_cell_price_delta_{v}", data["price"][v],
                cfg.cells.delta_cmap, cfg.cells.delta_vmin, cfg.cells.delta_vmax,
                latex_or_plain(r"Local price $\Delta$", "Local price Delta"),
                latex_or_plain(rf"Local price $\Delta$ --- {_VLABEL[v]}", f"Local price Delta — {_VLABEL[v]}")))
        if want_profit:
            paths.append(_render_choropleth(
                bundle, cfg, out_dir, f"07_cell_profit_delta_{v}", data["profit"][v],
                cfg.cells.delta_cmap, cfg.cells.delta_vmin, cfg.cells.delta_vmax,
                latex_or_plain(r"Local profit $\Delta$", "Local profit Delta"),
                latex_or_plain(rf"Local profit $\Delta$ --- {_VLABEL[v]}", f"Local profit Delta — {_VLABEL[v]}")))

    if want_hhi:
        vmin = cfg.hhi.vmin if cfg.hhi.vmin is not None else (0.0)
        vmax = cfg.hhi.vmax if cfg.hhi.vmax is not None else (1.0 if cfg.hhi.normalised else 10000.0)
        paths.append(_render_choropleth(
            bundle, cfg, out_dir, "08_local_hhi", data["hhi"], cfg.hhi.cmap, vmin, vmax,
            latex_or_plain(rf"Local HHI ({cfg.hhi.group_by})", f"Local HHI ({cfg.hhi.group_by})"),
            latex_or_plain(rf"Local market HHI (r={cfg.hhi.radius_m:.0f} m)", f"Local HHI (r={cfg.hhi.radius_m:.0f} m)")))

    return paths
