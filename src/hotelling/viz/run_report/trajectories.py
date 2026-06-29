"""Trajectory & time-series figures for the run-report pipeline.

Produces (numbers refer to the pipeline spec):
    1  price_trajectory          mean price (global + per chain type) + MA + p^N/p^M levels
    2  profit_trajectory         mean per-store profit (global + per chain) + MA + pi^N/pi^M
    3  delta_trajectory          price Δ (dashed) and profit Δ (solid), global + per chain
  extras:
       delta_over_time           two-panel price-Δ / profit-Δ
       market_shares             chain-type inside share over time
       welfare                   total consumer-surplus logsum over time (dense path only)
       global_hhi                market-wide HHI over time
       price_dispersion          cross-store price std over time
       final_price_distribution  per-store price histogram vs Nash/mono per chain type

All text is LaTeX; legends sit outside the axes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import metrics as _m
from .style import CHAIN_TYPES, latex_or_plain

logger = logging.getLogger(__name__)

_VARIANTS = ("global", "discount", "standard", "bio")


# ── shared helpers ───────────────────────────────────────────────────────────

def _save(fig, out_dir: Path, name: str, cfg) -> Path:
    out = Path(out_dir) / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg.global_.dpi, bbox_inches="tight",
                transparent=cfg.global_.transparent, pad_inches=0.05)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def _legend_outside(ax, cfg, handles=None, labels=None, title=None):
    kw = cfg.legend.kwargs()
    if title is not None:
        kw["title"] = title
    if handles is not None:
        ax.legend(handles=handles, labels=labels, **kw)
    else:
        ax.legend(**kw)


def _colour(bundle_cfg, v: str) -> str:
    return bundle_cfg.colours.for_type(v)



# ── 1 / 2 : price & profit trajectories ──────────────────────────────────────

def _trajectory(bundle, cfg, out_dir, kind: str, name: str) -> Path:
    import matplotlib.pyplot as plt
    steps, series = _m.chain_mean_series(bundle, kind)
    nash, mono = _m.chain_benchmark_levels(bundle, kind)
    w = _m.steps_to_points(cfg.trajectory.ma_window_steps, bundle.analysis_step_spacing)

    fig, ax = plt.subplots(figsize=tuple(cfg.trajectory.figsize))
    handles: List = []
    for v in _VARIANTS:
        c = _colour(cfg, v)
        y = series[v]
        if cfg.trajectory.raw_alpha > 0 and w > 1:
            ax.plot(steps, y, color=c, lw=0.8, alpha=cfg.trajectory.raw_alpha, zorder=1)
        #ma = _m.moving_average(y, w, cfg.trajectory.ma_kind)
        ma = pd.Series(y).rolling(window=w).mean()
        lbl = {"global": r"Global", "discount": r"Discount $(D)$",
               "standard": r"Standard $(S)$", "bio": r"Bio $(B)$"}[v]
        line, = ax.plot(steps, ma, color=c, lw=2.0 if v == "global" else 1.6,
                        label=lbl, zorder=3)
        handles.append(line)
        if cfg.trajectory.show_benchmarks and v != "global":
            ax.axhline(nash[v], color=c, ls="--", lw=0.9, alpha=0.5, zorder=2)
            ax.axhline(mono[v], color=c, ls=":", lw=0.9, alpha=0.5, zorder=2)
    if cfg.trajectory.show_benchmarks:
        ax.axhline(nash["global"], color="black", ls="--", lw=1.1, alpha=0.6, zorder=2)
        ax.axhline(mono["global"], color="black", ls=":", lw=1.1, alpha=0.6, zorder=2)

    bench_handles = [
        Line2D([0], [0], color="black", ls="--", lw=1.1, label=latex_or_plain(r"Nash $p^N$" if kind == "price" else r"Nash $\pi^N$", "Nash")),
        Line2D([0], [0], color="black", ls=":", lw=1.1, label=latex_or_plain(r"Monopoly $p^M$" if kind == "price" else r"Monopoly $\pi^M$", "Monopoly")),
    ]
    ax.set_xlabel(r"Simulation step $t$")
    if kind == "price":
        ax.set_ylabel(r"Mean price (EUR)")
        ax.set_title(latex_or_plain(r"Price trajectories ($1000$-MA) by chain type", "Price trajectories"))
    else:
        ax.set_ylabel(r"Mean profit per store (EUR)")
        ax.set_title(latex_or_plain(r"Per-store profit trajectories ($1000$-MA)", "Profit trajectories"))
    _legend_outside(ax, cfg, handles=handles + bench_handles,
                    labels=[h.get_label() for h in handles + bench_handles])
    fig.tight_layout()
    return _save(fig, out_dir, name, cfg)


def plot_price_trajectory(bundle, cfg, out_dir) -> Path:
    return _trajectory(bundle, cfg, out_dir, "price", "01_price_trajectory")


def plot_profit_trajectory(bundle, cfg, out_dir) -> Path:
    return _trajectory(bundle, cfg, out_dir, "profit", "02_profit_trajectory")


# ── 3 : combined price/profit delta trajectory ───────────────────────────────

def plot_delta_trajectory(bundle, cfg, out_dir) -> Path:
    import matplotlib.pyplot as plt
    clip = tuple(cfg.trajectory.delta_clip)
    w = _m.steps_to_points(cfg.trajectory.ma_window_steps, bundle.analysis_step_spacing)

    steps_p, sp = _m.chain_mean_series(bundle, "price")
    steps_pi, spi = _m.chain_mean_series(bundle, "profit")
    #sp_ma = {v: _m.moving_average(sp[v], w, cfg.trajectory.ma_kind) for v in _VARIANTS}
    sp_ma = {v: pd.Series(sp[v]).rolling(window=w).mean() for v in _VARIANTS}
    #spi_ma = {v: _m.moving_average(spi[v], w, cfg.trajectory.ma_kind) for v in _VARIANTS}
    spi_ma = {v: pd.Series(spi[v]).rolling(window=w).mean() for v in _VARIANTS}
    dp = _m.delta_series_from_means(sp_ma, bundle, "price", clip)
    dpi = _m.delta_series_from_means(spi_ma, bundle, "profit", clip)

    fig, ax = plt.subplots(figsize=tuple(cfg.trajectory.figsize))
    for v in _VARIANTS:
        c = _colour(cfg, v)
        ax.plot(steps_p, dp[v], color=c, ls="--", lw=1.5, alpha=0.9)
        ax.plot(steps_pi, dpi[v], color=c, ls="-", lw=1.8)
    ax.axhline(0.0, color="grey", lw=0.8, alpha=0.7)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel(r"Simulation step $t$")
    ax.set_ylabel(latex_or_plain(r"Calvano $\Delta$", "Calvano Delta"))
    ax.set_title(latex_or_plain(r"Collusion index $\Delta(t)$: price (dashed) vs.\ profit (solid)",
                                "Collusion index Delta(t)"))
    colour_handles = [Line2D([0], [0], color=_colour(cfg, v), lw=2,
                             label={"global": "Global", "discount": "Discount", "standard": "Standard", "bio": "Bio"}[v])
                      for v in _VARIANTS]
    style_handles = [Line2D([0], [0], color="black", ls="--", lw=1.5, label=latex_or_plain(r"$\Delta$ price", "Delta price")),
                     Line2D([0], [0], color="black", ls="-", lw=1.8, label=latex_or_plain(r"$\Delta$ profit", "Delta profit"))]
    _legend_outside(ax, cfg, handles=colour_handles + style_handles,
                    labels=[h.get_label() for h in colour_handles + style_handles])
    fig.tight_layout()
    return _save(fig, out_dir, "03_delta_trajectory", cfg)


# ── extras ───────────────────────────────────────────────────────────────────

def plot_delta_over_time(bundle, cfg, out_dir) -> Path:
    import matplotlib.pyplot as plt
    clip = tuple(cfg.trajectory.delta_clip)
    w = _m.steps_to_points(cfg.trajectory.ma_window_steps, bundle.analysis_step_spacing)
    steps_p, sp = _m.chain_mean_series(bundle, "price")
    steps_pi, spi = _m.chain_mean_series(bundle, "profit")
    #dp = _m.delta_series_from_means({v: _m.moving_average(sp[v], w, cfg.trajectory.ma_kind) for v in _VARIANTS}, bundle, "price", clip)
    dp = _m.delta_series_from_means({v: pd.Series(sp[v]).rolling(window=w).mean() for v in _VARIANTS}, bundle, "price", clip)
    #dpi = _m.delta_series_from_means({v: _m.moving_average(spi[v], w, cfg.trajectory.ma_kind) for v in _VARIANTS}, bundle, "profit", clip)
    dpi = _m.delta_series_from_means({v: pd.Series(spi[v]).rolling(window=w).mean() for v in _VARIANTS}, bundle, "profit", clip)

    fig, axes = plt.subplots(1, 2, figsize=(cfg.trajectory.figsize[0] * 1.25, cfg.trajectory.figsize[1]))
    for ax, (d, steps, ttl) in zip(axes, [(dp, steps_p, latex_or_plain(r"Price $\Delta$", "Price Delta")),
                                          (dpi, steps_pi, latex_or_plain(r"Profit $\Delta$", "Profit Delta"))]):
        for v in _VARIANTS:
            ax.plot(steps, d[v], color=_colour(cfg, v), lw=1.6,
                    label={"global": "Global", "discount": "Discount", "standard": "Standard", "bio": "Bio"}[v])
        ax.axhline(0.0, color="grey", lw=0.8); ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel(r"Simulation step $t$"); ax.set_ylabel(latex_or_plain(r"$\Delta$", "Delta"))
        ax.set_title(ttl)
    _legend_outside(axes[1], cfg)
    fig.tight_layout()
    return _save(fig, out_dir, "03b_delta_panels", cfg)


def plot_market_shares(bundle, cfg, out_dir) -> Path:
    import matplotlib.pyplot as plt
    D = bundle.get_analysis_demands()
    steps = bundle.analysis_steps
    tot = D.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=tuple(cfg.trajectory.figsize))
    for ct in CHAIN_TYPES:
        m = bundle.type_masks[ct]
        share = D[:, m].sum(axis=1) / tot[:, 0]
        #ax.plot(steps, _m.moving_average(share, _m.steps_to_points(cfg.trajectory.ma_window_steps, bundle.analysis_step_spacing), cfg.trajectory.ma_kind),
        ax.plot(steps, pd.Series(share).rolling(window=_m.steps_to_points(cfg.trajectory.ma_window_steps, bundle.analysis_step_spacing)).mean(),
                color=_colour(cfg, ct), lw=1.6,
                label={"discount": "Discount", "standard": "Standard", "bio": "Bio"}[ct])
    ax.set_xlabel(r"Simulation step $t$"); ax.set_ylabel("Inside-market share")
    ax.set_title("Chain-type market shares over time")
    _legend_outside(ax, cfg)
    fig.tight_layout()
    return _save(fig, out_dir, "extra_market_shares", cfg)


def plot_global_hhi(bundle, cfg, out_dir) -> Path:
    import matplotlib.pyplot as plt
    gids, _, ng = _m.group_ids_from_labels(
        bundle.chains if cfg.hhi.group_by == "chain" else bundle.chain_types)
    D = bundle.get_analysis_demands()
    steps = bundle.analysis_steps
    hhi = np.array([_m.global_hhi(D[i], gids, ng, cfg.hhi.normalised) for i in range(len(steps))])
    fig, ax = plt.subplots(figsize=tuple(cfg.trajectory.figsize))
    ax.plot(steps, hhi, color="black", lw=1.6, label="Market HHI")
    ax.set_xlabel(r"Simulation step $t$")
    ax.set_ylabel("HHI" + (" (normalised)" if cfg.hhi.normalised else ""))
    ax.set_title(f"Market-wide HHI over time (by {cfg.hhi.group_by})")
    _legend_outside(ax, cfg)
    fig.tight_layout()
    return _save(fig, out_dir, "extra_global_hhi", cfg)


def plot_price_dispersion(bundle, cfg, out_dir) -> Path:
    import matplotlib.pyplot as plt
    P = bundle.analysis_prices
    steps = bundle.analysis_steps
    fig, ax = plt.subplots(figsize=tuple(cfg.trajectory.figsize))
    w = _m.steps_to_points(cfg.trajectory.ma_window_steps, bundle.analysis_step_spacing)
    #ax.plot(steps, _m.moving_average(P.std(axis=1), w, cfg.trajectory.ma_kind),
    ax.plot(steps, pd.Series(P.std(axis=1)).rolling(window=w).mean(),
            color="black", lw=1.6, label="Global")
    for ct in CHAIN_TYPES:
        m = bundle.type_masks[ct]
        #ax.plot(steps, _m.moving_average(P[:, m].std(axis=1), w, cfg.trajectory.ma_kind),
        ax.plot(steps, pd.Series(P[:, m].std(axis=1)).rolling(window=w).mean(),
                color=_colour(cfg, ct), lw=1.3,
                label={"discount": "Discount", "standard": "Standard", "bio": "Bio"}[ct])
    ax.set_xlabel(r"Simulation step $t$"); ax.set_ylabel("Cross-store price std.\\ (EUR)")
    ax.set_title("Price dispersion over time")
    _legend_outside(ax, cfg)
    fig.tight_layout()
    return _save(fig, out_dir, "extra_price_dispersion", cfg)


def plot_welfare(bundle, cfg, out_dir):
    import matplotlib.pyplot as plt
    if bundle.city.dist2_km2 is None:
        logger.info("Welfare trajectory skipped: dense distance matrix unavailable.")
        return None
    rows = bundle.frame_rows
    steps = bundle.recorded_steps[rows]
    cs = np.array([_m.total_consumer_surplus(bundle, bundle.prices_at(int(t)), bundle.efforts_at(int(t)))
                   for t in rows])
    fig, ax = plt.subplots(figsize=tuple(cfg.trajectory.figsize))
    ax.plot(steps, cs, color="black", lw=1.6, marker=".", label="Total CS proxy")
    ax.set_xlabel(r"Simulation step $t$"); ax.set_ylabel("Consumer surplus (EUR, logsum)")
    ax.set_title("Consumer-surplus welfare proxy over time")
    _legend_outside(ax, cfg)
    fig.tight_layout()
    return _save(fig, out_dir, "extra_welfare", cfg)


def plot_final_price_distribution(bundle, cfg, out_dir) -> Path:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(cfg.trajectory.figsize[0] * 1.3, cfg.trajectory.figsize[1]),
                             sharey=True)
    for ax, ct in zip(axes, CHAIN_TYPES):
        m = bundle.type_masks[ct]
        c = _colour(cfg, ct)
        ax.hist(bundle.learned_prices[m], bins=20, color=c, alpha=0.55, edgecolor="black", lw=0.4)
        ax.axvline(float(bundle.p_nash[m].mean()), color="black", ls="--", lw=1.2,
                   label=latex_or_plain(r"$p^N$", "Nash"))
        ax.axvline(float(bundle.p_mono[m].mean()), color="black", ls=":", lw=1.2,
                   label=latex_or_plain(r"$p^M$", "Mono"))
        ax.set_title({"discount": "Discount", "standard": "Standard", "bio": "Bio"}[ct])
        ax.set_xlabel("Final price (EUR)")
    axes[0].set_ylabel("Number of stores")
    _legend_outside(axes[2], cfg)
    fig.suptitle("Converged per-store price distribution vs.\\ benchmarks")
    fig.tight_layout()
    return _save(fig, out_dir, "extra_final_price_distribution", cfg)
