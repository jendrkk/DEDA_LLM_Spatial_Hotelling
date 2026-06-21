#!/usr/bin/env python3
"""
Data-section maps for presentation.

Produces three transparent PNG files at DPI=200 in report/figures/maps/:
  total_demand.png      — effective consumer mass choropleth
  supermarkets.png      — store locations by chain/type
  consumer_types.png    — pi_H_res (share H-type consumers) choropleth

Run from repo root:
    conda activate py314
    python report/plot_maps.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data" / "processed"
FIG_DIR   = REPO_ROOT / "report" / "figures" / "maps"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── matplotlib / LaTeX config ───────────────────────────────────────────────
try:
    matplotlib.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    _USETEX = True
except Exception:
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif"})
    _USETEX = False

matplotlib.rcParams.update({
    "axes.labelsize":  10,
    "axes.titlesize":  11,
    "legend.fontsize":  8,
    "legend.title_fontsize": 9,
    "figure.facecolor": "none",
    "axes.facecolor":   "none",
    "savefig.facecolor": "none",
})

DPI           = 200
FIGSIZE_43    = (8, 6)
BASEMAP_ALPHA = 1
LAMBDA_VAL    = 429

# ── contextily ─────────────────────────────────────────────────────────────
try:
    import contextily as ctx
    _HAS_CTX = True
    _BASEMAP_SRC = ctx.providers.CartoDB.Positron
except ImportError:
    _HAS_CTX = False
    warnings.warn("contextily not installed — maps will have no OSM basemap.")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_gdf(path: Path) -> gpd.GeoDataFrame:
    """Read parquet as GeoDataFrame. gpd.read_parquet decodes WKB bytes."""
    gdf = gpd.read_parquet(path)
    return gdf


def load_all_data() -> dict:
    """Load and CRS-normalise all five spatial datasets."""
    demand   = _load_gdf(DATA_DIR / "demand_grid.parquet")
    stores   = _load_gdf(DATA_DIR / "supermarkets.parquet")
    clusters = _load_gdf(DATA_DIR / "prime_location_clusters.parquet")
    malls    = _load_gdf(DATA_DIR / "grid_malls.parquet")
    stations = _load_gdf(DATA_DIR / "grid_with_stations.parquet")

    for gdf in (demand, stores, clusters, malls, stations):
        if gdf.crs is None:
            gdf.set_crs(epsg=3035, inplace=True)
        elif gdf.crs.to_epsg() != 3035:
            gdf.to_crs(epsg=3035, inplace=True)

    return dict(demand=demand, stores=stores, clusters=clusters,
                malls=malls, stations=stations)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def compute_phi_i(demand: gpd.GeoDataFrame) -> pd.Series:
    sc   = demand["station_class_normalized"].fillna(0.0)
    clus = demand["has_cluster"].fillna(False).astype(float)
    mall = demand["has_mall"].fillna(False).astype(float)
    return (0.4 * sc + 0.3 * clus + 0.3 * mall).rename("phi_i")


def compute_pi_H_res(demand: gpd.GeoDataFrame) -> pd.Series:
    esix = demand.get("esix_normalized")
    si   = demand.get("si_normalized")
    pi   = pd.Series(0.5, index=demand.index, dtype=float)
    if esix is not None and si is not None:
        both   = esix.notna() & si.notna()
        only_e = esix.notna() & si.isna()
        only_s = si.notna()   & esix.isna()
        pi[both]   = (esix[both] + si[both]) / 2.0
        pi[only_e] = esix[only_e].values
        pi[only_s] = si[only_s].values
    elif esix is not None:
        pi = esix.fillna(0.5).astype(float).rename("pi_H_res")
    elif si is not None:
        pi = si.fillna(0.5).astype(float).rename("pi_H_res")
    return pi.clip(0.0, 1.0).rename("pi_H_res")


def _station_centroids(demand: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mask = demand["station_class"].notna()
    sub  = demand.loc[mask, ["geometry", "station_class"]].copy()
    sub["geometry"] = sub.geometry.centroid
    return sub


def _cluster_circles(clusters: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Approximate each cluster as a circle: r = sqrt(area/pi)."""
    rows = []
    for _, row in clusters.iterrows():
        area = float(row["area"]) if pd.notna(row.get("area")) else 0.0
        r    = np.sqrt(area / np.pi) if area > 0 else 500.0
        pt   = row["geometry"]  # cluster centroid in EPSG:3035
        rows.append({"cluster_id": row["cluster_id"],
                     "geometry": pt.buffer(r)})
    return gpd.GeoDataFrame(rows, crs=clusters.crs)


def _mall_centroids(malls: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mask = malls["has_mall"].fillna(False)
    sub  = malls.loc[mask, ["geometry"]].copy()
    sub["geometry"] = sub.geometry.centroid
    return sub


def _to3857(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(epsg=3857)


def _add_basemap(ax: plt.Axes, alpha: float = BASEMAP_ALPHA) -> None:
    if not _HAS_CTX:
        return
    try:
        ctx.add_basemap(ax, source=_BASEMAP_SRC, zoom="auto",
                        reset_extent=False, alpha=alpha)
    except Exception as exc:
        warnings.warn(f"Basemap fetch failed: {exc}")


def _fix_extent(ax: plt.Axes, bounds: tuple) -> None:
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])


# ─────────────────────────────────────────────────────────────────────────────
# Map 1 — Total demand choropleth
# ─────────────────────────────────────────────────────────────────────────────

def plot_total_demand(data: dict) -> None:
    demand   = data["demand"]
    malls    = data["malls"]
    stations = data["stations"]
    clusters = data["clusters"]

    phi          = compute_phi_i(demand)
    total_demand = demand["Einwohner"].astype(float) + LAMBDA_VAL * phi

    gdf_w  = _to3857(demand.assign(total_demand=total_demand))
    sta_w  = _to3857(_station_centroids(stations))
    circ_w = _to3857(_cluster_circles(clusters))
    mall_w = _to3857(_mall_centroids(malls))
    bounds = gdf_w.total_bounds

    fig, ax = plt.subplots(figsize=FIGSIZE_43)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    gdf_w.plot(column="total_demand", ax=ax, cmap="YlOrRd",
               legend=True,
               legend_kwds={"label": r"$\omega_i + \lambda\phi_i$",
                             "orientation": "vertical", "shrink": 0.60,
                             "pad": 0.02, "aspect": 28},
               alpha=0.35, linewidth=0,
               missing_kwds={"color": "none"})

    _fix_extent(ax, bounds)
    _add_basemap(ax)
    _fix_extent(ax, bounds)

    # Station dots — Blues discrete colormap (class 1=best → darkest)
    sc_vals = sorted(stations["station_class"].dropna().unique())
    _scmap  = plt.cm.get_cmap("Blues", len(sc_vals) + 3)
    sc_col  = {v: _scmap(0.95 - 0.55 * i / max(len(sc_vals) - 1, 1))
               for i, v in enumerate(sorted(sc_vals, reverse=True))}
    for v in sorted(sc_vals):
        mask = sta_w["station_class"] == v
        if mask.sum() == 0:
            continue
        sta_w[mask].plot(ax=ax, color=sc_col[v], markersize=5,
                         marker="o", zorder=6,
                         label=f"Station class {int(v)}")

    # Cluster boundaries
    if not circ_w.empty:
        circ_w.boundary.plot(ax=ax, color="#1a3a5c", linewidth=2.5,
                             linestyle="--", zorder=7,
                             label="Prime-location cluster")

    # Mall dots — black squares
    if not mall_w.empty:
        mall_w.plot(ax=ax, color="black", marker="s", markersize=3,
                    zorder=8, label="Shopping mall", alpha=0.35)

    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85,
              facecolor="white", edgecolor="none", bbox_to_anchor=(1, 0.5),
              bbox_transform=ax.transAxes)
    ax.set_title(r"Effective demand mass $\omega_i + \lambda\phi_i$",
                 pad=8, fontsize=11)

    out = FIG_DIR / "total_demand.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", transparent=True,
                pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Map 2 — Supermarket locations
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_MARKER = {"standard": "o", "discount": "^", "bio": "s"}
_TYPE_CMAP   = {"standard": ("Blues",  0.38, 0.90),
                "discount": ("Reds",   0.33, 0.88),
                "bio":      ("Greens", 0.36, 0.90)}
_TYPE_LABEL  = {"standard": r"Standard ($\circ$)",
                "discount": r"Discount ($\triangle$)",
                "bio":      r"Bio ($\blacksquare$)"}


def _build_chain_colours(stores: gpd.GeoDataFrame) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    for ctype, (cname, lo, hi) in _TYPE_CMAP.items():
        chains = sorted(stores.loc[stores["chain_type"] == ctype,
                                   "chain"].dropna().unique())
        cmap = plt.cm.get_cmap(cname)
        for i, ch in enumerate(chains):
            out[ch] = cmap(lo + (hi - lo) * i / max(len(chains) - 1, 1))
    return out


def plot_supermarkets(data: dict) -> None:
    demand = data["demand"]
    stores = data["stores"]

    bg_w  = _to3857(demand[["geometry"]].copy())
    st_w  = _to3857(stores.copy())
    bounds = bg_w.total_bounds
    chain_col = _build_chain_colours(st_w)

    fig, ax = plt.subplots(figsize=FIGSIZE_43)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    bg_w.plot(ax=ax, color="none", edgecolor="none", linewidth=0)
    _fix_extent(ax, bounds)
    _add_basemap(ax, alpha=0.9)
    _fix_extent(ax, bounds)

    legend_handles: list = []
    for ctype in ["standard", "discount", "bio"]:
        tmask  = st_w["chain_type"] == ctype
        marker = _TYPE_MARKER[ctype]
        for chain in sorted(st_w.loc[tmask, "chain"].dropna().unique()):
            cmask = tmask & (st_w["chain"] == chain)
            if cmask.sum() == 0:
                continue
            col = chain_col.get(chain, "grey")
            st_w[cmask].plot(ax=ax, color=col, marker=marker,
                             markersize=4.5, zorder=5,
                             linewidth=0.3, edgecolor="white")
            legend_handles.append(
                mpatches.Patch(facecolor=col, label=chain, linewidth=0))

    type_handles = [
        plt.Line2D([0],[0], marker=_TYPE_MARKER[t], color="w",
                   markerfacecolor="grey", markersize=8,
                   label=_TYPE_LABEL[t], linestyle="None")
        for t in ["standard", "discount", "bio"]
    ]
    ax.legend(handles=type_handles + legend_handles,
              loc="lower left", fontsize=6, framealpha=0.85,
              facecolor="white", edgecolor="none", ncol=2,
              columnspacing=0.5, bbox_to_anchor=(1, 0.5),
              bbox_transform=ax.transAxes)
    ax.set_axis_off()
    ax.set_title("Supermarket locations by chain and type", pad=8)

    out = FIG_DIR / "supermarkets.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", transparent=True,
                pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Map 3 — Share of H-type consumers
# ─────────────────────────────────────────────────────────────────────────────

def plot_consumer_types(data: dict) -> None:
    demand   = data["demand"]
    clusters = data["clusters"]
    stations = data["stations"]

    pi_H   = compute_pi_H_res(demand)
    gdf_w  = _to3857(demand.assign(pi_H_res=pi_H))
    sta_w  = _to3857(_station_centroids(stations))
    circ_w = _to3857(_cluster_circles(clusters))
    bounds = gdf_w.total_bounds

    fig, ax = plt.subplots(figsize=FIGSIZE_43)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    gdf_w.plot(column="pi_H_res", ax=ax, cmap="RdYlGn",
               vmin=0.0, vmax=1.0,
               legend=True,
               legend_kwds={
                   "label": r"$\pi_{H,i}$ (share of high-status consumers)",
                   "orientation": "vertical", "shrink": 0.60,
                   "pad": 0.02, "aspect": 28},
               alpha=0.75, linewidth=0,
               missing_kwds={"color": "lightgrey", "label": "No data"})

    _fix_extent(ax, bounds)
    _add_basemap(ax, alpha=0.85)
    _fix_extent(ax, bounds)

    if not sta_w.empty:
        sta_w.plot(ax=ax, color="#555555", markersize=3.5,
                   marker="o", zorder=6, label="Transit station")

    if not circ_w.empty:
        circ_w.boundary.plot(ax=ax, color="black", linewidth=2.0,
                             linestyle="--", zorder=7,
                             label="Prime-location cluster")

    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85,
              facecolor="white", edgecolor="none", bbox_to_anchor=(1, 0.5),
              bbox_transform=ax.transAxes)
    ax.set_title(
        r"Share of high-status consumers $\pi_{H,i}$ (LOR social index)",
        pad=8)

    out = FIG_DIR / "consumer_types.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", transparent=True,
                pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading spatial data …")
    data = load_all_data()
    print(f"  demand_grid : {len(data['demand'])} cells")
    print(f"  supermarkets: {len(data['stores'])} stores")
    print(f"  clusters    : {len(data['clusters'])} prime-location clusters")
    print()
    print("Rendering maps …")
    plot_total_demand(data)
    plot_supermarkets(data)
    plot_consumer_types(data)
    print(f"\nAll maps saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
