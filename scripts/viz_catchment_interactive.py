#!/usr/bin/env python3
"""
Interactive catchment-range visualiser — slider + static OSM basemap.

Architecture
------------
All data is loaded and the basemap is pre-rendered ONCE at startup.
Slider events only remove/replace two variable artists (the catchment
PolyCollection and the target-store star) plus the info text — the basemap
image and the stores scatter are never touched again after initial draw.

Layout
------
  Left  (63%): map axes with fixed extent  →  ax_map
  Right (37%): monospace info panel        →  ax_info
  Below map  : horizontal colorbar         →  ax_cbar
  Bottom     : integer store-ID slider     →  ax_slide

Usage
-----
    conda activate py314
    cd DEDA_LLM_Spatial_Hotelling
    python scripts/viz_catchment_interactive.py
    python scripts/viz_catchment_interactive.py --init-store 42
    python scripts/viz_catchment_interactive.py --catchment-minutes 10 --catchment-k-min 2 --catchment-k-max 20

Requires contextily for the OSM basemap:
    pip install contextily --break-system-packages
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
import yaml

try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    print(
        "WARNING: contextily not installed — no basemap.\n"
        "Install: pip install contextily --break-system-packages",
        flush=True,
    )

# ── Use the non-deprecated colormap API (matplotlib ≥ 3.7) ───────────────────
CMAP_CATCHMENT = matplotlib.colormaps["YlOrRd"]

CHAIN_TYPE_COLORS: dict[str, str] = {
    "discount": "royalblue",
    "standard": "firebrick",
    "bio":      "forestgreen",
}
CHAIN_TYPE_LABELS: dict[str, str] = {
    "discount": "Discount (D)",
    "standard": "Standard (S)",
    "bio":      "Bio (B)",
}

# Zoom level for the static OSM basemap.
# For the Berlin inner Ringbahn (~15 km × 10 km), zoom 13 gives ~4 892 m/tile:
# roughly 3×2 tiles → 768×512 px image — small and fast to pre-render.
BASEMAP_ZOOM = 13


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive catchment visualiser — slider + static OSM basemap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/env/berlin_inner_ring_calibrated.yaml",
    )
    p.add_argument("--catchment-minutes", type=float, default=None,
                   help="Override catchment_minutes from config")
    p.add_argument("--catchment-k-min",   type=int,   default=None)
    p.add_argument("--catchment-k-max",   type=int,   default=None)
    p.add_argument("--init-store",         type=int,   default=0,
                   help="Store ID shown on launch")
    p.add_argument("--zoom",               type=int,   default=BASEMAP_ZOOM,
                   help="OSM tile zoom level (12–15 recommended for inner ring)")
    return p.parse_args()


# ── Geometry: polygon → vertex array ────────────────────────────────────────

def _geom_verts(geom) -> np.ndarray:
    """Return exterior ring coordinates as (K, 2) float64."""
    if geom.geom_type == "Polygon":
        return np.asarray(geom.exterior.coords, dtype=np.float64)[:, :2]
    if geom.geom_type == "MultiPolygon":
        largest = max(geom.geoms, key=lambda g: g.area)
        return np.asarray(largest.exterior.coords, dtype=np.float64)[:, :2]
    return np.zeros((3, 2), dtype=np.float64)


# ── CSR inversion ─────────────────────────────────────────────────────────────

def invert_csr_for_store(
    indptr:  np.ndarray,
    indices: np.ndarray,
    tt_arr:  np.ndarray,
    target:  int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cell_row_indices, travel_times) for cells containing store *target*."""
    hit = np.where(indices.astype(np.int64) == target)[0]
    if hit.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.float64)
    # searchsorted on indptr: indptr[i] ≤ p < indptr[i+1] → row i
    rows = (np.searchsorted(indptr.astype(np.int64), hit, side="right") - 1).astype(np.int64)
    return rows, tt_arr[hit].astype(np.float64)


# ── Info-text builder ─────────────────────────────────────────────────────────

def build_info_text(
    sid: int, chain: str, chain_type: str, loc_x: float, loc_y: float,
    catchment_minutes: float, catchment_k_min: int, catchment_k_max: int,
    n_catch_pop: int, n_catch_total: int, M_pop: int, M: int,
    tt_pop: np.ndarray, competitors: dict[int, int],
    n_competitors: int, mean_stores_catch_cell: float, global_mean: float,
    stores_df,
) -> str:
    sep  = "=" * 52
    sep2 = "-" * 52
    lines = [
        sep,
        f"  Store ID    : {sid}",
        f"  Chain       : {chain}",
        f"  Chain type  : {chain_type}",
        f"  Location    : ({loc_x:.0f}, {loc_y:.0f})  [EPSG:3035]",
        "",
        "  Catchment parameters  (stationary — set at load time)",
        f"    radius    : {catchment_minutes:.0f} min  (transit, one-way)",
        f"    k_min     : {catchment_k_min}  (min stores / cell, any radius)",
        f"    k_max     : {catchment_k_max}  (hard cap / cell)",
        f"    distance  : transit time, GTFS+OSM — NOT Euclidean",
        f"    stationary: YES — fixed; only demand FLOW changes w/ prices",
        "",
        f"  Catchment of store {sid}  (inverted CSR view)",
        f"    Populated cells  : {n_catch_pop:,}  of  {M_pop:,} pop. cells",
        f"    Total cells      : {n_catch_total:,}  of  {M:,}",
    ]
    if tt_pop.size > 0:
        lines.append(
            f"    Transit time     : min={tt_pop.min():.1f}  "
            f"mean={tt_pop.mean():.1f}  max={tt_pop.max():.1f}  min"
        )
    lines += [
        "",
        "  Competitor landscape",
        f"    Stores sharing ≥1 pop. cell : {n_competitors}",
        f"    Mean stores / catch. cell   : {mean_stores_catch_cell:.1f}  (incl. store {sid})",
        f"    Global mean stores / cell   : {global_mean:.1f}",
        "",
        "  Top-10 competitors  (shared populated cells)",
        f"  {'ID':>4}  {'Type':9}  {'Chain':<25}  {'Cells':>5}",
        f"  {sep2}",
    ]
    top10 = sorted(competitors.items(), key=lambda x: -x[1])[:10]
    for jj, cnt in top10:
        c_chain = str(stores_df.iloc[jj].get("chain", "?"))
        c_type  = str(stores_df.iloc[jj].get("chain_type", "?"))
        lines.append(f"  {jj:4d}  {c_type:9s}  {c_chain:<25s}  {cnt:>5d}")
    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    catchment_minutes = (
        args.catchment_minutes if args.catchment_minutes is not None
        else float(cfg.get("catchment_minutes", 20.0))
    )
    catchment_k_min = (
        args.catchment_k_min if args.catchment_k_min is not None
        else int(cfg.get("catchment_k_min", 12))
    )
    catchment_k_max = (
        args.catchment_k_max if args.catchment_k_max is not None
        else int(cfg.get("catchment_k_max", 60))
    )
    zoom_level = args.zoom

    grid_path   = ROOT / cfg["grid_path"]
    stores_path = ROOT / cfg["stores_path"]
    tt_path     = ROOT / cfg["travel_times_path"]

    # ── Load stores ──────────────────────────────────────────────────────────
    print("Loading supermarkets...", flush=True)
    stores_gdf = gpd.read_parquet(stores_path).reset_index(drop=True)
    N = len(stores_gdf)
    store_ids = [str(i) for i in range(N)]

    # ── Load demand grid, filter to populated cells ───────────────────────────
    print("Loading demand grid (populated cells only)...", flush=True)
    grid_gdf = gpd.read_parquet(grid_path)
    grid_gdf = grid_gdf.sort_values("GITTER_ID_100m").reset_index(drop=True)
    if grid_gdf["GITTER_ID_100m"].duplicated().sum() > 0:
        grid_gdf = (
            grid_gdf.drop_duplicates("GITTER_ID_100m", keep="first")
            .reset_index(drop=True)
        )
    M = len(grid_gdf)
    cell_ids = grid_gdf["GITTER_ID_100m"].tolist()

    pop_col   = "Einwohner" if "Einwohner" in grid_gdf.columns else None
    pop_mask  = (grid_gdf[pop_col].fillna(0) > 0) if pop_col else pd.Series(True, index=grid_gdf.index)
    grid_pop  = grid_gdf[pop_mask].copy()
    M_pop     = len(grid_pop)

    # Fast lookup: original DataFrame row-index → positional index inside grid_pop
    orig_to_pop_pos = {orig_idx: pos for pos, orig_idx in enumerate(grid_pop.index)}
    print(f"  Grid: {M:,} total | {M_pop:,} populated | Stores: {N}", flush=True)

    # ── Reproject to EPSG:3857 for contextily ────────────────────────────────
    print("Reprojecting to EPSG:3857...", flush=True)
    grid_pop_3857  = grid_pop.to_crs(epsg=3857)
    stores_3857    = stores_gdf.to_crs(epsg=3857)

    store_xs         = np.asarray([g.x for g in stores_3857.geometry], dtype=np.float64)
    store_ys         = np.asarray([g.y for g in stores_3857.geometry], dtype=np.float64)
    store_chain_types = stores_gdf["chain_type"].fillna("standard").values

    # Pre-build polygon vertex arrays for ALL populated cells (used by PolyCollection)
    print("Pre-building polygon vertex cache...", flush=True)
    _pop_verts = [_geom_verts(g) for g in grid_pop_3857.geometry]

    # ── Fixed map extent: buffered bounding box of populated grid ─────────────
    bounds_3857 = grid_pop_3857.total_bounds          # (xmin, ymin, xmax, ymax)
    xmin_g, ymin_g, xmax_g, ymax_g = bounds_3857
    pad_x = (xmax_g - xmin_g) * 0.06
    pad_y = (ymax_g - ymin_g) * 0.06
    MAP_XLIM = (xmin_g - pad_x, xmax_g + pad_x)
    MAP_YLIM = (ymin_g - pad_y, ymax_g + pad_y)

    # ── Pre-render static basemap (one network call) ──────────────────────────
    basemap_img = basemap_ext = None
    if HAS_CTX:
        print(f"Pre-rendering OSM basemap at zoom={zoom_level}...", flush=True)
        try:
            # bounds2img(w, s, e, n, ...) → (img, (w_out, e_out, s_out, n_out))
            basemap_img, basemap_ext = ctx.bounds2img(
                MAP_XLIM[0], MAP_YLIM[0], MAP_XLIM[1], MAP_YLIM[1],
                zoom=zoom_level,
                source=ctx.providers.OpenStreetMap.Mapnik,
                ll=False,  # coordinates are already in EPSG:3857
            )
            print(
                f"  Basemap: {basemap_img.shape[1]}×{basemap_img.shape[0]} px "
                f"(w={basemap_ext[0]:.0f}, e={basemap_ext[1]:.0f}, "
                f"s={basemap_ext[2]:.0f}, n={basemap_ext[3]:.0f})",
                flush=True,
            )
        except Exception as e:
            print(f"  Basemap pre-render failed: {e} — continuing without basemap.",
                  flush=True)
            basemap_img = basemap_ext = None

    # ── Load travel times ─────────────────────────────────────────────────────
    print("Loading travel times...", flush=True)
    tt_raw = pd.read_parquet(tt_path)
    tt_raw["from_id"] = tt_raw["from_id"].astype(str)
    tt_raw["to_id"]   = tt_raw["to_id"].astype(str)
    print(f"  {len(tt_raw):,} travel-time rows", flush=True)

    # ── Build catchment CSR ───────────────────────────────────────────────────
    print(
        f"Building catchment CSR  "
        f"(radius={catchment_minutes:.0f} min | k_min={catchment_k_min} | k_max={catchment_k_max})...",
        flush=True,
    )
    from hotelling.spatial.loader import build_catchment

    indptr, indices, tt_min = build_catchment(
        tt_df=tt_raw,
        cell_ids=cell_ids,
        store_ids=store_ids,
        transport_cost=float(cfg.get("transport_cost", 0.01)),
        transport_exponent=1.0,
        catchment_minutes=float(catchment_minutes),
        k_min=catchment_k_min,
        k_max=catchment_k_max,
        nan_fill_minutes=float(cfg.get("nan_fill_minutes", 120.0)),
    )
    nnz              = int(indptr[-1])
    global_mean_per_cell = nnz / M if M > 0 else 0.0
    print(f"  CSR built: NNZ={nnz:,}, global mean={global_mean_per_cell:.1f} stores/cell",
          flush=True)

    # ── Figure layout — light theme ───────────────────────────────────────────
    #
    #   [0.02, 0.13, 0.59, 0.85]   ax_map   — map (left 61%)
    #   [0.03, 0.09, 0.57, 0.018]  ax_cbar  — horizontal colorbar (below map)
    #   [0.10, 0.03, 0.49, 0.040]  ax_slide — slider
    #   [0.63, 0.02, 0.35, 0.96]   ax_info  — monospace info panel (right 35%)
    #
    fig = plt.figure(figsize=(19, 11))
    fig.patch.set_facecolor("white")

    ax_map   = fig.add_axes([0.02, 0.13, 0.59, 0.85])
    ax_cbar  = fig.add_axes([0.03, 0.09, 0.57, 0.018])
    ax_slide = fig.add_axes([0.10, 0.03, 0.49, 0.040])
    ax_info  = fig.add_axes([0.63, 0.03, 0.34, 0.95])

    ax_map.set_facecolor("#f0f0f0")   # light grey fallback (under basemap)
    ax_info.set_facecolor("#f7f7f9")
    for spine in ax_info.spines.values():
        spine.set_edgecolor("#ccccdd")

    ax_info.set_xticks([]); ax_info.set_yticks([])

    # ── Draw permanent elements on ax_map ─────────────────────────────────────
    ax_map.set_xlim(*MAP_XLIM)
    ax_map.set_ylim(*MAP_YLIM)
    ax_map.set_aspect("equal")
    ax_map.tick_params(labelsize=7, colors="#555555")
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#aaaaaa")

    # 1. Basemap — pre-rendered, drawn once
    if basemap_img is not None:
        # basemap_ext = (west_out, east_out, south_out, north_out)
        # matplotlib extent = [left, right, bottom, top]
        w_out, e_out, s_out, n_out = basemap_ext
        ax_map.imshow(
            basemap_img,
            extent=[w_out, e_out, s_out, n_out],
            origin="upper",
            aspect="equal",
            zorder=0,
            interpolation="bilinear",
        )
        # Restore extent after imshow (imshow can modify axes limits)
        ax_map.set_xlim(*MAP_XLIM)
        ax_map.set_ylim(*MAP_YLIM)

    # 2. All stores — colour by chain type, drawn ONCE (permanent)
    for ct, color in CHAIN_TYPE_COLORS.items():
        mask = store_chain_types == ct
        if mask.any():
            ax_map.scatter(
                store_xs[mask], store_ys[mask],
                c=color, s=28, alpha=0.88, linewidths=0,
                zorder=5, label=CHAIN_TYPE_LABELS[ct],
            )

    # ── Colorbar — created ONCE with a dedicated cax ──────────────────────────
    # vmin=0, vmax=catchment_minutes: fixed range, never changes with slider.
    norm_cb = mcolors.Normalize(vmin=0, vmax=catchment_minutes)
    sm      = plt.cm.ScalarMappable(cmap=CMAP_CATCHMENT, norm=norm_cb)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    cbar.set_label(
        f"Transit time to selected store (min)  "
        f"[yellow = near, red = far]  |  "
        f"radius = {catchment_minutes:.0f} min",
        fontsize=8.5, color="#333333",
    )
    cbar.ax.tick_params(labelsize=7.5, colors="#333333")

    # ── Legend — created once ────────────────────────────────────────────────
    _legend_handles = [
        mpatches.Patch(facecolor="royalblue",   edgecolor="none", label="Discount (D)"),
        mpatches.Patch(facecolor="firebrick",   edgecolor="none", label="Standard (S)"),
        mpatches.Patch(facecolor="forestgreen", edgecolor="none", label="Bio (B)"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="gold", markeredgecolor="black",
               markersize=13, linestyle="None",
               label="Selected store (target)"),
    ]
    _leg = ax_map.legend(
        handles=_legend_handles, loc="upper left", fontsize=8.5,
        framealpha=0.92, facecolor="white", edgecolor="#aaaaaa",
    )
    ax_map.add_artist(_leg)   # preserve after ax_map.set_title calls

    # ── Slider ────────────────────────────────────────────────────────────────
    init_id = max(0, min(args.init_store, N - 1))
    slider  = Slider(
        ax=ax_slide,
        label="Store ID",
        valmin=0, valmax=N - 1,
        valinit=init_id,
        valstep=1,
        color="#4477cc",
    )
    slider.label.set_fontsize(9)
    slider.valtext.set_fontsize(9)

    # ── Mutable variable artists (replaced on each slider event) ─────────────
    _var: dict = {
        "catchment": None,  # PolyCollection
        "star":      None,  # PathCollection (target star)
    }

    # ── Update function ───────────────────────────────────────────────────────
    def update(_val=None) -> None:
        sid = int(slider.val)

        # Remove old variable artists (safe even on first call where they're None)
        for key in ("catchment", "star"):
            if _var[key] is not None:
                _var[key].remove()
                _var[key] = None

        # ── Invert CSR for this store ────────────────────────────────────────
        cell_rows_all, cell_tt_all = invert_csr_for_store(indptr, indices, tt_min, sid)

        # ── Filter to populated cells ────────────────────────────────────────
        pop_bool   = pop_mask.values
        keep       = pop_bool[cell_rows_all]
        cell_rows_pop = cell_rows_all[keep]
        cell_tt_pop   = cell_tt_all[keep]
        n_catch_total = len(cell_rows_all)
        n_catch_pop   = len(cell_rows_pop)

        # Positional indices inside grid_pop (for _pop_verts lookup)
        pop_iloc = np.array(
            [orig_to_pop_pos.get(int(r), -1) for r in cell_rows_pop],
            dtype=np.int64,
        )
        valid        = pop_iloc >= 0
        pop_iloc_v   = pop_iloc[valid]
        cell_tt_v    = cell_tt_pop[valid]

        # ── Competitor stats (populated cells only) ──────────────────────────
        comp_overlap: dict[int, int] = {}
        stores_per_cell: list[int] = []
        for ci in cell_rows_pop:
            s, e = int(indptr[ci]), int(indptr[ci + 1])
            stores_per_cell.append(e - s)
            for p in range(s, e):
                jj = int(indices[p])
                if jj != sid:
                    comp_overlap[jj] = comp_overlap.get(jj, 0) + 1

        n_comp   = len(comp_overlap)
        mean_s   = float(np.mean(stores_per_cell)) if stores_per_cell else 0.0

        # ── Catchment PolyCollection (populated cells, alpha=0.5) ────────────
        if pop_iloc_v.size > 0:
            verts  = [_pop_verts[i] for i in pop_iloc_v]
            colors = CMAP_CATCHMENT(norm_cb(cell_tt_v))   # (K, 4) RGBA
            coll   = PolyCollection(
                verts,
                facecolors=colors,
                edgecolors="none",
                linewidths=0,
                alpha=0.50,
                zorder=2,
            )
            ax_map.add_collection(coll)
            _var["catchment"] = coll

        # ── Target store star ────────────────────────────────────────────────
        tx = float(stores_3857.iloc[sid].geometry.x)
        ty = float(stores_3857.iloc[sid].geometry.y)
        star = ax_map.scatter(
            [tx], [ty],
            c="gold", s=350, marker="*",
            edgecolors="black", linewidths=1.0,
            zorder=8,
        )
        _var["star"] = star

        # ── Map title ────────────────────────────────────────────────────────
        target_row = stores_gdf.iloc[sid]
        chain      = str(target_row.get("chain",      "?"))
        chain_type = str(target_row.get("chain_type", "?"))
        ax_map.set_title(
            f"Store {sid}: {chain}  ({chain_type})  ·  "
            f"{n_catch_pop:,} populated cells  ·  {n_comp} competitors",
            fontsize=11, color="#222222", pad=6,
        )

        # ── Info panel ───────────────────────────────────────────────────────
        ax_info.cla()
        ax_info.set_facecolor("#f7f7f9")
        ax_info.set_xticks([]); ax_info.set_yticks([])
        for spine in ax_info.spines.values():
            spine.set_edgecolor("#ccccdd")

        ax_info.set_title("  Catchment Info  ", fontsize=9.5, color="#333355",
                           fontweight="bold", pad=4)

        loc_x = float(target_row.geometry.x)
        loc_y = float(target_row.geometry.y)
        text  = build_info_text(
            sid=sid, chain=chain, chain_type=chain_type,
            loc_x=loc_x, loc_y=loc_y,
            catchment_minutes=catchment_minutes,
            catchment_k_min=catchment_k_min,
            catchment_k_max=catchment_k_max,
            n_catch_pop=n_catch_pop,
            n_catch_total=n_catch_total,
            M_pop=M_pop, M=M,
            tt_pop=cell_tt_v,
            competitors=comp_overlap,
            n_competitors=n_comp,
            mean_stores_catch_cell=mean_s,
            global_mean=global_mean_per_cell,
            stores_df=stores_gdf,
        )
        ax_info.text(
            0.02, 0.985, text,
            transform=ax_info.transAxes,
            fontsize=7.2, family="monospace",
            color="#1a1a2a",
            verticalalignment="top",
            horizontalalignment="left",
        )

        fig.canvas.draw_idle()

    # ── Wire slider and do initial draw ──────────────────────────────────────
    slider.on_changed(update)
    update()   # initial draw at init_id

    # ── Figure suptitle ───────────────────────────────────────────────────────
    fig.text(
        0.315, 0.998,
        f"Berlin Inner-Ringbahn · Catchment Explorer  "
        f"[{catchment_minutes:.0f}-min radius | k_min={catchment_k_min} | k_max={catchment_k_max}]",
        ha="center", va="top", fontsize=12, fontweight="bold", color="#222222",
    )

    plt.show()


if __name__ == "__main__":
    main()
