#!/usr/bin/env python3
"""
Visualize the catchment range of a single grocery store.

The "catchment range of store j" is the set of INSPIRE 100m grid cells
that include store j in their transit-time consideration set, as built by
build_catchment() from travel_times.parquet. This set is stationary — it is
computed once at load time and does not change during simulation. Only the
demand FLOW within the catchment varies per period as prices change.

Usage (activate conda env py314 first):
    cd DEDA_LLM_Spatial_Hotelling
    python scripts/viz_catchment.py 42
    python scripts/viz_catchment.py 42 --catchment-minutes 25
    python scripts/viz_catchment.py 42 --out results/catchment_42.png

Color convention:
    Discount (D) — royalblue
    Standard (S) — firebrick
    Bio (B)      — forestgreen
    Target store — gold star ★

Cells are shaded by transit travel time to store j:
    yellow = near (0 min) → red/brown = far (catchment_minutes)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D


# ── Constants ─────────────────────────────────────────────────────────────────

CHAIN_TYPE_COLORS = {
    "discount": "royalblue",
    "standard": "firebrick",
    "bio":      "forestgreen",
}
CHAIN_TYPE_LABELS = {
    "discount": "Discount (D)",
    "standard": "Standard (S)",
    "bio":      "Bio (B)",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Catchment-range visualisation for a single grocery store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("store_id", type=int, help="Store integer ID (0-indexed, positional index in supermarkets.parquet)")
    p.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/env/berlin_inner_ring_calibrated.yaml",
        help="Env config YAML",
    )
    p.add_argument("--catchment-minutes", type=float, default=None,
                   help="Override catchment_minutes from config (transit-time radius)")
    p.add_argument("--catchment-k-min",   type=int,   default=None,
                   help="Override catchment_k_min (guaranteed min stores/cell)")
    p.add_argument("--catchment-k-max",   type=int,   default=None,
                   help="Override catchment_k_max (hard cap stores/cell)")
    p.add_argument("--out", type=Path, default=None,
                   help="Save figure to this path instead of plt.show()")
    return p.parse_args()


# ── CSR inversion helper ──────────────────────────────────────────────────────

def invert_csr_for_store(
    indptr:     np.ndarray,   # (M+1,) int64
    indices:    np.ndarray,   # (NNZ,) int32
    tt_min:     np.ndarray,   # (NNZ,) float64
    target_col: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (cell_row_indices, travel_times) for every cell that includes
    target_col in its consideration set.

    The CSR is cell-indexed: indices[indptr[i]:indptr[i+1]] = stores for cell i.
    To find which cells contain store j we scan all positions p where
    indices[p] == j, then recover the cell i via binary search on indptr.

    Parameters
    ----------
    indptr     : CSR row-pointer array (M+1,)
    indices    : store column indices (NNZ,)
    tt_min     : transit times for each CSR entry (NNZ,)
    target_col : store integer index j

    Returns
    -------
    cell_rows  : (K,) int64 — row indices of cells that include store j
    cell_tt    : (K,) float64 — travel time from each such cell to store j
    """
    # All positions p in indices where indices[p] == j
    hit_pos = np.where(np.asarray(indices, dtype=np.int64) == target_col)[0]
    if len(hit_pos) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    # For position p in [indptr[i], indptr[i+1]):
    #   np.searchsorted(indptr, p, side='right') - 1 == i
    cell_rows = (
        np.searchsorted(np.asarray(indptr, dtype=np.int64), hit_pos, side="right") - 1
    ).astype(np.int64)

    return cell_rows, tt_min[hit_pos].astype(np.float64)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
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

    grid_path   = ROOT / cfg["grid_path"]
    stores_path = ROOT / cfg["stores_path"]
    tt_path     = ROOT / cfg["travel_times_path"]

    # ── Load stores ───────────────────────────────────────────────────────────
    print(f"Loading supermarkets from {stores_path}...", flush=True)
    stores = gpd.read_parquet(stores_path).reset_index(drop=True)
    N = len(stores)

    if not (0 <= args.store_id < N):
        sys.exit(f"ERROR: store_id={args.store_id} out of range [0, {N-1}]")

    target     = stores.iloc[args.store_id]
    chain      = str(target.get("chain",      "Unknown"))
    chain_type = str(target.get("chain_type", "Unknown"))
    loc_x      = float(target.geometry.x)
    loc_y      = float(target.geometry.y)

    # ── Load demand grid ──────────────────────────────────────────────────────
    print(f"Loading demand grid from {grid_path}...", flush=True)
    grid_gdf = gpd.read_parquet(grid_path)
    grid_gdf = grid_gdf.sort_values("GITTER_ID_100m").reset_index(drop=True)

    n_dupes = grid_gdf["GITTER_ID_100m"].duplicated().sum()
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes} duplicate GITTER_ID_100m rows — deduplicating.")
        grid_gdf = (
            grid_gdf.drop_duplicates("GITTER_ID_100m", keep="first")
            .reset_index(drop=True)
        )

    M         = len(grid_gdf)
    cell_ids  = grid_gdf["GITTER_ID_100m"].tolist()
    store_ids = [str(i) for i in range(N)]
    print(f"  Grid: {M:,} cells | Stores: {N}", flush=True)

    # ── Load travel times ─────────────────────────────────────────────────────
    print(f"Loading travel times from {tt_path}...", flush=True)
    tt_raw = pd.read_parquet(tt_path)
    tt_raw["from_id"] = tt_raw["from_id"].astype(str)
    tt_raw["to_id"]   = tt_raw["to_id"].astype(str)
    print(f"  Travel-time table: {len(tt_raw):,} rows", flush=True)

    # ── Build catchment CSR ───────────────────────────────────────────────────
    print(
        f"\nBuilding catchment CSR  "
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
    nnz                   = int(indptr[-1])
    global_mean_per_cell  = nnz / M if M > 0 else 0.0

    # ── Invert CSR → catchment of store j ────────────────────────────────────
    print(f"Inverting CSR for store {args.store_id}...", flush=True)
    cell_rows, cell_tt = invert_csr_for_store(indptr, indices, tt_min, args.store_id)
    n_catch = len(cell_rows)

    # ── Competitor overlap ────────────────────────────────────────────────────
    # For each cell in store j's catchment, collect all OTHER stores in that cell.
    # competitor_overlap[j'] = number of shared catchment cells with target store j.
    competitor_overlap: dict[int, int] = {}
    stores_per_catchment_cell: list[int] = []

    for ci in cell_rows:
        s, e = int(indptr[ci]), int(indptr[ci + 1])
        stores_in_cell = int(e - s)
        stores_per_catchment_cell.append(stores_in_cell)
        for p in range(s, e):
            jj = int(indices[p])
            if jj != args.store_id:
                competitor_overlap[jj] = competitor_overlap.get(jj, 0) + 1

    n_competitors = len(competitor_overlap)
    mean_stores_in_catch_cell = (
        float(np.mean(stores_per_catchment_cell)) if stores_per_catchment_cell else 0.0
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  Store ID    : {args.store_id}")
    print(f"  Chain       : {chain}")
    print(f"  Chain type  : {chain_type}")
    print(f"  Location    : ({loc_x:.0f}, {loc_y:.0f})  [EPSG:3035, metres]")
    print()
    print(f"  Catchment parameters (stationary — set at load time, not t)")
    print(f"    catchment_minutes : {catchment_minutes:.0f} min  "
          f"[transit time, r5py GTFS+OSM, one-way]")
    print(f"    catchment_k_min   : {catchment_k_min}  "
          f"[min stores guaranteed per cell, regardless of radius]")
    print(f"    catchment_k_max   : {catchment_k_max}  [hard cap per cell]")
    print(f"    distance metric   : transit travel time (minutes), NOT Euclidean")
    print(f"    stationarity      : YES — CSR built once; unchanged across all T steps")
    print(f"    what changes at t : demand FLOW within the catchment (prices → MNL shares)")
    print()
    print(f"  Catchment of store {args.store_id} (inverted CSR view)")
    print(f"    Cells in catchment : {n_catch:,}  of  {M:,}  "
          f"({100.0 * n_catch / M:.1f}% of grid)")
    if n_catch > 0:
        print(f"    Travel time [min]  : min={cell_tt.min():.1f}  "
              f"mean={cell_tt.mean():.1f}  max={cell_tt.max():.1f}")
    print()
    print(f"  Competitor landscape")
    print(f"    Stores sharing ≥1 catchment cell  : {n_competitors}  "
          f"(of {N-1} total other stores)")
    print(f"    Mean stores/catchment-cell        : {mean_stores_in_catch_cell:.1f}  "
          f"(incl. store {args.store_id})")
    print(f"    Global mean stores/cell (full CSR): {global_mean_per_cell:.1f}")
    print(f"{sep}")
    print()

    top5 = sorted(competitor_overlap.items(), key=lambda x: -x[1])[:5]
    print(f"  Top-5 competitors by shared-cell count:")
    print(f"  {'ID':>4}  {'Type':9}  {'Chain':<28}  {'Shared cells':>12}")
    print(f"  {'-'*60}")
    for jj, cnt in top5:
        c_chain = str(stores.iloc[jj].get("chain", "?"))
        c_type  = str(stores.iloc[jj].get("chain_type", "?"))
        print(f"  {jj:4d}  {c_type:9s}  {c_chain:<28s}  {cnt:>12d}")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 11))

    # 1. Background: all grid cells, very light
    grid_gdf.plot(
        ax=ax, color="#f0f0f0", edgecolor="#d8d8d8",
        linewidth=0.06, zorder=1,
    )

    # 2. Catchment cells: shaded by travel time (YlOrRd: yellow=near, red=far)
    if n_catch > 0:
        catch_gdf = grid_gdf.iloc[cell_rows].copy()
        catch_gdf = catch_gdf.reset_index(drop=True)
        catch_gdf["tt"] = cell_tt
        catch_gdf.plot(
            ax=ax, column="tt", cmap="YlOrRd",
            vmin=0.0, vmax=catchment_minutes,
            edgecolor="none", linewidth=0, zorder=2,
            legend=True,
            legend_kwds={
                "label": f"Transit travel time to store {args.store_id} (min)  "
                         f"[yellow=near, red=far]",
                "shrink": 0.55,
                "fraction": 0.028,
                "pad": 0.02,
                "aspect": 28,
            },
        )

    # 3. All other stores, colour-coded by chain type (small markers, behind target)
    for ct, color in CHAIN_TYPE_COLORS.items():
        subset = stores[(stores["chain_type"] == ct) & (stores.index != args.store_id)]
        if len(subset) == 0:
            continue
        ax.scatter(
            subset.geometry.x, subset.geometry.y,
            c=color, s=14, alpha=0.80, zorder=4, linewidths=0,
        )

    # 4. Target store: prominent gold star
    ax.scatter(
        [loc_x], [loc_y],
        c="gold", s=320, marker="*", zorder=6,
        edgecolors="black", linewidths=1.3,
    )

    # 5. Title
    ax.set_title(
        f"Catchment range — Store {args.store_id}: {chain}  ({chain_type})\n"
        f"{n_catch:,} cells  |  {catchment_minutes:.0f}-min transit radius  "
        f"|  {n_competitors} competitor stores  "
        f"|  avg {mean_stores_in_catch_cell:.1f} stores/cell (in catchment)",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("Easting (EPSG:3035, m)", fontsize=9)
    ax.set_ylabel("Northing (EPSG:3035, m)", fontsize=9)
    ax.tick_params(labelsize=8)

    # 6. Legend
    legend_handles = [
        mpatches.Patch(facecolor="royalblue",   edgecolor="none", label="Discount (D)"),
        mpatches.Patch(facecolor="firebrick",   edgecolor="none", label="Standard (S)"),
        mpatches.Patch(facecolor="forestgreen", edgecolor="none", label="Bio (B)"),
        Line2D(
            [0], [0], marker="*", color="w",
            markerfacecolor="gold", markeredgecolor="black",
            markersize=13, label=f"Store {args.store_id}: {chain}",
        ),
        mpatches.Patch(facecolor="#f0f0f0", edgecolor="#d8d8d8",
                       label="Outside catchment"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.92)

    ax.set_aspect("equal")
    plt.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
