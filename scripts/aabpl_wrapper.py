"""
GEO_02 — High-Employment Cluster Identification via AABPL
==========================================================
Identifies spatially concentrated employment hotspots across Berlin using the
AABPL (Approximation Algorithm for Building Place Locations) clustering method.

Typical usage from GEO_02_city_data.ipynb
------------------------------------------
    import sys
    sys.path.insert(0, str(Path.cwd().parent / "scripts"))
    from aabpl_wrapper import detect_employment_clusters

    clusters_df, summary = detect_employment_clusters(
        gebaeude_centroid,          # GeoDataFrame with approx_empl
        weight_col="approx_empl",
        out_dir=PATH_PROCESSED / "employment_clusters",
    )

Input GeoDataFrame
------------------
Any CRS is accepted — the function reprojects to WGS84 internally.
Required column: ``approx_empl`` (or whatever you pass as ``weight_col``).
Geometry must be point (building centroids).

Output
------
clusters_df : pd.DataFrame
    One row per detected AABPL cluster with columns:
    cluster_id, sum, n_cells, centroid_x, centroid_y.

summary : pd.DataFrame
    Single-row summary with dominant cluster centroid and metadata.

If ``out_dir`` is provided, both DataFrames are written as CSV files:
    <out_dir>/berlin_employment_clusters.csv
    <out_dir>/berlin_cluster_summary.csv
"""

from __future__ import annotations

from pathlib import Path    

import geopandas as gpd
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from aabpl.main import detect_cluster_cells

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_WGS84 = "EPSG:4326"


def detect_employment_clusters(
    gdf: gpd.GeoDataFrame,
    *,
    weight_col: str = "approx_empl",
    radius_m: int = 500,
    k_percentile: float = 99.5,
    min_empl: float = 1.0,
    out_dir: Path | str | None = None,
    silent: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect high-employment clusters in a GeoDataFrame of building centroids.

    Parameters
    ----------
    gdf : GeoDataFrame
        Building centroids for Berlin (any CRS; reprojected to WGS84 internally).
        Must contain a numeric column named ``weight_col``.
    weight_col : str
        Column used as the AABPL importance weight.  Default ``"approx_empl"``.
    radius_m : int
        AABPL search radius in metres.  500 m works well for building-level
        data; raise to 1000 m if no clusters are found.
    k_percentile : float
        Percentile threshold for cluster membership (AABPL default = 99.5).
        Lower to 95.0 for Berlin's polycentric employment structure if needed.
    min_empl : float
        Buildings with ``weight_col <= min_empl`` are excluded before running
        AABPL to remove residential or vacant structures.
    out_dir : Path or str, optional
        If provided, ``berlin_employment_clusters.csv`` and
        ``berlin_cluster_summary.csv`` are written to this directory.
    silent : bool
        Suppress AABPL's internal progress output.

    Returns
    -------
    clusters_df : pd.DataFrame
        One row per detected cluster.
    summary : pd.DataFrame
        Single-row summary with dominant cluster centroid and metadata.
    """
    # ------------------------------------------------------------------
    # 1. Validate input
    # ------------------------------------------------------------------
    if weight_col not in gdf.columns:
        raise ValueError(
            f"Column '{weight_col}' not found in GeoDataFrame. "
            f"Available columns: {gdf.columns.tolist()}"
        )

    # ------------------------------------------------------------------
    # 2. Reproject → WGS84 and extract lon/lat
    # ------------------------------------------------------------------
    gdf_wgs = gdf.to_crs(_WGS84) if gdf.crs is not None else gdf.copy()
    pts = pd.DataFrame({
        "lon":      gdf_wgs.geometry.x,
        "lat":      gdf_wgs.geometry.y,
        weight_col: gdf_wgs[weight_col],
    }).dropna()
    pts = pts[pts[weight_col] > min_empl].copy()

    n = len(pts)
    if not silent:
        print(f"  {n:,} buildings with {weight_col} > {min_empl}")
    if n < 10:
        raise ValueError(
            f"Only {n} buildings remain after filtering — too few for AABPL. "
            "Check that the GeoDataFrame has valid non-zero values in "
            f"'{weight_col}'."
        )

    # ------------------------------------------------------------------
    # 3. Run AABPL
    # ------------------------------------------------------------------
    # For each building centroid, AABPL:
    #   (a) sums weight_col of all neighbours within radius_m,
    #   (b) compares each local sum to a random-point null distribution,
    #   (c) labels locations above k_percentile as cluster cells,
    #   (d) merges adjacent cluster cells into named cluster polygons.
    if not silent:
        print(
            f"  Running AABPL (r={radius_m} m, k={k_percentile}th pct) "
            f"over {n:,} building centroids..."
        )

    method = ""
    weight_share = 1.0

    try:
        grid = detect_cluster_cells(
            pts               = pts,
            crs               = _WGS84,
            r                 = radius_m,
            c                 = [weight_col],
            k_th_percentile   = k_percentile,
            exclude_pt_itself = False,
            sum_suffix        = f"_{radius_m}m",
            cluster_suffix    = "_cluster",
            x                 = "lon",
            y                 = "lat",
            silent            = silent,
        )

        # create_clusters_df_for_column is keyed by the base column name.
        clusters_df = grid.create_clusters_df_for_column(
            cluster_column = weight_col,
            target_crs     = _WGS84,
        )

        if len(clusters_df) == 0:
            raise ValueError(
                "No clusters found. Try lowering k_percentile (e.g. 95.0) "
                "or increasing radius_m (e.g. 1000)."
            )

        if not silent:
            print(f"  {len(clusters_df)} cluster(s) found:")
            print(
                clusters_df[["cluster_id", "sum", "n_cells",
                             "centroid_x", "centroid_y"]].to_string(index=False)
            )

        dominant     = clusters_df.loc[clusters_df["sum"].idxmax()]
        dom_lon      = float(dominant["centroid_x"])
        dom_lat      = float(dominant["centroid_y"])
        weight_share = float(dominant["sum"] / clusters_df["sum"].sum())
        method       = (
            f"AABPL dominant cluster centroid "
            f"(r={radius_m} m, k={k_percentile}, share={weight_share:.1%})"
        )

        if not silent:
            print(
                f"\n  Dominant cluster → lon={dom_lon:.5f}, lat={dom_lat:.5f} "
                f"({weight_share:.1%} of total {weight_col})"
            )
            if len(clusters_df) > 1:
                print(
                    f"  Note: {len(clusters_df)} clusters — Berlin's polycentric "
                    "employment structure is reflected."
                )

    except Exception as exc:
        # Fallback: employment-weighted centroid of top-10 % buildings
        if not silent:
            print(f"  AABPL error: {exc}")
            print(f"  Fallback: weighted centroid of top 10 % buildings by {weight_col}...")

        threshold = pts[weight_col].quantile(0.90)
        top_pts   = pts[pts[weight_col] >= threshold]
        w         = top_pts[weight_col].values
        dom_lon   = float(np.average(top_pts["lon"].values, weights=w))
        dom_lat   = float(np.average(top_pts["lat"].values, weights=w))
        method    = f"Weighted centroid of top 10 % buildings by {weight_col} (AABPL fallback)"

        clusters_df = pd.DataFrame([{
            "cluster_id": 0,
            "sum":        float(w.sum()),
            "n_cells":    len(top_pts),
            "centroid_x": dom_lon,
            "centroid_y": dom_lat,
        }])

        if not silent:
            print(f"  Fallback centroid → lon={dom_lon:.5f}, lat={dom_lat:.5f}")

    # ------------------------------------------------------------------
    # 4. Build summary
    # ------------------------------------------------------------------
    summary = pd.DataFrame([{
        "city":         "berlin",
        "n_buildings":  n,
        "n_clusters":   len(clusters_df),
        "dominant_lon": dom_lon,
        "dominant_lat": dom_lat,
        "empl_share":   round(weight_share, 4),
        "method":       method,
    }])

    # ------------------------------------------------------------------
    # 5. Optionally save
    # ------------------------------------------------------------------
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clusters_df.to_csv(out_dir / "berlin_employment_clusters.csv", index=False)
        summary.to_csv(out_dir / "berlin_cluster_summary.csv", index=False)
        if not silent:
            print(f"\n  Saved → {out_dir / 'berlin_employment_clusters.csv'}")
            print(f"  Saved → {out_dir / 'berlin_cluster_summary.csv'}")

    return clusters_df, summary


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # When run directly (e.g. `python scripts/aabpl_wrapper.py`), load the
    # snapshot that GEO_02_city_data.ipynb writes before calling this script.
    REPO_ROOT  = Path(__file__).resolve().parents[1]
    DATA_PROC  = REPO_ROOT / "data" / "processed"
    INPUT_FILE = DATA_PROC / "gebaeude_for_aabpl.parquet"

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run GEO_02_city_data.ipynb first and save building centroids "
            "(geometry in any CRS, approx_empl column) to that path."
        )

    print(f"Loading {INPUT_FILE} ...")
    raw = gpd.read_parquet(INPUT_FILE)

    clusters, summary = detect_employment_clusters(
        raw,
        out_dir=DATA_PROC / "employment_clusters",
    )

    print(f"\n{'='*60}")
    print("SUMMARY — Employment Clusters (Berlin)")
    print(f"{'='*60}")
    print(summary[["city", "n_buildings", "n_clusters",
                   "dominant_lon", "dominant_lat"]].to_string(index=False))
    print(
        "\nNext → load 'berlin_employment_clusters.csv' in the Hotelling "
        "simulation to position firms relative to Berlin's employment density."
    )
