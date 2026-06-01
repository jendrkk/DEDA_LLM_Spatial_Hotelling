"""Berlin spatial market snapshot, animation, and interactive slider.

Responsibility: render per-cell choropleth market metrics from a finished
DenseLog run on a contextily OpenStreetMap basemap, plus a FuncAnimation and
an ipywidgets interactive slider for Jupyter notebooks.

Public API
----------
load_run              Load run artefacts and spatial GeoDataFrames.
prices_efforts_at     Decode DenseLog index arrays at a given time step.
plot_market_snapshot  Single-frame choropleth + scatter on OSM basemap.
animate_market        FuncAnimation over a sequence of time steps.
interactive_slider    ipywidgets IntSlider calling plot_market_snapshot.

Key dependencies
----------------
matplotlib (viz), contextily>=1.5 (viz), xyzservices>=2023.10 (viz),
geopandas (spatial), ipywidgets (notebooks – optional).

Data alignment rules
--------------------
Stores   : ``gpd.read_parquet(stores_path).reset_index(drop=True)``
           → row *j* == firm *j* == DenseLog column *j*.
Cells    : ``gpd.read_parquet(grid_path).sort_values("GITTER_ID_100m")
           .drop_duplicates(subset="GITTER_ID_100m", keep="first")
           .reset_index(drop=True)``
           → row *i* == ``city.dist2_km2`` row *i* == cell-metric index *i*.
Both parquets live in EPSG:3035; reprojected to EPSG:3857 for contextily.

References
----------
Anderson, de Palma & Thisse (1992) *Discrete Choice Theory of Product
Differentiation*, Ch. 3.
Calvano, E. et al. (2020) *Artificial Intelligence, Algorithmic Pricing,
and Collusion*, AER §II.A.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from hotelling.core.market import cell_metrics

# Repository root: src/hotelling/viz/spatial_map.py → parents[3] = repo root
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Lazy-import guards
# ---------------------------------------------------------------------------

def _require_mpl():
    try:
        import matplotlib
        return matplotlib
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for spatial maps. "
            "Install with: pip install 'hotelling[viz]'"
        ) from exc


def _require_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for spatial maps. "
            "Install with: pip install 'hotelling[viz]'"
        ) from exc


def _require_ctx():
    try:
        import contextily as ctx
        return ctx
    except ImportError as exc:
        raise ImportError(
            "contextily is required for OSM basemap tiles. "
            "Install with: pip install 'hotelling[viz]'"
        ) from exc


def _require_gpd():
    try:
        import geopandas as gpd
        return gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for spatial maps. "
            "Install with: pip install 'hotelling[spatial]'"
        ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_env_cfg(cfg: dict) -> dict:
    """Extract the env sub-dict from a run config, falling back to flat layout."""
    return cfg.get("env", cfg)


def _get_transport_cost(cfg: dict) -> float:
    """Read transport_cost from a run config dict (env block or top-level)."""
    env = _get_env_cfg(cfg)
    return float(env.get("transport_cost", 0.01))


def _resolve_data_path(raw: str) -> Path:
    """Resolve a data path: absolute paths pass through; relative → repo root."""
    p = Path(raw)
    return p if p.is_absolute() else _REPO_ROOT / p


def _geom_verts(geom: Any) -> np.ndarray:
    """Return exterior ring coordinates as (K, 2) float64 for a polygon-like geom."""
    if geom.geom_type == "Polygon":
        return np.asarray(geom.exterior.coords, dtype=np.float64)[:, :2]
    if geom.geom_type == "MultiPolygon":
        largest = max(geom.geoms, key=lambda g: g.area)
        return np.asarray(largest.exterior.coords, dtype=np.float64)[:, :2]
    # Fallback for degenerate geometries: single degenerate triangle
    return np.zeros((3, 2), dtype=np.float64)


def _build_poly_collection(
    grid_gdf: Any,
    values: np.ndarray,
    cmap: Any,
    norm: Any,
    alpha: float = 0.65,
) -> Any:
    """Build a :class:`matplotlib.collections.PolyCollection` from a polygon GDF.

    Parameters
    ----------
    grid_gdf : GeoDataFrame of polygon cells (EPSG:3857).
    values : (M,) array to colour by.
    cmap, norm : matplotlib colormap and norm objects.
    alpha : polygon fill opacity.

    Returns
    -------
    matplotlib.collections.PolyCollection
    """
    from matplotlib.collections import PolyCollection

    verts = [_geom_verts(g) for g in grid_gdf.geometry]
    return PolyCollection(
        verts,
        array=np.asarray(values, dtype=np.float64),
        cmap=cmap,
        norm=norm,
        linewidths=0,
        edgecolors="none",
        alpha=alpha,
        zorder=2,
    )


def _categorical_setup(firms: list) -> Tuple[Any, Any, list]:
    """Return (cmap, norm, chain_labels) for dominant_chain choropleth."""
    mpl = _require_mpl()
    plt = _require_plt()

    N = len(firms)
    chain_labels = [getattr(f, "chain", None) or f.id for f in firms]
    cmap_obj = plt.get_cmap("tab20" if N <= 20 else "hsv", N)
    norm_obj = mpl.colors.BoundaryNorm(
        boundaries=np.arange(-0.5, N + 0.5, 1.0), ncolors=N
    )
    return cmap_obj, norm_obj, chain_labels


def _continuous_setup(metric: str, dense_log: Any) -> Tuple[Any, Any]:
    """Return (cmap_viridis, norm) with vmin/vmax from price_grid for price-like metrics."""
    mpl = _require_mpl()
    plt = _require_plt()

    cmap_obj = plt.get_cmap("viridis")  # caller can override; placeholder
    if metric == "expected_price":
        vmin = float(dense_log.price_grid.min())
        vmax = float(dense_log.price_grid.max())
    else:
        vmin, vmax = 0.0, 1.0  # overridden after first-frame computation
    norm_obj = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return cmap_obj, norm_obj


def _add_dominant_chain_legend(ax: Any, firms: list, cmap: Any, norm: Any) -> None:
    """Attach a categorical chain legend to *ax* for dominant_chain metric."""
    from matplotlib.patches import Patch

    seen: dict = {}
    handles = []
    for j, f in enumerate(firms):
        label = getattr(f, "chain", None) or f.id
        if label not in seen:
            seen[label] = j
            color = cmap(norm(j))
            handles.append(Patch(facecolor=color, label=label))
    ax.legend(handles=handles, title="Dominant chain", loc="upper right",
              fontsize=7, title_fontsize=8, framealpha=0.8)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_run(
    run_dir: Path | str,
) -> Tuple[Any, Any, list, Any, Any, dict]:
    """Load all artefacts for a finished DenseLog run.

    Reads ``run_dir/config.yaml``, rebuilds :class:`~hotelling.core.city.City`
    and firms via :func:`~hotelling.spatial.loader.load_berlin_city`, loads the
    :class:`~hotelling.simulation.dense_log.DenseLog`, and reloads the grid and
    store parquets for geometry only — reprojected to EPSG:3857 for contextily.

    Parameters
    ----------
    run_dir : Path to a finished simulation run directory (must contain
        ``config.yaml`` and DenseLog binary files).

    Returns
    -------
    dense_log : DenseLog
    city : City
    firms : list[Firm]
    grid_gdf_3857 : GeoDataFrame of demand-grid cells in EPSG:3857.
        Row order matches ``city.dist2_km2`` rows.
    stores_gdf_3857 : GeoDataFrame of stores in EPSG:3857.
        Row order matches ``city.firms`` and DenseLog columns.
    cfg : dict  — the raw config loaded from ``run_dir/config.yaml``.

    Raises
    ------
    FileNotFoundError
        If ``run_dir/config.yaml`` or any required parquet is absent.
    KeyError
        If ``lambda_val`` is missing from the env config block.

    Notes
    -----
    Relative parquet paths in the config are resolved against the repository
    root (the directory containing ``src/``), not the process working directory.

    Data alignment guarantees
    -------------------------
    Cells are sorted by ``GITTER_ID_100m`` ascending and deduplicated
    (mirroring :func:`~hotelling.spatial.loader.load_berlin_city`) so that
    ``grid_gdf_3857.iloc[i]`` corresponds to ``city.dist2_km2[i, :]``.
    Stores are ``reset_index(drop=True)`` so that ``stores_gdf_3857.iloc[j]``
    corresponds to ``city.firms[j]``.
    """
    import yaml

    gpd = _require_gpd()

    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {run_dir}")

    with cfg_path.open() as f:
        cfg: dict = yaml.safe_load(f) or {}

    env_cfg = _get_env_cfg(cfg)

    # Resolve parquet paths (relative → repo root)
    grid_path = _resolve_data_path(
        env_cfg.get("grid_path", "data/processed/demand_grid.parquet")
    )
    stores_path = _resolve_data_path(
        env_cfg.get("stores_path", "data/processed/supermarkets.parquet")
    )
    tt_path = _resolve_data_path(
        env_cfg.get("travel_times_path", "data/processed/travel_times.parquet")
    )

    # Build City + Firms (mirrors runner.py logic exactly)
    from hotelling.spatial.loader import load_berlin_city

    city, firms = load_berlin_city(
        grid_path=grid_path,
        stores_path=stores_path,
        travel_times_path=tt_path,
        lambda_val=float(env_cfg["lambda_val"]),
        q_S=float(env_cfg.get("q_S", 0.8)),
        q_B=float(env_cfg.get("q_B", 1.5)),
        alpha_L=float(env_cfg.get("alpha_L", 0.5)),
        alpha_H=float(env_cfg.get("alpha_H", 1.5)),
        beta_effort=float(env_cfg.get("beta_effort", 0.001)),
        kappa0=float(env_cfg.get("kappa0", 1.0)),
        store_size=float(env_cfg.get("store_size", 600.0)),
        transport_cost=float(env_cfg.get("transport_cost", 0.01)),
        a0=float(env_cfg.get("a0", env_cfg.get("outside_option", -1.0))),
        mu=float(env_cfg.get("mu", env_cfg.get("logit_scale", 0.25))),
        nan_fill_minutes=float(env_cfg.get("nan_fill_minutes", 120.0)),
        marginal_cost_D=float(env_cfg.get("marginal_cost_D", 0.0)),
        marginal_cost_S=float(env_cfg.get("marginal_cost_S", 0.0)),
        marginal_cost_B=float(env_cfg.get("marginal_cost_B", 0.0)),
    )

    # Load DenseLog
    from hotelling.simulation.dense_log import DenseLog

    dense_log = DenseLog.load(run_dir)

    # Reload geometry-only GDFs with canonical cell ordering
    grid_raw = gpd.read_parquet(grid_path)
    grid_gdf = (
        grid_raw
        .sort_values("GITTER_ID_100m")
        .drop_duplicates(subset="GITTER_ID_100m", keep="first")
        .reset_index(drop=True)
    )
    stores_gdf = gpd.read_parquet(stores_path).reset_index(drop=True)

    # Reproject to EPSG:3857 (Web Mercator) for contextily
    grid_gdf_3857 = grid_gdf.to_crs(epsg=3857)
    stores_gdf_3857 = stores_gdf.to_crs(epsg=3857)

    return dense_log, city, firms, grid_gdf_3857, stores_gdf_3857, cfg


def prices_efforts_at(
    dense_log: Any,
    t: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode DenseLog index arrays into prices and efforts at step *t*.

    Parameters
    ----------
    dense_log : DenseLog instance (loaded via :meth:`DenseLog.load`).
    t : simulation step index (0-based).

    Returns
    -------
    prices_t : ndarray of shape (N,), float64 — prices at step *t*.
    efforts_t : ndarray of shape (N,), float64 — effort levels at step *t*.

    Notes
    -----
    ``price_idx`` and ``effort_idx`` are stored as ``int8`` indices into
    ``price_grid`` and ``effort_grid`` respectively.  The int8 encoding uses
    the range ``[0, m-1]`` where *m* is the grid size.
    """
    pidx = dense_log.price_idx[t].astype(np.int64)
    eidx = dense_log.effort_idx[t].astype(np.int64)
    prices_t = dense_log.price_grid[pidx].astype(np.float64)
    efforts_t = dense_log.effort_grid[eidx].astype(np.float64)
    return prices_t, efforts_t


def _plot_snapshot_from_loaded(
    dense_log: Any,
    city: Any,
    firms: list,
    grid_gdf: Any,
    stores_gdf: Any,
    cfg: dict,
    t: int,
    *,
    metric: str = "expected_price",
    cmap: str = "viridis",
    basemap: Optional[Any] = None,
    ax: Optional[Any] = None,
    save_path: Optional[Path] = None,
    point_size_by_demand: bool = True,
) -> Any:
    """Internal: render a single frame from already-loaded run artefacts."""
    mpl = _require_mpl()
    plt = _require_plt()
    ctx = _require_ctx()

    tc = _get_transport_cost(cfg)
    prices_t, efforts_t = prices_efforts_at(dense_log, t)
    metric_vals = cell_metrics(
        prices_t, efforts_t, city, transport_cost=tc, metric=metric
    )
    demands_t = dense_log.demands[t].astype(np.float64)
    N_firms = len(firms)

    # --- Colormap / norm setup -------------------------------------------
    is_categorical = metric == "dominant_chain"
    if is_categorical:
        cmap_obj, norm_obj, chain_labels = _categorical_setup(firms)
    else:
        cmap_obj = plt.get_cmap(cmap)
        if metric == "expected_price":
            vmin = float(dense_log.price_grid.min())
            vmax = float(dense_log.price_grid.max())
        else:
            finite = metric_vals[np.isfinite(metric_vals)]
            vmin = float(finite.min()) if finite.size else 0.0
            vmax = float(finite.max()) if finite.size else 1.0
        norm_obj = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # --- Figure setup -------------------------------------------------------
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()
    ax.set_axis_off()

    # --- Choropleth: PolyCollection ----------------------------------------
    poly_coll = _build_poly_collection(grid_gdf, metric_vals, cmap_obj, norm_obj)
    ax.add_collection(poly_coll)

    minx, miny, maxx, maxy = grid_gdf.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")

    # --- OSM basemap (fetched once, rendered below choropleth) --------------
    if basemap is None:
        basemap = ctx.providers.OpenStreetMap.Mapnik
    ctx.add_basemap(ax, source=basemap, zoom="auto", reset_extent=False)

    # Restore limits after contextily (add_basemap can resize the axes)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # --- Store scatter ------------------------------------------------------
    sx = stores_gdf.geometry.x.values
    sy = stores_gdf.geometry.y.values

    if is_categorical:
        # Dominant-chain plot: scatter uses prices, independent colormap
        sc_cmap = plt.get_cmap(cmap)
        sc_norm = mpl.colors.Normalize(
            vmin=float(dense_log.price_grid.min()),
            vmax=float(dense_log.price_grid.max()),
        )
        sc_vals = prices_t
    else:
        # Shared continuous norm across choropleth and scatter
        sc_cmap, sc_norm, sc_vals = cmap_obj, norm_obj, prices_t

    if point_size_by_demand:
        sizes = np.sqrt(np.clip(demands_t, 0, None)) * 3.0
        sizes = np.clip(sizes, 20, 600)
    else:
        sizes = 80.0

    sc = ax.scatter(
        sx, sy,
        c=sc_vals, cmap=sc_cmap, norm=sc_norm,
        s=sizes, zorder=5, edgecolors="k", linewidths=0.4,
    )

    # --- Colorbar / legend --------------------------------------------------
    if is_categorical:
        _add_dominant_chain_legend(ax, firms, cmap_obj, norm_obj)
        # Separate colorbar for prices at stores
        cbar = plt.colorbar(sc, ax=ax, shrink=0.45, pad=0.01)
        cbar.set_label("Store price (€)", fontsize=9)
    else:
        cbar = plt.colorbar(poly_coll, ax=ax, shrink=0.6, pad=0.01)
        cbar.set_label(metric.replace("_", " ").title(), fontsize=9)

    # --- Title --------------------------------------------------------------
    run_name = Path(dense_log.run_dir).name
    ax.set_title(f"{run_name} | step {t} | {metric}", fontsize=10)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_market_snapshot(
    run_dir: Path | str,
    t: int,
    *,
    metric: str = "expected_price",
    cmap: str = "viridis",
    basemap: Optional[Any] = None,
    ax: Optional[Any] = None,
    save_path: Optional[Path] = None,
    point_size_by_demand: bool = True,
) -> Any:
    """Render a single-frame Berlin market choropleth on an OSM basemap.

    Combines a per-cell metric choropleth (filled polygon layer) with a
    supermarket scatter plot coloured by current prices, both on a contextily
    OpenStreetMap basemap.

    Parameters
    ----------
    run_dir : Path to a finished simulation run directory.
    t : Time step to visualise (0-based index into DenseLog).
    metric : Per-cell metric passed to :func:`~hotelling.core.market.cell_metrics`.
        One of ``"expected_price"``, ``"served_demand"``,
        ``"dominant_chain"``, ``"consumer_surplus"``.
    cmap : Matplotlib colormap name for continuous metrics. Default ``"viridis"``.
    basemap : contextily tile provider.  ``None`` →
        ``contextily.providers.OpenStreetMap.Mapnik``.
    ax : Existing :class:`matplotlib.axes.Axes` to draw into.  ``None`` →
        creates a new 10×10 inch figure.
    save_path : If given, save the figure to this path (PNG/PDF/SVG).
    point_size_by_demand : If ``True``, scale marker area ∝ √(demand_t) so
        high-volume stores appear larger.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    **Colour-scale sharing** — for continuous metrics (``expected_price``,
    ``served_demand``, ``consumer_surplus``) the choropleth and the store
    scatter share the *same* :class:`~matplotlib.colors.Normalize` instance
    and a single colorbar, so one scale reads both layers.  The norm range is
    taken from ``dense_log.price_grid`` for ``expected_price``; for other
    metrics it is derived from the data range at step *t*.

    **Dominant chain** — a categorical ``tab20`` colormap is used for the
    choropleth; the stores scatter is coloured independently by price and
    gets its own colorbar.  A discrete chain legend is added.

    References
    ----------
    Anderson, de Palma & Thisse (1992) Ch. 3.
    Calvano et al. (2020) AER §II.A.
    """
    dense_log, city, firms, grid_gdf, stores_gdf, cfg = load_run(run_dir)
    return _plot_snapshot_from_loaded(
        dense_log, city, firms, grid_gdf, stores_gdf, cfg, t,
        metric=metric,
        cmap=cmap,
        basemap=basemap,
        ax=ax,
        save_path=save_path,
        point_size_by_demand=point_size_by_demand,
    )


def animate_market(
    run_dir: Path | str,
    *,
    timesteps: Optional[Sequence[int]] = None,
    stride: Optional[int] = None,
    metric: str = "expected_price",
    cmap: str = "viridis",
    fps: int = 8,
    save_path: Optional[Path] = None,
) -> Path:
    """Create a FuncAnimation of the Berlin market choropleth over time.

    Fetches the OSM basemap **once** before the animation loop; per-frame
    updates only call :meth:`~matplotlib.collections.PolyCollection.set_array`
    and :meth:`~matplotlib.collections.PathCollection.set_array` /
    :meth:`~matplotlib.collections.PathCollection.set_sizes` — never
    ``add_basemap`` or ``.plot()`` — so rendering is fast.

    Parameters
    ----------
    run_dir : Path to a finished simulation run directory.
    timesteps : Explicit sequence of integer time steps to animate.
        Overrides *stride*.  ``None`` → auto-determined (see below).
    stride : Step between frames when *timesteps* is ``None``.
        ``None`` → use ``aggregate.parquet`` "step" column if present, else
        ``range(0, T, max(1, T // 60))``.
    metric : Per-cell metric; same choices as :func:`plot_market_snapshot`.
    cmap : Matplotlib colormap name for continuous metrics.
    fps : Frames per second for the saved file.
    save_path : Output path.  Suffix ``.gif`` → :class:`PillowWriter`;
        suffix ``.mp4`` → :class:`FFMpegWriter`.
        ``None`` → ``run_dir / "animation.gif"``.

    Returns
    -------
    Path to the saved animation file.

    Notes
    -----
    The axis extent is locked to the 3857 grid bounding box before basemap
    fetch and restored afterwards so contextily cannot resize the axes between
    frames.

    References
    ----------
    Anderson, de Palma & Thisse (1992) Ch. 3.
    Calvano et al. (2020) AER §II.A.
    """
    mpl = _require_mpl()
    plt = _require_plt()
    ctx = _require_ctx()

    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    from matplotlib.collections import PolyCollection

    run_dir = Path(run_dir)
    dense_log, city, firms, grid_gdf, stores_gdf, cfg = load_run(run_dir)
    tc = _get_transport_cost(cfg)
    T = dense_log._t_written
    N_firms = len(firms)
    run_name = run_dir.name

    # ── Determine frame list ────────────────────────────────────────────────
    if timesteps is not None:
        frames_list = list(timesteps)
    elif stride is not None:
        frames_list = list(range(0, T, stride))
    else:
        agg_path = run_dir / "aggregate.parquet"
        if agg_path.exists():
            import pandas as pd

            agg = pd.read_parquet(agg_path)
            if "step" in agg.columns:
                frames_list = agg["step"].dropna().astype(int).tolist()
            else:
                frames_list = list(range(0, T, max(1, T // 60)))
        else:
            frames_list = list(range(0, T, max(1, T // 60)))

    if not frames_list:
        raise ValueError(f"No frames to animate in run at {run_dir}")

    # ── Colormap / norm (computed from first frame for non-price metrics) ───
    is_categorical = metric == "dominant_chain"
    prices_0, efforts_0 = prices_efforts_at(dense_log, frames_list[0])
    metric_0 = cell_metrics(prices_0, efforts_0, city, transport_cost=tc, metric=metric)

    if is_categorical:
        cmap_obj, norm_obj, chain_labels = _categorical_setup(firms)
    else:
        cmap_obj = plt.get_cmap(cmap)
        if metric == "expected_price":
            vmin = float(dense_log.price_grid.min())
            vmax = float(dense_log.price_grid.max())
        else:
            finite = metric_0[np.isfinite(metric_0)]
            vmin = float(finite.min()) if finite.size else 0.0
            vmax = float(finite.max()) if finite.size else 1.0
        norm_obj = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Scatter uses shared price norm for price-like metrics
    if is_categorical or metric != "expected_price":
        sc_cmap = plt.get_cmap(cmap)
        sc_norm = mpl.colors.Normalize(
            vmin=float(dense_log.price_grid.min()),
            vmax=float(dense_log.price_grid.max()),
        )
    else:
        sc_cmap, sc_norm = cmap_obj, norm_obj

    # ── Build figure and PolyCollection ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()

    verts = [_geom_verts(g) for g in grid_gdf.geometry]
    poly_coll = PolyCollection(
        verts,
        array=metric_0,
        cmap=cmap_obj,
        norm=norm_obj,
        linewidths=0,
        edgecolors="none",
        alpha=0.65,
        zorder=2,
    )
    ax.add_collection(poly_coll)

    # Fix extent BEFORE basemap fetch so contextily tiles the correct area
    minx, miny, maxx, maxy = grid_gdf.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")

    # ── Fetch OSM basemap ONCE ──────────────────────────────────────────────
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik,
        zoom="auto",
        reset_extent=False,
    )

    # Restore extent (contextily may resize the axes)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # ── Store scatter (built once) ──────────────────────────────────────────
    sx = stores_gdf.geometry.x.values
    sy = stores_gdf.geometry.y.values
    demands_0 = dense_log.demands[frames_list[0]].astype(np.float64)
    init_sizes = np.clip(np.sqrt(np.clip(demands_0, 0, None)) * 3.0, 20, 600)

    sc = ax.scatter(
        sx, sy,
        c=prices_0, cmap=sc_cmap, norm=sc_norm,
        s=init_sizes, zorder=5, edgecolors="k", linewidths=0.4,
    )

    # ── Colorbar / legend ───────────────────────────────────────────────────
    if is_categorical:
        _add_dominant_chain_legend(ax, firms, cmap_obj, norm_obj)
        plt.colorbar(sc, ax=ax, shrink=0.45, pad=0.01, label="Store price (€)")
    else:
        plt.colorbar(poly_coll, ax=ax, shrink=0.6, pad=0.01,
                     label=metric.replace("_", " ").title())

    title_artist = ax.set_title(
        f"{run_name} | t = {frames_list[0]} | {metric}", fontsize=10
    )
    fig.tight_layout()

    # ── Per-frame update (never calls add_basemap or .plot()) ───────────────
    def _update(frame_idx: int):
        t = frames_list[frame_idx]
        prices_t, efforts_t = prices_efforts_at(dense_log, t)
        metric_t = cell_metrics(
            prices_t, efforts_t, city, transport_cost=tc, metric=metric
        )
        demands_t = dense_log.demands[t].astype(np.float64)

        poly_coll.set_array(metric_t)

        sc.set_array(prices_t)
        sizes_t = np.clip(np.sqrt(np.clip(demands_t, 0, None)) * 3.0, 20, 600)
        sc.set_sizes(sizes_t)

        title_artist.set_text(f"{run_name} | t = {t} | {metric}")
        return poly_coll, sc, title_artist

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frames_list),
        blit=True,
        interval=max(1, 1000 // fps),
    )

    # ── Save ────────────────────────────────────────────────────────────────
    if save_path is None:
        save_path = run_dir / "animation.gif"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = save_path.suffix.lower()
    if suffix == ".mp4":
        writer = FFMpegWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)

    anim.save(str(save_path), writer=writer)
    plt.close(fig)
    return save_path


def interactive_slider(
    run_dir: Path | str,
    *,
    metric: str = "expected_price",
    cmap: str = "viridis",
) -> None:
    """ipywidgets interactive slider for Jupyter notebooks.

    Displays an :class:`~ipywidgets.widgets.IntSlider` ranging from 0 to
    ``T - 1`` with step ``max(1, T // 200)``.  On each slider change,
    :func:`plot_market_snapshot` is called and the resulting figure is
    displayed inline.

    The run data (City, DenseLog, GDFs) is loaded **once** before the slider
    is created so that slider interactions do not trigger redundant I/O.

    Parameters
    ----------
    run_dir : Path to a finished simulation run directory.
    metric : Per-cell metric; same choices as :func:`plot_market_snapshot`.
    cmap : Matplotlib colormap name for continuous metrics.

    Raises
    ------
    ImportError
        If ipywidgets is not installed.  The error message instructs the
        user to run ``pip install 'hotelling[notebooks]'``.

    Notes
    -----
    Intended for use in Jupyter notebooks / JupyterLab.  Call inside a
    notebook cell; the slider appears inline in the cell output.
    """
    try:
        import ipywidgets as widgets
        from ipywidgets import interact
    except ImportError as exc:
        raise ImportError(
            "ipywidgets is required for the interactive slider. "
            "Install with: pip install 'hotelling[notebooks]'"
        ) from exc

    plt = _require_plt()
    ctx = _require_ctx()  # noqa: F841 — validate availability early

    run_dir = Path(run_dir)
    dense_log, city, firms, grid_gdf, stores_gdf, cfg = load_run(run_dir)
    T = dense_log._t_written
    step = max(1, T // 200)

    def _display_frame(t: int) -> None:
        fig = _plot_snapshot_from_loaded(
            dense_log, city, firms, grid_gdf, stores_gdf, cfg, t,
            metric=metric,
            cmap=cmap,
        )
        plt.show()
        plt.close(fig)

    interact(
        _display_frame,
        t=widgets.IntSlider(min=0, max=T - 1, step=step, value=0,
                            description="Step"),
    )
