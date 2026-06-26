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
# Chain-type marker registry (for scatter grouping)
# ---------------------------------------------------------------------------

#: Matplotlib marker code keyed by normalised chain-type string.
#: "discount" → ▼ downward triangle (cheap/basic tier)
#: "standard" → ● circle (mid tier, neutral)
#: "bio"      → ■ square (premium/differentiated tier)
_CHAIN_TYPE_MARKERS: dict[str, str] = {
    "discount": "v",
    "standard": "o",
    "bio":      "s",
}

#: Human-readable legend label keyed by normalised chain-type string.
_CHAIN_TYPE_LABELS: dict[str, str] = {
    "discount": "Discount (D)",
    "standard": "Standard (S)",
    "bio":      "Bio (B)",
}

#: Canonical iteration order for grouped scatter (controls legend entry order).
_CHAIN_TYPE_ORDER: list[str] = ["discount", "standard", "bio"]

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

def _get_chain_types(stores_gdf: Any, n_stores: int) -> np.ndarray:
    """Return a length-N object array of normalised chain-type strings.

    Reads the ``chain_type`` column from *stores_gdf* if present (row *j*
    must correspond to store *j* — i.e. the GDF has been
    ``reset_index(drop=True)``), normalises each value to lowercase-stripped,
    and maps anything outside ``{'discount', 'standard', 'bio'}`` to
    ``'standard'``.

    Parameters
    ----------
    stores_gdf : GeoDataFrame of stores (any CRS), row *j* == store *j*.
    n_stores : expected length N; used only for the all-standard fallback
        when ``chain_type`` column is absent.

    Returns
    -------
    np.ndarray of shape (N,) with dtype ``object``; every element is one of
    ``'discount'``, ``'standard'``, or ``'bio'``.
    """
    _known = {"discount", "standard", "bio"}
    if "chain_type" in stores_gdf.columns:
        raw = stores_gdf["chain_type"].fillna("standard").astype(str)
        normalised = raw.str.strip().str.lower().to_numpy()
        out = np.where(np.isin(normalised, list(_known)), normalised, "standard")
        return out.astype(object)
    # Fallback: column absent (should not happen for supermarkets.parquet v2+
    # which requires chain_type per loader.py).
    return np.full(n_stores, "standard", dtype=object)


def _scatter_by_chain(
    ax: Any,
    sx: np.ndarray,
    sy: np.ndarray,
    colors: np.ndarray,
    sizes: Any,
    chain_types: np.ndarray,
    cmap: Any,
    norm: Any,
) -> Tuple[list, Optional[Any]]:
    """Plot the store scatter grouped by chain type, one call per group.

    Each chain type is rendered with a distinct marker from
    ``_CHAIN_TYPE_MARKERS`` while sharing the same *cmap* / *norm* so the
    colour scale is globally consistent across all groups.  All groups use
    ``zorder=5``, black edge colour, and ``linewidths=0.4``.  The ``label=``
    attribute on each ``PathCollection`` is set so that a subsequent
    ``ax.legend(title="Chain type", ...)`` call picks them up automatically.

    Parameters
    ----------
    ax : ``matplotlib.axes.Axes`` to draw on.
    sx, sy : (N,) float64 arrays — store x/y coordinates in EPSG:3857.
    colors : (N,) float64 array — values mapped through *cmap* / *norm*;
        typically ``prices_t``.
    sizes : (N,) float64 numpy array **or** a plain Python/NumPy scalar.
        If a NumPy array, each group receives ``sizes[mask]``.
        If a scalar, that value is broadcast identically across all groups.
    chain_types : (N,) object array produced by :func:`_get_chain_types`.
    cmap : matplotlib ``Colormap`` object.
    norm : matplotlib ``Normalize`` (or ``BoundaryNorm``) object.

    Returns
    -------
    artists : list of ``matplotlib.collections.PathCollection``
        One element per non-empty chain type, in ``_CHAIN_TYPE_ORDER``
        order (discount → standard → bio).  May be an empty list if every
        mask is False (degenerate input).
    colorbar_source : first element of *artists*, or ``None`` when *artists*
        is empty.  Suitable as the first positional argument to
        ``plt.colorbar(colorbar_source, ax=ax, ...)``.
    """
    artists: list = []
    for ct in _CHAIN_TYPE_ORDER:
        mask = (chain_types == ct)
        if not mask.any():
            continue
        # Per-element sizes when sizes is a NumPy array; scalar broadcast otherwise.
        s = sizes[mask] if isinstance(sizes, np.ndarray) else sizes
        sc = ax.scatter(
            sx[mask],
            sy[mask],
            c=colors[mask],
            cmap=cmap,
            norm=norm,
            s=s,
            zorder=5,
            edgecolors="k",
            linewidths=0.4,
            marker=_CHAIN_TYPE_MARKERS.get(ct, "o"),
            label=_CHAIN_TYPE_LABELS.get(ct, ct),
        )
        artists.append(sc)
    colorbar_source: Optional[Any] = artists[0] if artists else None
    return artists, colorbar_source


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


def _build_global_norm(
    dense_log: Any,
    city: Any,
    firms: list,
    cfg: dict,
    frames: Sequence[int],
    metric: str,
    cmap: str,
) -> Tuple[Any, Any]:
    """Compute a single consistent (cmap, norm) pair spanning *all* given frames.

    For ``"expected_price"`` the norm is derived from ``dense_log.price_grid``
    and requires no frame scan.  For all other continuous metrics, every frame
    in *frames* is evaluated with :func:`~hotelling.core.market.cell_metrics`
    and the global finite min/max is used as ``vmin``/``vmax``.  For
    ``"dominant_chain"`` the :class:`~matplotlib.colors.BoundaryNorm` on
    ``N_firms`` bins is returned unchanged — it is already frame-independent.

    Parameters
    ----------
    dense_log : DenseLog instance.
    city : City instance.
    firms : list of Firm objects.
    cfg : run config dict (used to extract ``transport_cost``).
    frames : sequence of time-step indices to scan.
    metric : one of the four metric strings.
    cmap : matplotlib colormap name.

    Returns
    -------
    cmap_obj : matplotlib Colormap
    norm_obj : matplotlib Normalize (or BoundaryNorm for dominant_chain)
    """
    mpl = _require_mpl()
    plt = _require_plt()

    if metric == "dominant_chain":
        cmap_obj, norm_obj, _ = _categorical_setup(firms)
        return cmap_obj, norm_obj

    cmap_obj = plt.get_cmap(cmap)
    tc = _get_transport_cost(cfg)

    if metric == "expected_price":
        # Price-grid bounds are frame-independent by construction.
        vmin = float(dense_log.price_grid.min())
        vmax = float(dense_log.price_grid.max())
    else:
        # Scan every frame in the given sequence to find the global finite range.
        g_min, g_max = np.inf, -np.inf
        for t in frames:
            prices_t, efforts_t = prices_efforts_at(dense_log, t)
            m_t = cell_metrics(
                prices_t, efforts_t, city, transport_cost=tc, metric=metric
            )
            finite = m_t[np.isfinite(m_t)]
            if finite.size:
                g_min = min(g_min, float(finite.min()))
                g_max = max(g_max, float(finite.max()))
        vmin = g_min if np.isfinite(g_min) else 0.0
        vmax = g_max if np.isfinite(g_max) else 1.0

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
        dense_distances=bool(env_cfg.get("dense_distances", False)),
        catchment_minutes=(
            float(env_cfg["catchment_minutes"])
            if "catchment_minutes" in env_cfg
            else None
        ),
    )

    # Load DenseLog
    from hotelling.simulation.dense_log import DenseLog

    dense_log = DenseLog.load(run_dir)
    # Bind city so that dense_log.to_dataframe() auto-reconstructs demands/profits
    # on lean runs (store_demand_profit=False). No-op on non-lean runs.
    dense_log.attach_city(
        city,
        transport_cost=float(env_cfg.get("transport_cost", 0.01)),
    )

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

    Chain-specific grids
    --------------------
    When the run used ``--chs-grid``, ``dense_log.store_price_grids`` is an
    ``(N, m)`` array giving each store its own price grid, and ``price_idx[t]``
    must be decoded per store: ``prices_t[j] = store_price_grids[j, price_idx[t, j]]``.
    For global-grid runs ``store_price_grids`` is ``None`` and the shared
    ``price_grid`` is used (unchanged behaviour). Effort grids are never
    chain-specific, so efforts always decode through ``effort_grid``.
    """
    pidx = dense_log.price_idx[t].astype(np.int64)
    eidx = dense_log.effort_idx[t].astype(np.int64)

    store_price_grids = getattr(dense_log, "store_price_grids", None)
    if store_price_grids is not None:
        # Per-store decode: row j of store_price_grids holds store j's grid.
        n_stores = pidx.shape[0]
        prices_t = store_price_grids[np.arange(n_stores), pidx].astype(np.float64)
    else:
        # Global grid shared by all stores (unchanged path).
        prices_t = dense_log.price_grid[pidx].astype(np.float64)

    efforts_t = dense_log.effort_grid[eidx].astype(np.float64)
    return prices_t, efforts_t


def _demands_at(
    dense_log: Any,
    t: int,
    city: Any,
    transport_cost: float,
    firm_arrays: Any = None,
) -> Optional[np.ndarray]:
    """Return (N,) float64 demands at DenseLog row *t*; ``None`` if unrecoverable.

    Three cases:

    1. ``dense_log.demands`` is not ``None`` (normal non-lean run):
       return the stored float array directly.
    2. ``dense_log.demands`` is ``None`` AND *city* is provided (lean run):
       reconstruct via :func:`~hotelling.core.market.market_clearing_arrays`
       using decoded prices/efforts from the index arrays at row *t*.
    3. ``dense_log.demands`` is ``None`` AND *city* is ``None``:
       return ``None``; the caller should fall back to uniform scatter sizes.

    Parameters
    ----------
    dense_log : DenseLog instance.
    t : DenseLog row index (0-based, not absolute simulation step).
    city : City — required for reconstruction in case 2; may be ``None``.
    transport_cost : float — disutility coefficient used during the run.
    firm_arrays : FirmArrays | None — pre-built per-firm attribute struct.
        When ``None`` and reconstruction is needed, built automatically from
        ``city.firms``.  Pass a precomputed instance to avoid re-building on
        every call (important for animation loops).

    Returns
    -------
    (N,) float64 array or ``None``.
    """
    if dense_log.demands is not None:
        return dense_log.demands[t].astype(np.float64)
    if city is None:
        return None
    from hotelling.core.market import market_clearing_arrays, precompute_firm_arrays
    fa = firm_arrays if firm_arrays is not None else precompute_firm_arrays(city.firms)
    prices_t, efforts_t = prices_efforts_at(dense_log, t)
    d, _ = market_clearing_arrays(
        prices_t.astype(np.float64),
        efforts_t.astype(np.float64),
        city,
        float(transport_cost),
        fa,
    )
    return d.astype(np.float64)


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
    norm: Optional[Any] = None,
) -> Any:
    """Internal: render a single frame from already-loaded run artefacts.

    Parameters
    ----------
    norm : optional pre-built matplotlib Normalize (or BoundaryNorm).
        When supplied — e.g. from :func:`_build_global_norm` — this exact
        norm is used for the choropleth and (for ``expected_price``) the
        scatter, so every call shares the same colour scale.  When ``None``
        the norm is derived from the current frame's data range, which is
        appropriate for one-off snapshots but inconsistent across frames.
    """
    mpl = _require_mpl()
    plt = _require_plt()
    ctx = _require_ctx()

    tc = _get_transport_cost(cfg)
    prices_t, efforts_t = prices_efforts_at(dense_log, t)
    metric_vals = cell_metrics(
        prices_t, efforts_t, city, transport_cost=tc, metric=metric
    )
    # demands_t is fetched lazily below, only when point_size_by_demand=True

    # --- Colormap / norm setup -------------------------------------------
    is_categorical = metric == "dominant_chain"
    if is_categorical:
        cmap_obj, norm_obj, chain_labels = _categorical_setup(firms)
    elif norm is not None:
        # Use caller-supplied norm (global range for consistent scales).
        cmap_obj = plt.get_cmap(cmap)
        norm_obj = norm
    else:
        # Per-frame fallback: suitable for standalone one-off snapshots.
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
    # The scatter always shows store *prices* coloured on the price-grid range
    # so the store dots are frame-consistent regardless of the choropleth metric.
    # Exception: expected_price shares the choropleth norm (both are price-valued).
    # Stores are grouped by chain type (discount/standard/bio) and rendered with
    # distinct marker shapes: ▼ (v), ● (o), ■ (s) respectively.
    sx = stores_gdf.geometry.x.values
    sy = stores_gdf.geometry.y.values

    _price_norm = mpl.colors.Normalize(
        vmin=float(dense_log.price_grid.min()),
        vmax=float(dense_log.price_grid.max()),
    )
    if is_categorical:
        sc_cmap = plt.get_cmap(cmap)
        sc_norm = _price_norm
    elif metric == "expected_price":
        # Shared norm: choropleth and scatter both represent price values.
        sc_cmap, sc_norm = cmap_obj, norm_obj
    else:
        # Scatter is independently coloured by price on the fixed price-grid range.
        sc_cmap, sc_norm = cmap_obj, _price_norm

    if point_size_by_demand:
        _demands_t = _demands_at(dense_log, t, city, tc)
        if _demands_t is not None:
            sizes = np.clip(np.sqrt(np.clip(_demands_t, 0, None)) * 3.0, 20, 600)
        else:
            sizes = np.full(len(firms), 80.0)
    else:
        sizes = 80.0

    # Extract chain type per store (N,) → 'discount' | 'standard' | 'bio'.
    chain_types = _get_chain_types(stores_gdf, len(firms))

    # One scatter call per chain type; distinct marker per group.
    scatter_artists, colorbar_sc = _scatter_by_chain(
        ax, sx, sy, prices_t, sizes, chain_types, sc_cmap, sc_norm
    )

    # --- Colorbar / legend --------------------------------------------------
    # Chain-type marker legend (always present regardless of metric).
    # Uses the label= attributes set in _scatter_by_chain.
    if scatter_artists:
        chain_type_leg = ax.legend(
            title="Chain type",
            loc="lower right",
            fontsize=8,
            title_fontsize=9,
            framealpha=0.8,
        )
        if is_categorical:
            # For dominant_chain we also need the brand-level legend at upper
            # right. Preserve chain_type_leg via add_artist BEFORE the second
            # ax.legend() call (which would otherwise replace it).
            ax.add_artist(chain_type_leg)

    if is_categorical:
        # Brand-level categorical legend at upper right (calls ax.legend internally).
        _add_dominant_chain_legend(ax, firms, cmap_obj, norm_obj)
        # Colorbar for the store-price scatter (independent of the choropleth).
        if colorbar_sc is not None:
            cbar = plt.colorbar(colorbar_sc, ax=ax, shrink=0.45, pad=0.01)
            cbar.set_label("Store price (€)", fontsize=9)
    elif metric == "expected_price":
        # Single shared colorbar covers both choropleth and scatter.
        cbar = plt.colorbar(poly_coll, ax=ax, shrink=0.6, pad=0.01)
        cbar.set_label(metric.replace("_", " ").title(), fontsize=9)
    else:
        # Two colorbars: one for the choropleth metric, one for store prices.
        cbar = plt.colorbar(poly_coll, ax=ax, shrink=0.6, pad=0.01)
        cbar.set_label(metric.replace("_", " ").title(), fontsize=9)
        if colorbar_sc is not None:
            cbar2 = plt.colorbar(colorbar_sc, ax=ax, shrink=0.35, pad=0.06)
            cbar2.set_label("Store price (€)", fontsize=8)

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
    **Colour scale** — for ``expected_price`` the choropleth and scatter share
    a single :class:`~matplotlib.colors.Normalize` fixed to
    ``dense_log.price_grid`` bounds (frame-independent).  For other continuous
    metrics the choropleth norm is derived from the data at step *t* (suitable
    for standalone snapshots; pass a pre-built *norm* via
    :func:`_plot_snapshot_from_loaded` for cross-frame consistency).  The
    store scatter always uses the price-grid range regardless of metric.

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
    **Consistent colour scale** — the choropleth norm is computed by scanning
    *all* frames in the animation sequence before any frame is rendered, so
    the same ``vmin``/``vmax`` applies throughout.  For ``expected_price`` the
    norm is always ``dense_log.price_grid`` bounds; for other metrics the
    global finite min/max across all frames is used.  The store-scatter dots
    always use the fixed price-grid range independent of the choropleth metric.

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

    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, ImageMagickWriter
    from matplotlib.collections import PolyCollection

    run_dir = Path(run_dir)
    dense_log, city, firms, grid_gdf, stores_gdf, cfg = load_run(run_dir)
    tc = _get_transport_cost(cfg)
    T = dense_log._rows_written
    N_firms = len(firms)
    run_name = run_dir.name

    # Extract chain type per store once — static for the whole animation.
    # Values are 'discount' | 'standard' | 'bio' in canonical store order 0..N-1.
    _anim_chain_types = _get_chain_types(stores_gdf, N_firms)

    # ── Determine frame list ────────────────────────────────────────────────
    # DenseLog rows are indexed 0..T-1 (T = dense_log._rows_written), and
    # prices_efforts_at / dense_log.demands index by ROW, not by absolute
    # simulation step. aggregate.parquet["step"] lives in the recorder's
    # absolute-step space (e.g. a single value == T_game for short runs) and
    # must NOT be used as a DenseLog row index — doing so raises IndexError
    # whenever a step >= rows_written. Default to the dense-log row space.
    if timesteps is not None:
        frames_list = [int(t) for t in timesteps]
    elif stride is not None:
        frames_list = list(range(0, T, max(1, int(stride))))
    else:
        frames_list = list(range(0, T, max(1, T // 60)))

    # Clamp to valid DenseLog rows; dedupe and keep ascending order.
    frames_list = sorted({t for t in frames_list if 0 <= t < T})
    if not frames_list:
        raise ValueError(
            f"No valid frames to animate in run at {run_dir} (rows_written={T})."
        )

    # ── Consistent colormap / norm across ALL animation frames ──────────────
    # _build_global_norm scans every frame in frames_list so the colour scale
    # is fixed for the entire animation (not just the first frame).
    is_categorical = metric == "dominant_chain"
    cmap_obj, norm_obj = _build_global_norm(
        dense_log, city, firms, cfg, frames_list, metric, cmap
    )
    if is_categorical:
        _, _, chain_labels = _categorical_setup(firms)

    # Initial frame data (needed to seed the artists before the loop)
    prices_0, efforts_0 = prices_efforts_at(dense_log, frames_list[0])
    metric_0 = cell_metrics(prices_0, efforts_0, city, transport_cost=tc, metric=metric)

    # Scatter always shows store prices on the fixed price-grid range.
    # Exception: expected_price shares the choropleth norm (both price-valued).
    _price_norm = mpl.colors.Normalize(
        vmin=float(dense_log.price_grid.min()),
        vmax=float(dense_log.price_grid.max()),
    )
    if is_categorical or metric != "expected_price":
        sc_cmap = plt.get_cmap(cmap)
        sc_norm = _price_norm
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
        alpha=0.5,
    )

    # Restore extent (contextily may resize the axes)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # ── Store scatter (built once, grouped by chain type) ───────────────────
    # Each chain type gets its own PathCollection with a distinct marker shape.
    # All groups share sc_cmap / sc_norm so the colour (= price) scale is uniform.
    # _chain_sc_list holds (mask, PathCollection) pairs for the _update closure.
    sx = stores_gdf.geometry.x.values
    sy = stores_gdf.geometry.y.values
    # Precompute FirmArrays once for lean-mode reconstruction (avoids per-frame rebuild).
    _anim_firm_arrays = None
    if dense_log.demands is None:
        from hotelling.core.market import precompute_firm_arrays
        _anim_firm_arrays = precompute_firm_arrays(city.firms)

    _demands_0 = _demands_at(dense_log, frames_list[0], city, tc, _anim_firm_arrays)
    init_sizes = (
        np.clip(np.sqrt(np.clip(_demands_0, 0, None)) * 3.0, 20, 600)
        if _demands_0 is not None
        else np.full(N_firms, 80.0)
    )

    _chain_sc_list: list = []   # list of (boolean mask, PathCollection) tuples
    for _ct in _CHAIN_TYPE_ORDER:
        _mask = (_anim_chain_types == _ct)
        if not _mask.any():
            continue
        _sc_ct = ax.scatter(
            sx[_mask],
            sy[_mask],
            c=prices_0[_mask],
            cmap=sc_cmap,
            norm=sc_norm,
            s=init_sizes[_mask],
            zorder=5,
            edgecolors="k",
            linewidths=0.4,
            marker=_CHAIN_TYPE_MARKERS.get(_ct, "o"),
            label=_CHAIN_TYPE_LABELS.get(_ct, _ct),
        )
        _chain_sc_list.append((_mask, _sc_ct))

    # Reference to the first non-empty scatter artist, used for the colorbar.
    _sc_for_cbar = _chain_sc_list[0][1] if _chain_sc_list else None

    # ── Colorbar / legend ───────────────────────────────────────────────────
    # Chain-type marker legend (always present regardless of metric).
    if _chain_sc_list:
        _chain_type_leg = ax.legend(
            title="Chain type",
            loc="lower right",
            fontsize=8,
            title_fontsize=9,
            framealpha=0.8,
        )
        if is_categorical:
            # Preserve chain_type_leg before the brand-level legend call
            # replaces ax._legend via its own ax.legend() invocation.
            ax.add_artist(_chain_type_leg)

    if is_categorical:
        _add_dominant_chain_legend(ax, firms, cmap_obj, norm_obj)
        if _sc_for_cbar is not None:
            plt.colorbar(_sc_for_cbar, ax=ax, shrink=0.45, pad=0.01,
                         label="Store price (€)")
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

        poly_coll.set_array(metric_t)

        # Update each per-chain-type scatter independently.
        _demands_t = _demands_at(dense_log, t, city, tc, _anim_firm_arrays)
        sizes_t = (
            np.clip(np.sqrt(np.clip(_demands_t, 0, None)) * 3.0, 20, 600)
            if _demands_t is not None
            else np.full(N_firms, 80.0)
        )
        for _mask, _sc_ct in _chain_sc_list:
            _sc_ct.set_array(prices_t[_mask])
            _sc_ct.set_sizes(sizes_t[_mask])

        title_artist.set_text(f"{run_name} | t = {t} | {metric}")
        # blit=True requires returning ALL artists that changed this frame.
        return (poly_coll, *[_sc_ct for _, _sc_ct in _chain_sc_list], title_artist)

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
    A single consistent colour norm is also pre-computed by scanning the
    sampled slider steps (``range(0, T, max(1, T // 200))``), so the colour
    scale remains identical across all slider positions.

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
    T = dense_log._rows_written
    step = max(1, T // 200)

    # ── Pre-compute a single consistent norm across all slider steps ─────────
    # This ensures the colour scale is identical for every slider position,
    # making visual comparison between time steps meaningful.
    _slider_frames = list(range(0, T, step))
    _, _global_norm = _build_global_norm(
        dense_log, city, firms, cfg, _slider_frames, metric, cmap
    )

    def _display_frame(t: int) -> None:
        fig = _plot_snapshot_from_loaded(
            dense_log, city, firms, grid_gdf, stores_gdf, cfg, t,
            metric=metric,
            cmap=cmap,
            norm=_global_norm,
        )
        plt.show()
        plt.close(fig)

    interact(
        _display_frame,
        t=widgets.IntSlider(min=0, max=T - 1, step=step, value=0,
                            description="Step"),
    )
