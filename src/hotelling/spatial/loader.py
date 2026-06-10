"""Berlin spatial data → simulation City/Firm loader.

Bridges the GEO data pipeline outputs (demand_grid.parquet,
supermarkets.parquet, travel_times.parquet) to the City and Firm dataclasses
consumed by the simulation engine.

Public API
----------
build_catchment         Build sparse per-cell store catchment (CSR arrays).
chain_type_to_quality   Map chain type string to quality intercept float.
load_berlin_city        Load full City + Firm list from parquet files.

Notes
-----
City.dist2_km2 stores travel-time minutes (not km²). The field name is
inherited from the City dataclass but in the Berlin model the distance proxy
is transit travel time. transport_cost (€/min) is calibrated accordingly.

The canonical cell order is sorted ascending by GITTER_ID_100m string.
The canonical store order is sorted ascending by store integer index (0..N-1).
Both orderings are enforced by sort operations so that dist2_km2[i, j] is the
travel time from cell i to store j, matching cell_pop[i] and Firm at index j.

Two distance representations are supported (selected by dense_distances kwarg):

Dense path (dense_distances=True, default for inner-ring):
    _build_dist_matrix pivots travel_times to a dense (M, N) array.
    Consumed by the existing numba logit kernels and equilibrium solvers.

Sparse catchment path (dense_distances=False, required for full Berlin):
    build_catchment produces three CSR arrays without ever materialising the
    dense matrix.  Consumed by the catchment-aware demand kernels (Prompt 4).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from hotelling.core.city import City
from hotelling.core.firm import Firm

logger = logging.getLogger(__name__)

__all__ = [
    "build_catchment",
    "chain_type_to_quality",
    "load_berlin_city",
    "populate_catchment_precompute",
]

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def chain_type_to_quality(chain_type: str, q_S: float, q_B: float) -> float:
    """Map chain type string to quality intercept.

    Quality intercepts: q_D = 0.0 (normalised baseline),
    q_S = standard tier, q_B = bio/premium tier.

    Parameters
    ----------
    chain_type : "discount", "standard", or "bio"
    q_S : quality intercept for standard chains (q_D = 0)
    q_B : quality intercept for bio/premium chains

    Returns
    -------
    float quality intercept; 0.0 for unknown chain type
    """
    mapping = {"discount": 0.0, "standard": q_S, "bio": q_B}
    return mapping.get(str(chain_type).strip().lower(), 0.0)


def chain_type_to_marginal_cost(
    chain_type: str,
    c_D: float = 0.0,
    c_S: float = 0.0,
    c_B: float = 0.0,
) -> float:
    """Map chain type string to marginal cost.

    ADR-014 baseline sets all costs to 0. Override via c_D, c_S, c_B.
    """
    mapping = {"discount": c_D, "standard": c_S, "bio": c_B}
    return mapping.get(str(chain_type).strip().lower(), 0.0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_phi_i(df: pd.DataFrame, w_hub: float = 0.4, w_prime: float = 0.3,
                   w_lcl: float = 0.3) -> pd.Series:
    """Compute phi_i footfall index from constituent columns.

    phi_i = w_hub * phi_hub + w_prime * phi_prime + w_lcl * phi_lcl

    phi_hub   = station_class_normalized (NaN → 0)
    phi_prime = has_cluster cast to float (0 or 1)
    phi_lcl   = has_mall cast to float (0 or 1)

    Parameters
    ----------
    df : GeoDataFrame or DataFrame containing constituent columns
    w_hub, w_prime, w_lcl : weights (must sum to 1.0)
    """
    phi_hub = pd.Series(0.0, index=df.index, dtype=float)
    if "station_class_normalized" in df.columns:
        phi_hub = df["station_class_normalized"].fillna(0.0).astype(float)

    phi_prime = pd.Series(0.0, index=df.index, dtype=float)
    if "has_cluster" in df.columns:
        phi_prime = df["has_cluster"].fillna(False).astype(float)
    elif "cluster_id" in df.columns:
        phi_prime = df["cluster_id"].notna().astype(float)

    phi_lcl = pd.Series(0.0, index=df.index, dtype=float)
    if "has_mall" in df.columns:
        phi_lcl = df["has_mall"].fillna(False).astype(float)

    return (w_hub * phi_hub + w_prime * phi_prime + w_lcl * phi_lcl).clip(0.0, 1.0)


def _compute_pi_H(df: pd.DataFrame) -> pd.Series:
    """Compute pi_H_res residential H-type consumer share from social indices.

    Uses esix_normalized and si_normalized (both in [0,1]).
    Both present → arithmetic mean. Only one → use that. Neither → 0.5.
    NaN cells fall back to the other column, then to 0.5.
    """
    has_esix = "esix_normalized" in df.columns
    has_si   = "si_normalized"   in df.columns

    if has_esix and has_si:
        esix = df["esix_normalized"]
        si   = df["si_normalized"]
        both = esix.notna() & si.notna()
        only_esix = esix.notna() & si.isna()
        only_si   = si.notna()   & esix.isna()
        result = pd.Series(0.5, index=df.index, dtype=float)
        result[both]      = (esix[both] + si[both]) / 2.0
        result[only_esix] = esix[only_esix]
        result[only_si]   = si[only_si]
    elif has_esix:
        result = df["esix_normalized"].fillna(0.5).astype(float)
    elif has_si:
        result = df["si_normalized"].fillna(0.5).astype(float)
    else:
        result = pd.Series(0.5, index=df.index, dtype=float)

    return result.clip(0.0, 1.0)


def _build_dist_matrix(
    cell_ids: list[str],
    store_ids: list[str],
    tt_df: pd.DataFrame,
    nan_fill: float,
) -> np.ndarray:
    """Build the (M, N) travel-time distance matrix.

    Parameters
    ----------
    cell_ids : list of M canonical INSPIRE cell ID strings (row order)
    store_ids : list of N store integer-index strings, e.g. ["0","1",...] (col order)
    tt_df : DataFrame with columns from_id (str), to_id (str), travel_time (float)
    nan_fill : value used for (cell, store) pairs with no travel time data

    Returns
    -------
    np.ndarray shape (M, N) float64; rows = cells, cols = stores
    """
    # Pivot to wide format: rows = from_id, cols = to_id
    pivot = tt_df.pivot_table(
        index="from_id",
        columns="to_id",
        values="travel_time",
        aggfunc="first",  # take first if duplicates
    )
    # Reindex to canonical cell and store order; missing entries become NaN
    pivot = pivot.reindex(index=cell_ids, columns=store_ids)
    mat = pivot.to_numpy(dtype=np.float64, na_value=np.nan)
    # Fill NaN with nan_fill (unreachable pairs get large travel-time penalty)
    np.nan_to_num(mat, copy=False, nan=nan_fill)
    return mat


# ---------------------------------------------------------------------------
# Travel-time alignment check
# ---------------------------------------------------------------------------

def _check_travel_time_alignment(
    tt_df: pd.DataFrame,
    cell_ids: list[str],
    store_ids: list[str],
) -> None:
    """Validate that travel_times to_id / from_id are consistent with the
    loaded stores and demand grid.

    Raises
    ------
    ValueError
        If to_id values reference stores outside {0..N-1}, indicating a
        mismatch between the travel_times file and the supermarkets file
        (e.g. inner-ring tt against full-stores parquet or vice versa).

    Logs a warning if many cells have no travel-time rows at all.
    """
    store_ids_set = set(store_ids)
    bad_to = set(tt_df["to_id"].unique()) - store_ids_set
    if bad_to:
        sample = sorted(bad_to)[:8]
        raise ValueError(
            f"travel_times.parquet references {len(bad_to)} store IDs absent from "
            f"supermarkets.parquet (N={len(store_ids)}). "
            f"Sample bad IDs: {sample}. "
            "This usually means mismatched inner-ring vs. full-grid parquet files. "
            "Re-run the GEO pipeline with matching scope."
        )

    cell_ids_set = set(cell_ids)
    missing_cells = cell_ids_set - set(tt_df["from_id"].unique())
    if missing_cells:
        pct = 100.0 * len(missing_cells) / len(cell_ids)
        if pct > 20.0:
            logger.warning(
                "%d / %d cells (%.0f%%) have zero travel-time rows in "
                "travel_times.parquet. Verify that demand_grid and travel_times "
                "come from the same GEO pipeline run and cover the same area.",
                len(missing_cells), len(cell_ids), pct,
            )
        else:
            logger.info(
                "%d / %d cells (%.1f%%) have no travel-time entry "
                "(expected for boundary cells or cells outside the routing network).",
                len(missing_cells), len(cell_ids), pct,
            )


# ---------------------------------------------------------------------------
# Sparse catchment CSR builder
# ---------------------------------------------------------------------------

def build_catchment(
    tt_df: pd.DataFrame,
    cell_ids: list[str],
    store_ids: list[str],
    transport_cost: float,       # forwarded to Prompt-4 kernel; not used here
    transport_exponent: float,   # forwarded to Prompt-4 kernel; not used here
    catchment_minutes: float,
    k_min: int,
    k_max: int,
    nan_fill_minutes: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a sparse per-cell store catchment without materialising the dense matrix.

    For each cell in canonical order (``cell_ids``), selects the subset of
    stores that are within ``catchment_minutes`` transit-time, guarantees at
    least ``k_min`` stores (padding with the nearest beyond the radius if
    needed), and caps at ``k_max`` stores.  Cells that have no travel-time
    rows at all receive an empty span (``indptr[i] == indptr[i+1]``).

    The result is the standard CSR (Compressed Sparse Row) representation:
    the catchment for cell ``i`` is
    ``indices[indptr[i]:indptr[i+1]]`` and
    ``tt_min[indptr[i]:indptr[i+1]]``.

    This function operates entirely on the *long-format* ``tt_df`` table
    (columns ``from_id``, ``to_id``, ``travel_time``); it never pivots to a
    dense (M × N) matrix.

    Parameters
    ----------
    tt_df : long-format travel-time table; columns from_id (str), to_id (str),
        travel_time (float, minutes).
    cell_ids : M canonical cell IDs in canonical row order (sorted GITTER_ID_100m).
    store_ids : N store IDs in canonical column order ("0".."N-1").
    transport_cost : forwarded for forward-compatibility with the Prompt-4
        catchment kernel that computes per-entry disutility; unused here.
    transport_exponent : forwarded for same reason; unused here.
    catchment_minutes : radius threshold (transit-time minutes); stores with
        travel_time > catchment_minutes are excluded unless needed for k_min.
    k_min : minimum number of stores per cell regardless of the radius.
    k_max : hard cap on stores per cell (memory bound).
    nan_fill_minutes : travel-time sentinels at or above this value are treated
        as unreachable and excluded before the catchment selection.

    Returns
    -------
    indptr : ndarray of shape (M+1,), dtype int64
        CSR row-pointer array.  ``indptr[0] == 0``, ``indptr[M] == NNZ``.
    indices : ndarray of shape (NNZ,), dtype int32
        Store column indices for each non-zero catchment entry.
    tt_min : ndarray of shape (NNZ,), dtype float64
        Travel times (minutes) in the same order as ``indices``.

    Notes
    -----
    The ``(transport_cost, transport_exponent)`` parameters are present in the
    signature for forward-compatibility: the Prompt-4 catchment kernel will
    convert these raw travel-time minutes into per-entry utility costs at
    kernel call time.  They are not used during catchment construction.
    """
    if k_min > k_max:
        raise ValueError(f"k_min={k_min} must be <= k_max={k_max}")

    M = len(cell_ids)
    store_col: dict[str, int] = {s: i for i, s in enumerate(store_ids)}
    cell_set = set(cell_ids)

    # Filter to valid, finite pairs — never include nan_fill sentinels
    valid_mask = (
        tt_df["from_id"].isin(cell_set)
        & tt_df["to_id"].isin(store_col)
        & tt_df["travel_time"].notna()
        & (tt_df["travel_time"] < nan_fill_minutes)
    )
    filt = tt_df.loc[valid_mask, ["from_id", "to_id", "travel_time"]].copy()
    filt["_col"] = filt["to_id"].map(store_col).astype(np.int32)

    # Map from_id → canonical row index; sort by (row, travel_time) for a
    # single-pass CSR build without per-cell Python overhead.
    cell_order: dict[str, int] = {c: i for i, c in enumerate(cell_ids)}
    filt["_row"] = filt["from_id"].map(cell_order).astype(np.int32)
    filt.sort_values(["_row", "travel_time"], inplace=True, ignore_index=True)

    rows_arr = filt["_row"].to_numpy(dtype=np.int32)
    cols_arr = filt["_col"].to_numpy(dtype=np.int32)
    tt_arr   = filt["travel_time"].to_numpy(dtype=np.float64)

    # Locate per-cell slice boundaries using binary search on the sorted rows array
    row_range = np.arange(M, dtype=np.int32)
    cell_starts = np.searchsorted(rows_arr, row_range,   side="left")
    cell_ends   = np.searchsorted(rows_arr, row_range,   side="right")

    # Build CSR output arrays
    indptr = np.empty(M + 1, dtype=np.int64)
    indptr[0] = 0

    indices_parts: list[np.ndarray] = []
    tt_parts:      list[np.ndarray] = []
    n_empty = 0
    catchment_sizes: list[int] = []

    for i in range(M):
        s, e = int(cell_starts[i]), int(cell_ends[i])
        if s == e:                              # cell has no tt rows at all
            n_empty += 1
            indptr[i + 1] = indptr[i]
            continue

        # arr_tt is already sorted ascending (rows sorted by travel_time above)
        # Find how many stores are within the catchment radius
        n_within = int(np.searchsorted(tt_arr[s:e], catchment_minutes, side="right"))

        if n_within >= k_min:
            # Take all within-radius stores, capped at k_max
            n_keep = min(n_within, k_max)
        else:
            # Pad with nearest-beyond-radius up to k_min, still capped at k_max
            n_keep = min(max(k_min, 1), e - s, k_max)

        sl = slice(s, s + n_keep)
        indices_parts.append(cols_arr[sl])
        tt_parts.append(tt_arr[sl])
        indptr[i + 1] = indptr[i] + n_keep
        catchment_sizes.append(n_keep)

    indices = (
        np.concatenate(indices_parts).astype(np.int32)
        if indices_parts
        else np.empty(0, dtype=np.int32)
    )
    tt_min = (
        np.concatenate(tt_parts).astype(np.float64)
        if tt_parts
        else np.empty(0, dtype=np.float64)
    )

    nnz = int(indptr[M])
    cs = np.array(catchment_sizes, dtype=np.float64) if catchment_sizes else np.array([0.0])
    logger.info(
        "Catchment CSR: M=%d cells, NNZ=%d, "
        "mean=%.1f, median=%.1f stores/cell, empty_cells=%d.",
        M, nnz,
        float(cs.mean()),
        float(np.median(cs)),
        n_empty,
    )

    return indptr, indices, tt_min


# ---------------------------------------------------------------------------
# Catchment kernel precompute
# ---------------------------------------------------------------------------

def populate_catchment_precompute(
    city: City,
    transport_cost: float,
    precompute_expweights: bool = False,
    low_precision_storage: bool = False,
) -> None:
    """Compute and store period-invariant catchment-kernel inputs on *city* in-place.

    Must be called **after** ``city.catch_indptr`` / ``catch_indices`` /
    ``catch_tt`` are populated (i.e. after ``build_catchment`` and ``City``
    construction).  Idempotent: safe to call again to rebuild when
    ``transport_cost`` changes.

    Populates
    ---------
    city.catch_C                : (NNZ,) float64 — ``-transport_cost * tt**exponent``
    city.catch_C_transport_cost : float — transport_cost baked into catch_C
    city.A_quality              : (2, N) float64 — ``alpha_h * quality_j``
    city.w_H, city.w_L          : (M,) float64 — combined consumer weights
    city.precompute_expweights  : bool — True if Kexp arrays were built
    city.catch_Kexp_L/H         : (NNZ,) float64 or float32 — exp-weight arrays
        (only when ``precompute_expweights=True`` and the guard passes)

    Parameters
    ----------
    city : City with catch_indptr, catch_indices, catch_tt already set.
    transport_cost : the tc value to bake into catch_C.  Must match the value
        that will be passed to ``market_clearing_arrays`` and the equilibrium
        solvers; mismatches trigger an inline recompute in those callers.
    precompute_expweights : if True, attempt to build Kexp arrays for the
        max-perf kernel.  Falls back to stable kernel with a warning if the
        max exponent ≥ 700 (numerical overflow guard).
    low_precision_storage : store Kexp arrays as float32 (halves memory;
        acceptable since they are multiplied by per-period float64 weights).

    Notes
    -----
    **Calibration guidance for transport_cost** — pass the same value used in
    :func:`load_berlin_city` (and stored in the env YAML as ``transport_cost``).
    The per-entry disutility is then
    ``|catch_C[p]| = transport_cost * travel_time[p]**exponent`` in €/min.
    """
    assert city.catch_indptr is not None, (
        "populate_catchment_precompute: city.catch_indptr is None — "
        "build_catchment must run first."
    )
    assert city.catch_indices is not None
    assert city.catch_tt is not None

    te = float(getattr(city, "transport_exponent", 1.0))
    tt = city.catch_tt.astype(np.float64, copy=False)

    # catch_C[p] = -transport_cost * tt[p]**te
    if te == 1.0:
        catch_C = np.ascontiguousarray(-transport_cost * tt, dtype=np.float64)
    else:
        catch_C = np.ascontiguousarray(-transport_cost * tt**te, dtype=np.float64)

    # A_quality[h, j] = alpha_h * quality_j
    qualities = np.array([f.quality for f in city.firms], dtype=np.float64)
    alpha_L = float(city.alpha[0])
    alpha_H = float(city.alpha[1])
    A_quality = np.ascontiguousarray(
        np.stack([alpha_L * qualities, alpha_H * qualities], axis=0),
        dtype=np.float64,
    )  # shape (2, N), C-contiguous

    # w_H[i], w_L[i] combined consumer-mass weights
    w_H = np.ascontiguousarray(
        city.cell_pop * city.pi_H
        + city.lambda_phi * city.pi_H_lambda_phi,
        dtype=np.float64,
    )
    w_L = np.ascontiguousarray(
        city.cell_pop * (1.0 - city.pi_H)
        + city.lambda_phi * (1.0 - city.pi_H_lambda_phi),
        dtype=np.float64,
    )

    city.catch_C = catch_C
    city.catch_C_transport_cost = float(transport_cost)
    city.A_quality = A_quality
    city.w_H = w_H
    city.w_L = w_L
    # Reset expweights; re-enabled below if guard passes
    city.precompute_expweights = False
    city.low_precision_storage = low_precision_storage
    city.catch_Kexp_L = None
    city.catch_Kexp_H = None

    nnz = int(city.catch_indptr[-1])
    N = len(city.firms)

    if precompute_expweights and nnz > 0:
        inv_mu = 1.0 / float(city.mu)
        idx = city.catch_indices.astype(np.int64)   # (NNZ,) for advanced indexing

        # a_L/H[p] = A_quality[h, indices[p]] + catch_C[p]
        a_L_plus_C = A_quality[0, idx] + catch_C   # (NNZ,) float64
        a_H_plus_C = A_quality[1, idx] + catch_C   # (NNZ,) float64

        max_exp_L = float((a_L_plus_C * inv_mu).max())
        max_exp_H = float((a_H_plus_C * inv_mu).max())

        if max_exp_L >= 700.0 or max_exp_H >= 700.0:
            logger.warning(
                "populate_catchment_precompute: max exponent %.1f >= 700; "
                "expweights path disabled — falling back to stable log-sum-exp kernel.",
                max(max_exp_L, max_exp_H),
            )
        else:
            kexp_dtype = np.float32 if low_precision_storage else np.float64
            city.catch_Kexp_L = np.ascontiguousarray(
                np.exp(a_L_plus_C * inv_mu), dtype=kexp_dtype
            )
            city.catch_Kexp_H = np.ascontiguousarray(
                np.exp(a_H_plus_C * inv_mu), dtype=kexp_dtype
            )
            city.precompute_expweights = True

    logger.info(
        "Catchment precompute: NNZ=%d, N=%d, expweights=%s, low_prec=%s.",
        nnz, N, city.precompute_expweights, low_precision_storage,
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_berlin_city(
    grid_path: Path | str = "data/processed/demand_grid.parquet",
    stores_path: Path | str = "data/processed/supermarkets.parquet",
    travel_times_path: Path | str = "data/processed/travel_times.parquet",
    *,
    lambda_val: float,
    q_S: float = 0.8,
    q_B: float = 1.5,
    alpha_L: float = 0.5,
    alpha_H: float = 1.5,
    beta_effort: float = 0.001,
    kappa0: float = 1.0,
    store_size: float = 600.0,
    transport_cost: float = 0.01,
    a0: float = -1.0,
    mu: float = 0.25,
    nan_fill_minutes: float = 120.0,
    marginal_cost_D: float = 0.0,
    marginal_cost_S: float = 0.0,
    marginal_cost_B: float = 0.0,
    phi_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    rent_scale: float = 0.0,
    rent_normalization: str = "mean_ratio",
    dense_distances: bool = False,
    catchment_minutes: float | None = None,
    catchment_k_min: int = 12,
    catchment_k_max: int = 80,
    precompute_expweights: bool = False,
    low_precision_storage: bool = False,
) -> tuple[City, list[Firm]]:
    """Load the Berlin inner-Ringbahn simulation environment from parquet files.

    Reads demand_grid.parquet (cells), supermarkets.parquet (stores), and
    travel_times.parquet (transit travel times), aligns their indices, and
    constructs a fully-populated City object together with the list of Firm
    objects.

    The canonical cell order is determined by sorting GITTER_ID_100m
    ascending. The canonical store order is store integer index 0..N-1
    ascending. Both orderings are enforced so that dist2_km2[i, j] is the
    travel-time proxy from cell i to store j, and city.firms[j] is the j-th
    store.

    City.dist2_km2 holds travel-time minutes (not km²). The field name is
    a misnomer inherited from the dataclass; transport_cost is calibrated in
    €/min accordingly.

    Parameters
    ----------
    grid_path : path to demand_grid.parquet (from build_demand_grid)
    stores_path : path to supermarkets.parquet (from process_supermarkets)
    travel_times_path : path to travel_times.parquet (from build_transit_travel_times)
    lambda_val : footfall bonus scalar (calibrate via calibrate_lambda)
    q_S : quality intercept for standard chains (q_D = 0 normalised)
    q_B : quality intercept for bio/premium chains
    alpha_L : low-type consumer marginal utility of chain quality
    alpha_H : high-type consumer marginal utility of chain quality
    beta_effort : homogeneous marginal utility of store effort (enters demand)
    kappa0 : quadratic effort cost coefficient (same for all stores, ADR-017)
    store_size : store floor-space in m² used with rent (ADR-015 baseline: rent=0)
    transport_cost : transport disutility parameter in €/min (since dist
        proxy = travel time in minutes). Default 0.01.
    a0 : outside option utility intercept (≤ 0)
    mu : logit scale parameter
    nan_fill_minutes : travel-time fill value for unreachable (cell, store)
        pairs. Default 120 min (2× the 60-min cap used in GEO_05).
    marginal_cost_D/S/B : per-unit marginal cost by chain type.
        ADR-014 baseline: all 0.0.
    phi_weights : (w_hub, w_prime, w_lcl) — weights for phi_i components.
        Must sum to 1.0. Default (0.4, 0.3, 0.3).
    rent_scale : master on/off switch and overall scale for the BRW-derived
        fixed cost (dimensionless, in the same units as per-period profit).
        0.0 (default) disables fixed costs entirely, preserving baseline
        behaviour (ADR-015).  A positive value enables heterogeneous
        fixed costs derived from the "brw" column in supermarkets.parquet.
        Calibration guidance: pick rent_scale as a target fraction of mean
        per-period GROSS margin at Bertrand-Nash prices, e.g. 0.05–0.15.
        Run scripts/run_baseline.py first to read the printed mean gross
        margin, then set rent_scale accordingly.  Do NOT auto-calibrate.
    rent_normalization : normalisation method for brw → fixed_cost mapping.
        "mean_ratio"   (default): fixed_cost_j = rent_scale * brw_j / mean(brw)
            → mean fixed cost equals rent_scale exactly.
        "median_ratio" (robust to outliers): fixed_cost_j = rent_scale * brw_j / median(brw)
        "minmax"       : fixed_cost_j = rent_scale * (brw_j - min) / (max - min)
            → range [0, rent_scale].
        Relative land-cost ordering across stores is preserved by all methods.
        See ADR-022 for the economic rationale and inertness-for-pricing caveat.
    dense_distances : if True, build the full (M, N) dense travel-time matrix
        (required for the existing logit kernels and equilibrium solvers).
        Default False — skip the dense pivot to save memory on large grids.
        Add ``dense_distances: true`` to the inner-ring env config to maintain
        backward-compatible behaviour.
    catchment_minutes : transit-time radius for the sparse catchment (minutes).
        Stores beyond this radius are excluded per cell unless needed for
        catchment_k_min.  None (default) disables catchment building.
    catchment_k_min : minimum stores per cell regardless of radius (default 12).
    catchment_k_max : hard cap on stores per cell (default 80, memory bound).

    Returns
    -------
    tuple[City, list[Firm]]
        City object with dist2_km2 set (dense path) or None (sparse path), and
        CSR catch_* fields set (sparse path) or None (dense path).

    Raises
    ------
    FileNotFoundError : if any required parquet file is absent
    KeyError : if required columns are missing from a parquet file
    ValueError : if rent_normalization is not one of the three recognised
        strings, or if travel_times references store IDs outside {0..N-1}.
    """
    grid_path  = Path(grid_path)
    stores_path = Path(stores_path)
    tt_path    = Path(travel_times_path)

    for p in (grid_path, stores_path, tt_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required parquet not found: {p}. "
                "Run the GEO pipeline notebooks first."
            )

    # ── Load grid ──────────────────────────────────────────────────────────
    logger.info("Loading demand grid from %s.", grid_path)
    grid_raw = gpd.read_parquet(grid_path)

    # Require GITTER_ID_100m for alignment
    if "GITTER_ID_100m" not in grid_raw.columns:
        raise KeyError(
            "demand_grid.parquet is missing 'GITTER_ID_100m'. "
            "Re-run build_demand_grid() to regenerate."
        )
    if "Einwohner" not in grid_raw.columns:
        raise KeyError("demand_grid.parquet is missing 'Einwohner' (residential population).")

    # Sort by GITTER_ID_100m to establish canonical cell order (reproducible)
    grid = grid_raw.sort_values("GITTER_ID_100m").reset_index(drop=True)

    # Safeguard: if demand_grid.parquet was generated before the
    # build_demand_grid() deduplication fix was applied, it will contain
    # ~32k rows (2× the expected 16k). Detect and fix this transparently.
    n_dupes = grid["GITTER_ID_100m"].duplicated().sum()
    if n_dupes > 0:
        logger.warning(
            "demand_grid.parquet contains %d duplicate GITTER_ID_100m rows "
            "(%d total rows, %d unique cells). "
            "This is caused by a now-fixed bug in build_demand_grid() where "
            "sjoin(predicate='intersects') on LOR polygons doubled boundary cells. "
            "Deduplicating automatically — re-run GEO_07_demand.ipynb to fix "
            "the source parquet.",
            n_dupes, len(grid), grid["GITTER_ID_100m"].nunique(),
        )
        grid = grid.drop_duplicates(
            subset="GITTER_ID_100m", keep="first"
        ).reset_index(drop=True)

    cell_ids: list[str] = grid["GITTER_ID_100m"].tolist()
    M = len(grid)
    logger.info("Grid: %d cells (canonical order: sorted by GITTER_ID_100m).", M)

    # ── Compute phi_i ──────────────────────────────────────────────────────
    if "phi_i" in grid.columns:
        phi_i = grid["phi_i"].fillna(0.0).values.astype(np.float64)
        logger.info("phi_i loaded directly from demand_grid.parquet.")
    else:
        w_hub, w_prime, w_lcl = phi_weights
        phi_i_s = _compute_phi_i(grid, w_hub=w_hub, w_prime=w_prime, w_lcl=w_lcl)
        phi_i = phi_i_s.values.astype(np.float64)
        logger.info(
            "phi_i computed from constituent columns (mean=%.4f, max=%.4f).",
            phi_i.mean(), phi_i.max(),
        )

    # ── Compute pi_H_res ──────────────────────────────────────────────────
    if "pi_H_res" in grid.columns:
        pi_H = grid["pi_H_res"].fillna(0.5).clip(0.0, 1.0).values.astype(np.float64)
        logger.info("pi_H_res loaded directly from demand_grid.parquet.")
    else:
        pi_H = _compute_pi_H(grid).values.astype(np.float64)
        logger.info("pi_H computed from esix_normalized/si_normalized (mean=%.4f).", pi_H.mean())

    cell_pop   = grid["Einwohner"].fillna(0).values.astype(np.float64)
    lambda_phi = lambda_val * phi_i
    # Transient consumers use same type share as residential per spec §G.1
    pi_H_lambda_phi = pi_H.copy()

    # ── Load stores ───────────────────────────────────────────────────────
    logger.info("Loading supermarkets from %s.", stores_path)
    stores_raw = gpd.read_parquet(stores_path)

    for col in ("geometry", "chain", "chain_type"):
        if col not in stores_raw.columns:
            raise KeyError(
                f"supermarkets.parquet is missing column '{col}'. "
                "Re-run process_supermarkets() to regenerate."
            )

    # Sort by integer index (reset_index ensures 0..N-1 order)
    stores = stores_raw.reset_index(drop=True)
    N = len(stores)
    store_ids: list[str] = [str(i) for i in range(N)]
    logger.info("Stores: %d supermarkets (canonical order: index 0..%d).", N, N - 1)

    # ── BRW-derived fixed costs ────────────────────────────────────────────
    # Per-store fixed_cost is a size-independent lump sum per period derived
    # from the Bodenrichtwert (land value €/m²) of each store location.
    # As an additive constant in profit it does NOT enter the price FOC:
    # Bertrand-Nash / joint-monopoly prices, the Q-table price grid, and
    # Calvano Δ are all invariant to it.  It shifts absolute profit levels
    # and is behaviourally relevant only at the entry/exit margin (Phase 1+).
    # See ADR-022.  The rent*size channel stays inert (rent=0.0, ADR-015).
    fixed_costs_arr: np.ndarray
    if rent_scale == 0.0 or "brw" not in stores.columns:
        if rent_scale != 0.0:
            logger.info(
                "rent_scale=%.4f requested but 'brw' column absent from "
                "supermarkets.parquet — setting fixed_cost=0.0 for all stores.",
                rent_scale,
            )
        else:
            logger.info(
                "rent_scale=0.0: fixed_cost disabled for all stores (ADR-015 baseline)."
            )
        fixed_costs_arr = np.zeros(N, dtype=np.float64)
    else:
        brw_raw = stores["brw"].copy().astype(float)
        invalid_mask = brw_raw.isna() | (brw_raw <= 0.0)
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            brw_median = float(brw_raw[~invalid_mask].median())
            brw_raw[invalid_mask] = brw_median
            logger.info(
                "BRW: filled %d invalid/NaN entries with median brw=%.1f.",
                n_invalid,
                brw_median,
            )
        brw = brw_raw.values.astype(np.float64)

        if rent_normalization == "mean_ratio":
            ref = brw.mean()
            normalised = brw / ref
        elif rent_normalization == "median_ratio":
            ref = float(np.median(brw))
            normalised = brw / ref
        elif rent_normalization == "minmax":
            lo, hi = brw.min(), brw.max()
            span = hi - lo
            normalised = (brw - lo) / span if span > 0.0 else np.zeros_like(brw)
        else:
            raise ValueError(
                f"Unknown rent_normalization={rent_normalization!r}. "
                "Choose from: mean_ratio, median_ratio, minmax"
            )

        fixed_costs_arr = rent_scale * normalised
        logger.info(
            "BRW fixed costs: rent_scale=%.4f, normalization=%r, "
            "mean=%.4f, min=%.4f, max=%.4f.",
            rent_scale,
            rent_normalization,
            float(fixed_costs_arr.mean()),
            float(fixed_costs_arr.min()),
            float(fixed_costs_arr.max()),
        )

    # Build Firm objects in canonical order
    firms: list[Firm] = []
    for i, row in stores.iterrows():
        ct = str(row["chain_type"]) if pd.notna(row.get("chain_type")) else "standard"
        quality = chain_type_to_quality(ct, q_S=q_S, q_B=q_B)
        mc      = chain_type_to_marginal_cost(
            ct, c_D=marginal_cost_D, c_S=marginal_cost_S, c_B=marginal_cost_B
        )
        location = (float(row.geometry.x), float(row.geometry.y))
        firm = Firm(
            id=str(i),
            location=location,
            marginal_cost=mc,
            quality=quality,
            kappa0=kappa0,
            size=store_size,
            rent=0.0,  # ADR-015 baseline; rent*size channel stays inert
            fixed_cost=float(fixed_costs_arr[i]),
            chain=str(row["chain"]) if pd.notna(row.get("chain")) else None,
            chain_type=ct,
        )
        firms.append(firm)
    logger.info("Built %d Firm objects.", len(firms))

    # ── Load travel times ────────────────────────────────────────────────
    logger.info("Loading travel times from %s.", tt_path)
    tt_raw = pd.read_parquet(tt_path)

    # Ensure consistent string types for join keys
    tt_raw = tt_raw.copy()
    tt_raw["from_id"] = tt_raw["from_id"].astype(str)
    tt_raw["to_id"]   = tt_raw["to_id"].astype(str)

    # ── Alignment check ───────────────────────────────────────────────────
    # Raises ValueError if to_id values fall outside {0..N-1}, which indicates
    # mismatched travel_times vs. supermarkets files (e.g. inner-ring tt vs.
    # full-grid stores or vice versa).
    _check_travel_time_alignment(tt_raw, cell_ids=cell_ids, store_ids=store_ids)

    # ── Dense distance matrix (inner-ring / small-grid path) ─────────────
    dist_matrix: np.ndarray | None = None
    if dense_distances:
        dist_matrix = _build_dist_matrix(
            cell_ids=cell_ids,
            store_ids=store_ids,
            tt_df=tt_raw,
            nan_fill=nan_fill_minutes,
        )  # shape (M, N)
        n_missing = int(np.sum(dist_matrix >= nan_fill_minutes))
        logger.info(
            "Dense distance matrix: shape %s, %d entries filled with "
            "nan_fill=%.1f min.",
            dist_matrix.shape, n_missing, nan_fill_minutes,
        )
    else:
        logger.info(
            "dense_distances=False: skipping dense (M×N) pivot "
            "(M=%d, N=%d → %.1f M entries saved).",
            M, N, M * N / 1e6,
        )

    # ── Sparse catchment CSR (full-grid / large-grid path) ────────────────
    catch_indptr:  np.ndarray | None = None
    catch_indices: np.ndarray | None = None
    catch_tt:      np.ndarray | None = None
    if catchment_minutes is not None:
        catch_indptr, catch_indices, catch_tt = build_catchment(
            tt_df=tt_raw,
            cell_ids=cell_ids,
            store_ids=store_ids,
            transport_cost=transport_cost,
            transport_exponent=1.0,  # ADR-020 linear; City.transport_exponent mirrors this
            catchment_minutes=catchment_minutes,
            k_min=catchment_k_min,
            k_max=catchment_k_max,
            nan_fill_minutes=nan_fill_minutes,
        )

    # ── City boundary ─────────────────────────────────────────────────────
    bounds = grid.total_bounds  # (minx, miny, maxx, maxy) in EPSG:3035
    boundary = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

    # ── Assemble City ────────────────────────────────────────────────────
    city = City(
        boundary=boundary,
        population_grid=None,          # not needed; cell_pop array used directly
        firms=firms,
        dist2_km2=dist_matrix,         # (M, N) or None on sparse path
        cell_pop=cell_pop,             # (M,) residential population
        lambda_phi=lambda_phi,         # (M,) λ * phi_i footfall addition
        pi_H=pi_H,                     # (M,) H-type residential share
        pi_H_lambda_phi=pi_H_lambda_phi,  # (M,) H-type transient share
        alpha=np.array([alpha_L, alpha_H], dtype=np.float64),
        beta=beta_effort,
        mu=mu,
        a0=a0,
        catch_indptr=catch_indptr,
        catch_indices=catch_indices,
        catch_tt=catch_tt,
    )

    logger.info(
        "City loaded: %d cells, %d stores, boundary=(%.0f,%.0f,%.0f,%.0f). "
        "dense=%s, catchment=%s.",
        M, N, *boundary,
        dense_distances,
        f"{catchment_minutes}min" if catchment_minutes is not None else "None",
    )

    # ── Catchment kernel precompute ──────────────────────────────────────
    if catch_indptr is not None:
        populate_catchment_precompute(
            city,
            transport_cost=transport_cost,
            precompute_expweights=precompute_expweights,
            low_precision_storage=low_precision_storage,
        )

    # ── Numba JIT warm-up ────────────────────────────────────────────────
    # Trigger JIT compilation so the first simulation step is not penalised.
    from hotelling.core.market import logit_demand, catchment_demand

    n_warm = min(2, N)
    m_warm = min(2, M)
    if n_warm > 0 and m_warm > 0:
        if catch_indptr is not None and city.catch_C is not None:
            # Warm up the catchment kernel with a tiny synthetic city
            catchment_demand(city=city, prices=np.zeros(N), efforts=np.zeros(N))
        elif dense_distances and dist_matrix is not None:
            _warm_dist = dist_matrix[:m_warm, :n_warm]
            logit_demand(
                prices=np.zeros(n_warm, dtype=np.float64),
                efforts=np.zeros(n_warm, dtype=np.float64),
                dist2_km2=_warm_dist,
                cell_pop=cell_pop[:m_warm],
                lambda_phi=lambda_phi[:m_warm],
                pi_H=pi_H[:m_warm],
                pi_H_lambda_phi=pi_H_lambda_phi[:m_warm],
                alpha=city.alpha,
                quality=np.array([f.quality for f in firms[:n_warm]], dtype=np.float64),
                beta=beta_effort,
                transport_cost=transport_cost,
                mu=mu,
                a0=a0,
            )
        else:
            # No dense dist and no catchment: tiny synthetic warm-up
            _warm_dist = np.ones((m_warm, n_warm), dtype=np.float64)
            logit_demand(
                prices=np.zeros(n_warm, dtype=np.float64),
                efforts=np.zeros(n_warm, dtype=np.float64),
                dist2_km2=_warm_dist,
                cell_pop=cell_pop[:m_warm],
                lambda_phi=lambda_phi[:m_warm],
                pi_H=pi_H[:m_warm],
                pi_H_lambda_phi=pi_H_lambda_phi[:m_warm],
                alpha=city.alpha,
                quality=np.array([f.quality for f in firms[:n_warm]], dtype=np.float64),
                beta=beta_effort,
                transport_cost=transport_cost,
                mu=mu,
                a0=a0,
            )

    return city, firms
