"""Berlin spatial data → simulation City/Firm loader.

Bridges the GEO data pipeline outputs (demand_grid.parquet,
supermarkets.parquet, travel_times.parquet) to the City and Firm dataclasses
consumed by the simulation engine.

Public API
----------
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
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from hotelling.core.city import City
from hotelling.core.firm import Firm

logger = logging.getLogger(__name__)

__all__ = ["chain_type_to_quality", "load_berlin_city"]

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

    Returns
    -------
    tuple[City, list[Firm]]
        City object with all precomputed arrays, and the list of N Firm
        objects in the same order as City.dist2_km2 columns.

    Raises
    ------
    FileNotFoundError : if any required parquet file is absent
    KeyError : if required columns are missing from a parquet file
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
            rent=0.0,  # ADR-015 baseline
            chain=str(row["chain"]) if pd.notna(row.get("chain")) else None,
        )
        firms.append(firm)
    logger.info("Built %d Firm objects.", len(firms))

    # ── Load and build distance matrix ───────────────────────────────────
    logger.info("Loading travel times from %s.", tt_path)
    tt_raw = pd.read_parquet(tt_path)

    # Ensure consistent string types for join keys
    tt_raw = tt_raw.copy()
    tt_raw["from_id"] = tt_raw["from_id"].astype(str)
    tt_raw["to_id"]   = tt_raw["to_id"].astype(str)

    dist_matrix = _build_dist_matrix(
        cell_ids=cell_ids,
        store_ids=store_ids,
        tt_df=tt_raw,
        nan_fill=nan_fill_minutes,
    )  # shape (M, N)

    n_missing = np.sum(dist_matrix >= nan_fill_minutes)
    logger.info(
        "Distance matrix: shape %s, %d entries filled with nan_fill (%.1f min).",
        dist_matrix.shape, int(n_missing), nan_fill_minutes,
    )

    # ── City boundary ─────────────────────────────────────────────────────
    bounds = grid.total_bounds  # (minx, miny, maxx, maxy) in EPSG:3035
    boundary = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

    # ── Assemble City ────────────────────────────────────────────────────
    city = City(
        boundary=boundary,
        population_grid=None,  # not needed; cell_pop array used directly
        firms=firms,
        dist2_km2=dist_matrix,         # (M, N) travel-time minutes
        cell_pop=cell_pop,             # (M,) residential population
        lambda_phi=lambda_phi,         # (M,) λ * phi_i footfall addition
        pi_H=pi_H,                     # (M,) H-type residential share
        pi_H_lambda_phi=pi_H_lambda_phi,  # (M,) H-type transient share
        alpha=np.array([alpha_L, alpha_H], dtype=np.float64),
        beta=beta_effort,
        mu=mu,
        a0=a0,
    )

    logger.info(
        "City loaded: %d cells, %d stores, boundary=(%.0f,%.0f,%.0f,%.0f).",
        M, N, *boundary,
    )
    return city, firms
