"""Competitor-set construction and price-signal utilities for state-space designs.

Builds demand-overlap competitor adjacency (CSR) with integer overlap weights,
separates same-chain-type from cross-chain-type competitors, and provides
Numba-accelerated weighted-mean + closest-bin matching.

Public API:
    build_competitor_sets — construct CSR competitor graphs
    weighted_mean_prices — compute demand-overlap-weighted competitor mean prices
    bin_prices — map continuous prices to discrete bin indices
    find_k_nearest_same_type — find k nearest same-type rivals per store
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numba as nb
import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


@dataclass
class CompetitorSets:
    """Container for precomputed competitor CSR graphs and bin infrastructure.

    All CSR arrays use int64 indices and float64 weights (overlap counts cast to float
    for direct use in weighted-mean computation without per-call dtype conversion).
    """
    # Same-chain-type competitors (demand-overlap weighted)
    same_indptr: np.ndarray    # (N+1,) int64
    same_indices: np.ndarray   # (nnz_same,) int64
    same_weights: np.ndarray   # (nnz_same,) float64 — overlap cell counts

    # Cross-chain-type competitors (demand-overlap weighted)
    cross_indptr: np.ndarray   # (N+1,) int64
    cross_indices: np.ndarray  # (nnz_cross,) int64
    cross_weights: np.ndarray  # (nnz_cross,) float64

    # All-type competitors (union)
    all_indptr: np.ndarray     # (N+1,) int64
    all_indices: np.ndarray    # (nnz_all,) int64
    all_weights: np.ndarray    # (nnz_all,) float64

    # Bin edges for state discretisation
    bin_edges_same: np.ndarray   # (n_bins+1,) float64
    bin_edges_cross: np.ndarray  # (n_bins+1,) float64
    bin_edges_all: np.ndarray    # (n_bins+1,) float64
    n_bins: int

    # Nearest same-type rivals (for calvano_local mode)
    # Shape (N, k_max) int64; padded with -1 for stores with < k_max same-type rivals
    nearest_same_type: np.ndarray | None = None


def build_competitor_sets(
    firms: list,
    catch_indptr: np.ndarray | None,
    catch_indices: np.ndarray | None,
    n_bins: int = 15,
    price_lo: float = 0.0,
    price_hi: float = 2.0,
    chain_type_price_ranges: dict[str, tuple[float, float]] | None = None,
    min_overlap_cells: int = 1,
    k_max_rivals: int = 3,
    fallback_n_euclidean: int = 10,
    firm_locations: np.ndarray | None = None,
) -> CompetitorSets:
    """Build demand-overlap weighted competitor CSR graphs.

    Parameters
    ----------
    firms : list of Firm objects
    catch_indptr, catch_indices : CSR arrays from city (demand-cell → store mapping)
    n_bins : number of price bins for state discretisation
    price_lo, price_hi : global price range for bin edges
    chain_type_price_ranges : optional per-chain-type (lo, hi) for bin edges
    min_overlap_cells : minimum shared demand cells to count as competitor (≥1)
    k_max_rivals : maximum k for nearest same-type rival lookup
    fallback_n_euclidean : if no catch_indptr, use N nearest Euclidean neighbors
    firm_locations : (N, 2) float64 array for Euclidean fallback

    Returns
    -------
    CompetitorSets dataclass
    """
    N = len(firms)
    chain_types = np.array([getattr(f, "chain_type", "standard") for f in firms], dtype=object)

    # --- Step 1: Build overlap count matrix (N, N) ---
    if catch_indptr is not None and catch_indices is not None:
        overlap = _build_overlap_matrix(N, catch_indptr, catch_indices)
    else:
        logger.warning(
            "No catchment CSR available; falling back to Euclidean %d-nearest.",
            fallback_n_euclidean,
        )
        if firm_locations is None:
            firm_locations = np.array([f.location for f in firms], dtype=np.float64)
        overlap = _euclidean_fallback(N, firm_locations, fallback_n_euclidean)

    # Apply minimum overlap threshold
    if min_overlap_cells > 1:
        overlap[overlap < min_overlap_cells] = 0

    # Zero diagonal (a store is never its own competitor)
    np.fill_diagonal(overlap, 0)

    # --- Step 2: Split into same-type and cross-type ---
    same_type_mask = chain_types[:, None] == chain_types[None, :]
    cross_type_mask = ~same_type_mask

    same_overlap = overlap * same_type_mask
    cross_overlap = overlap * cross_type_mask

    same_csr = csr_matrix(same_overlap.astype(np.float64))
    cross_csr = csr_matrix(cross_overlap.astype(np.float64))
    all_csr = csr_matrix(overlap.astype(np.float64))

    same_deg = np.diff(same_csr.indptr)
    cross_deg = np.diff(cross_csr.indptr)
    logger.info(
        "Competitor sets: same-type mean=%.1f/store (range %d-%d), "
        "cross-type mean=%.1f/store (range %d-%d), "
        "min_overlap_cells=%d",
        same_deg.mean(), same_deg.min(), same_deg.max(),
        cross_deg.mean(), cross_deg.min(), cross_deg.max(),
        min_overlap_cells,
    )

    # --- Step 3: Bin edges ---
    bin_edges_all = np.linspace(price_lo, price_hi, n_bins + 1)
    bin_edges_same = np.linspace(price_lo, price_hi, n_bins + 1)
    bin_edges_cross = np.linspace(price_lo, price_hi, n_bins + 1)

    if chain_type_price_ranges:
        all_lo = min(r[0] for r in chain_type_price_ranges.values())
        all_hi = max(r[1] for r in chain_type_price_ranges.values())
        bin_edges_same = np.linspace(all_lo, all_hi, n_bins + 1)
        bin_edges_cross = np.linspace(all_lo, all_hi, n_bins + 1)
        bin_edges_all = np.linspace(all_lo, all_hi, n_bins + 1)

    # --- Step 4: k nearest same-type rivals ---
    nearest_same = _find_k_nearest_same_type(same_overlap, chain_types, k_max_rivals)

    return CompetitorSets(
        same_indptr=same_csr.indptr.astype(np.int64),
        same_indices=same_csr.indices.astype(np.int64),
        same_weights=same_csr.data.astype(np.float64),
        cross_indptr=cross_csr.indptr.astype(np.int64),
        cross_indices=cross_csr.indices.astype(np.int64),
        cross_weights=cross_csr.data.astype(np.float64),
        all_indptr=all_csr.indptr.astype(np.int64),
        all_indices=all_csr.indices.astype(np.int64),
        all_weights=all_csr.data.astype(np.float64),
        bin_edges_same=bin_edges_same,
        bin_edges_cross=bin_edges_cross,
        bin_edges_all=bin_edges_all,
        n_bins=n_bins,
        nearest_same_type=nearest_same,
    )


def _build_overlap_matrix(
    N: int,
    catch_indptr: np.ndarray,
    catch_indices: np.ndarray,
) -> np.ndarray:
    """Build (N, N) int32 matrix counting shared demand cells between stores."""
    overlap = np.zeros((N, N), dtype=np.int32)
    M = len(catch_indptr) - 1
    for i in range(M):
        start = int(catch_indptr[i])
        end = int(catch_indptr[i + 1])
        stores = catch_indices[start:end]
        for ai in range(len(stores)):
            a = int(stores[ai])
            for bi in range(ai + 1, len(stores)):
                b = int(stores[bi])
                overlap[a, b] += 1
                overlap[b, a] += 1
    return overlap


def _euclidean_fallback(
    N: int,
    locations: np.ndarray,
    n_nearest: int,
) -> np.ndarray:
    """Boolean overlap proxy: each store's n_nearest Euclidean neighbors get weight 1."""
    from scipy.spatial import cKDTree

    overlap = np.zeros((N, N), dtype=np.int32)
    if N <= 1:
        return overlap
    tree = cKDTree(locations)
    k_query = min(n_nearest + 1, N)
    _, idx = tree.query(locations, k=k_query)
    if k_query == 1:
        idx = idx.reshape(N, 1)
    for j in range(N):
        for ni in idx[j]:
            ni = int(ni)
            if ni != j:
                overlap[j, ni] = 1
                overlap[ni, j] = 1
    return overlap


def _find_k_nearest_same_type(
    same_overlap: np.ndarray,
    chain_types: np.ndarray,
    k_max: int,
) -> np.ndarray:
    """For each store, find up to k_max same-type rivals ranked by overlap count.

    Returns (N, k_max) int64 array; padded with -1 for stores with fewer rivals.
    """
    N = len(chain_types)
    result = np.full((N, k_max), -1, dtype=np.int64)
    for j in range(N):
        row = same_overlap[j].copy().astype(np.float64)
        row[j] = 0
        nonzero = np.nonzero(row)[0]
        if len(nonzero) == 0:
            continue
        order = np.argsort(-row[nonzero])
        k_actual = min(k_max, len(order))
        result[j, :k_actual] = nonzero[order[:k_actual]]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Numba-accelerated computation used every simulation step
# ──────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def weighted_mean_prices(
    prices: np.ndarray,    # (N,) float64 — current actual prices
    indptr: np.ndarray,    # (N+1,) int64 — CSR row pointers
    indices: np.ndarray,   # (nnz,) int64 — CSR column indices
    weights: np.ndarray,   # (nnz,) float64 — overlap counts
) -> np.ndarray:
    """Compute demand-overlap-weighted mean competitor price for each store.

    For store j: p_comp_j = Σ(w_ji * p_i) / Σ(w_ji)  over i ∈ competitors(j)
    If store j has no competitors, returns its own price (self-referential fallback).

    Returns (N,) float64.
    """
    N = indptr.shape[0] - 1
    result = np.empty(N, dtype=np.float64)
    for j in range(N):
        wsum = 0.0
        psum = 0.0
        for k in range(indptr[j], indptr[j + 1]):
            i = indices[k]
            w = weights[k]
            psum += w * prices[i]
            wsum += w
        if wsum > 0.0:
            result[j] = psum / wsum
        else:
            result[j] = prices[j]
    return result


@nb.njit(cache=True)
def bin_prices(
    values: np.ndarray,  # (N,) float64 — values to bin
    edges: np.ndarray,   # (n_bins+1,) float64 — bin edges
    n_bins: int,
) -> np.ndarray:
    """Map continuous values to discrete bin indices in [0, n_bins-1].

    Uses linear search (n_bins is small, typically ≤15).
    Equivalent to np.clip(np.searchsorted(edges, values, side='right') - 1, 0, n_bins-1)
    but works inside Numba nopython mode.

    Returns (N,) int64.
    """
    N = values.shape[0]
    result = np.empty(N, dtype=np.int64)
    for j in range(N):
        v = values[j]
        b = 0
        for k in range(1, n_bins + 1):
            if v >= edges[k]:
                b = k
            else:
                break
        if b >= n_bins:
            b = n_bins - 1
        result[j] = b
    return result


@nb.njit(cache=True)
def rival_price_indices(
    price_idxs: np.ndarray,      # (N,) int64 — current price indices
    nearest_rivals: np.ndarray,  # (N, k) int64 — rival store indices (-1 = padding)
    m: int,                      # number of price bins for midpoint padding
) -> np.ndarray:
    """Extract rivals' price indices for the Calvano local state.

    Returns (N, k) int64 where padded positions (rival == -1) get the midpoint index m//2.
    """
    N, k = nearest_rivals.shape
    result = np.empty((N, k), dtype=np.int64)
    mid = m // 2
    for j in range(N):
        for r in range(k):
            rival = nearest_rivals[j, r]
            if rival < 0:
                result[j, r] = mid
            else:
                result[j, r] = price_idxs[rival]
    return result
