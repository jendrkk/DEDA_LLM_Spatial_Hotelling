"""Integration test: load_berlin_city on the full Berlin grid.

Skipped automatically when the full-grid GEO pipeline outputs are absent
(expected in CI and on developer machines without the full data download).

Run manually after producing:
    data/processed/demand_grid_full.parquet
    data/processed/supermarkets_full.parquet
    data/processed/travel_times_full.parquet

These are produced by the full-Berlin GEO pipeline notebooks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Resolve paths relative to repo root (tests may run from any cwd)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRID_PATH  = _REPO_ROOT / "data" / "processed" / "demand_grid_full.parquet"
_STORES_PATH = _REPO_ROOT / "data" / "processed" / "supermarkets_full.parquet"
_TT_PATH    = _REPO_ROOT / "data" / "processed" / "travel_times_full.parquet"

_FULL_DATA_AVAILABLE = (
    _GRID_PATH.exists() and _STORES_PATH.exists() and _TT_PATH.exists()
)

pytestmark = pytest.mark.skipif(
    not _FULL_DATA_AVAILABLE,
    reason=(
        "Full Berlin grid data absent "
        f"(checked: {_GRID_PATH}, {_STORES_PATH}, {_TT_PATH}). "
        "Run the full-Berlin GEO pipeline and re-run this test."
    ),
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_grid_city_firms():
    """Load the full Berlin City + Firms with catchment CSR once per module."""
    from hotelling.spatial.loader import load_berlin_city

    city, firms = load_berlin_city(
        grid_path=str(_GRID_PATH),
        stores_path=str(_STORES_PATH),
        travel_times_path=str(_TT_PATH),
        lambda_val=1500.0,         # placeholder; sufficient for structural tests
        transport_cost=0.5,
        dense_distances=False,     # full-grid path: no dense matrix
        catchment_minutes=25.0,
        catchment_k_min=12,
        catchment_k_max=80,
    )
    return city, firms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullGridCatchmentCSR:
    """Structural invariants on the sparse catchment returned by load_berlin_city."""

    def test_catch_indptr_not_none(self, full_grid_city_firms):
        city, _ = full_grid_city_firms
        assert city.catch_indptr is not None, "catch_indptr should be set on sparse path"

    def test_catch_indices_not_none(self, full_grid_city_firms):
        city, _ = full_grid_city_firms
        assert city.catch_indices is not None

    def test_catch_tt_not_none(self, full_grid_city_firms):
        city, _ = full_grid_city_firms
        assert city.catch_tt is not None

    def test_dist2_km2_is_none(self, full_grid_city_firms):
        """Sparse path must NOT materialise the dense matrix."""
        city, _ = full_grid_city_firms
        assert city.dist2_km2 is None, (
            "dist2_km2 should be None on the sparse path (dense_distances=False)"
        )

    def test_indptr_length_is_M_plus_1(self, full_grid_city_firms):
        city, _ = full_grid_city_firms
        M = len(city.cell_pop)
        assert len(city.catch_indptr) == M + 1, (
            f"catch_indptr length should be M+1={M+1}, "
            f"got {len(city.catch_indptr)}"
        )

    def test_indptr_monotone(self, full_grid_city_firms):
        city, _ = full_grid_city_firms
        diffs = np.diff(city.catch_indptr)
        assert np.all(diffs >= 0), "catch_indptr must be non-decreasing"

    def test_indptr_first_zero(self, full_grid_city_firms):
        city, _ = full_grid_city_firms
        assert city.catch_indptr[0] == 0

    def test_nnz_matches_indptr_last(self, full_grid_city_firms):
        city, _ = full_grid_city_firms
        nnz = int(city.catch_indptr[-1])
        assert len(city.catch_indices) == nnz
        assert len(city.catch_tt) == nnz

    def test_indices_in_valid_range(self, full_grid_city_firms):
        city, firms = full_grid_city_firms
        N = len(firms)
        assert int(city.catch_indices.min()) >= 0
        assert int(city.catch_indices.max()) < N, (
            f"catch_indices contains store index >= N={N}"
        )

    def test_tt_values_positive(self, full_grid_city_firms):
        city, _ = full_grid_city_firms
        assert np.all(city.catch_tt >= 0.0), "Travel times must be non-negative"

    def test_non_empty_cells_have_at_least_k_min(self, full_grid_city_firms):
        """Every cell that has at least one travel-time row must have ≥ k_min entries."""
        city, _ = full_grid_city_firms
        k_min = 12
        catchment_sizes = np.diff(city.catch_indptr)
        non_empty = catchment_sizes[catchment_sizes > 0]
        assert np.all(non_empty >= k_min), (
            f"Some non-empty cells have fewer than k_min={k_min} stores. "
            f"Min observed: {int(non_empty.min())}"
        )

    def test_catchment_sizes_at_most_k_max(self, full_grid_city_firms):
        """No cell should exceed k_max stores."""
        city, _ = full_grid_city_firms
        k_max = 80
        catchment_sizes = np.diff(city.catch_indptr)
        assert np.all(catchment_sizes <= k_max), (
            f"Some cells exceed k_max={k_max} stores. "
            f"Max observed: {int(catchment_sizes.max())}"
        )

    def test_dtypes(self, full_grid_city_firms):
        city, _ = full_grid_city_firms
        assert city.catch_indptr.dtype == np.int64,  f"catch_indptr dtype: {city.catch_indptr.dtype}"
        assert city.catch_indices.dtype == np.int32, f"catch_indices dtype: {city.catch_indices.dtype}"
        assert city.catch_tt.dtype == np.float64,    f"catch_tt dtype: {city.catch_tt.dtype}"

    def test_firms_count_matches_N(self, full_grid_city_firms):
        city, firms = full_grid_city_firms
        assert len(city.firms) == len(firms)

    def test_city_has_non_zero_population(self, full_grid_city_firms):
        city, _ = full_grid_city_firms
        assert city.cell_pop.sum() > 0


