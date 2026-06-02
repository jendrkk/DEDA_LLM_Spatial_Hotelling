"""Unit tests for the sparse catchment demand kernels.

Runs the validate_catchment harness on a tiny synthetic city and asserts
that both the stable log-sum-exp path and the expweights fast path stay
within their specified tolerances.

All tests are pure-Python / synthetic-data: no GIS parquet files required.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared tiny-city fixture
# ---------------------------------------------------------------------------

N_STORES = 6
N_CELLS  = 25
TC       = 0.01   # transport_cost
SEED     = 7

# Full-coverage catchment: use a radius larger than the max synthetic travel
# time (60 min) so every cell always keeps ALL stores.  This makes the
# catchment kernel compute EXACTLY the same quantity as the dense kernel,
# allowing numerical agreement checks at 1e-4 / 1e-3 tolerances.
_CATCHMENT_MINUTES_FULL = 1000.0  # >> max synthetic tt (60 min)


def _make_city(precompute_expweights: bool = False):
    """Build a tiny synthetic city with both dense and sparse representations.

    The catchment radius is set to 1000 minutes (much larger than the synthetic
    travel times drawn from [1, 60]) so that EVERY cell keeps ALL N_STORES
    stores.  This makes the catchment kernel numerically equivalent to the
    dense kernel and allows tolerance checks at 1e-4.
    """
    import pandas as pd

    from hotelling.core.city import City
    from hotelling.core.firm import Firm
    from hotelling.spatial.loader import (
        build_catchment,
        populate_catchment_precompute,
    )

    rng = np.random.default_rng(SEED)

    firms = [
        Firm(
            id=str(j),
            location=(float(j) * 100.0, 0.0),
            marginal_cost=0.0,
            quality=rng.uniform(0.0, 1.5),
            kappa0=1.0,
            size=600.0,
            rent=0.0,
            fixed_cost=0.0,
        )
        for j in range(N_STORES)
    ]

    tt_dense = rng.uniform(1.0, 60.0, size=(N_CELLS, N_STORES))
    cell_pop        = rng.uniform(50.0, 500.0, size=N_CELLS)
    lambda_phi      = rng.uniform(0.0, 50.0,  size=N_CELLS)
    pi_H            = rng.uniform(0.2, 0.8,   size=N_CELLS)
    pi_H_lambda_phi = pi_H.copy()

    cell_ids  = [f"cell_{i}" for i in range(N_CELLS)]
    store_ids = [str(j) for j in range(N_STORES)]
    rows, cols = np.meshgrid(np.arange(N_CELLS), np.arange(N_STORES), indexing="ij")
    tt_long = pd.DataFrame({
        "from_id":     [cell_ids[i]  for i in rows.ravel()],
        "to_id":       [store_ids[j] for j in cols.ravel()],
        "travel_time": tt_dense.ravel(),
    })

    indptr, indices, catch_tt = build_catchment(
        tt_df=tt_long,
        cell_ids=cell_ids,
        store_ids=store_ids,
        transport_cost=TC,
        transport_exponent=1.0,
        catchment_minutes=_CATCHMENT_MINUTES_FULL,
        k_min=N_STORES,
        k_max=N_STORES,
    )

    city = City(
        boundary=(0.0, 0.0, 1000.0, 1000.0),
        population_grid=None,
        firms=firms,
        dist2_km2=tt_dense,
        cell_pop=cell_pop,
        lambda_phi=lambda_phi,
        pi_H=pi_H,
        pi_H_lambda_phi=pi_H_lambda_phi,
        alpha=np.array([0.5, 1.5]),
        beta=0.001,
        mu=0.25,
        a0=-1.0,
        transport_exponent=1.0,
        catch_indptr=indptr,
        catch_indices=indices,
        catch_tt=catch_tt,
    )
    populate_catchment_precompute(
        city, transport_cost=TC, precompute_expweights=precompute_expweights
    )
    return city


@pytest.fixture(scope="module")
def tiny_city():
    return _make_city(precompute_expweights=False)


@pytest.fixture(scope="module")
def tiny_city_expw():
    return _make_city(precompute_expweights=True)


# ---------------------------------------------------------------------------
# Helper: dense reference demand
# ---------------------------------------------------------------------------

def _dense_demand(city, prices, efforts):
    from hotelling.core.market import logit_demand
    return logit_demand(
        prices=np.ascontiguousarray(prices,  dtype=np.float64),
        efforts=np.ascontiguousarray(efforts, dtype=np.float64),
        dist2_km2=city.dist2_km2,
        cell_pop=city.cell_pop,
        lambda_phi=city.lambda_phi,
        pi_H=city.pi_H,
        pi_H_lambda_phi=city.pi_H_lambda_phi,
        alpha=city.alpha,
        quality=np.array([f.quality for f in city.firms], dtype=np.float64),
        beta=city.beta,
        transport_cost=TC,
        mu=city.mu,
        a0=city.a0,
        transport_exponent=1.0,
    )


def _stable_demand(city, prices, efforts):
    from hotelling.core.market import _catchment_demand_jit
    N = len(city.firms)
    g = city.beta * np.asarray(efforts, dtype=np.float64) - np.asarray(prices, dtype=np.float64)
    inv_mu    = 1.0 / float(city.mu)
    a0_scaled = float(city.a0) * inv_mu
    return _catchment_demand_jit(
        g,
        np.ascontiguousarray(city.A_quality, dtype=np.float64),
        a0_scaled, inv_mu,
        np.ascontiguousarray(city.w_L, dtype=np.float64),
        np.ascontiguousarray(city.w_H, dtype=np.float64),
        city.catch_indptr.astype(np.int64, copy=False),
        city.catch_indices.astype(np.int32, copy=False),
        np.ascontiguousarray(city.catch_C, dtype=np.float64),
        N,
    )


def _expw_demand(city, prices, efforts):
    from hotelling.core.market import _catchment_demand_expw_jit
    N = len(city.firms)
    g = city.beta * np.asarray(efforts, dtype=np.float64) - np.asarray(prices, dtype=np.float64)
    inv_mu    = 1.0 / float(city.mu)
    a0_scaled = float(city.a0) * inv_mu
    return _catchment_demand_expw_jit(
        g,
        np.ascontiguousarray(city.catch_Kexp_L, dtype=np.float64),
        np.ascontiguousarray(city.catch_Kexp_H, dtype=np.float64),
        float(np.exp(a0_scaled)),
        inv_mu,
        np.ascontiguousarray(city.w_L, dtype=np.float64),
        np.ascontiguousarray(city.w_H, dtype=np.float64),
        city.catch_indptr.astype(np.int64, copy=False),
        city.catch_indices.astype(np.int32, copy=False),
        N,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCatchmentPrecompute:
    def test_fields_populated(self, tiny_city):
        assert tiny_city.catch_C        is not None
        assert tiny_city.A_quality      is not None
        assert tiny_city.w_H            is not None
        assert tiny_city.w_L            is not None
        assert tiny_city.catch_C.shape  == tiny_city.catch_indices.shape
        assert tiny_city.A_quality.shape == (2, N_STORES)
        assert tiny_city.w_H.shape      == tiny_city.cell_pop.shape
        assert tiny_city.w_L.shape      == tiny_city.cell_pop.shape

    def test_catch_C_sign(self, tiny_city):
        # All entries must be non-positive (disutility)
        assert float(tiny_city.catch_C.max()) <= 0.0

    def test_A_quality_values(self, tiny_city):
        alpha_L, alpha_H = float(tiny_city.alpha[0]), float(tiny_city.alpha[1])
        quals = np.array([f.quality for f in tiny_city.firms])
        np.testing.assert_allclose(tiny_city.A_quality[0], alpha_L * quals)
        np.testing.assert_allclose(tiny_city.A_quality[1], alpha_H * quals)

    def test_weights_positive(self, tiny_city):
        assert float(tiny_city.w_H.min()) >= 0.0
        assert float(tiny_city.w_L.min()) >= 0.0

    def test_weights_sum_to_total_mass(self, tiny_city):
        total_mass = tiny_city.cell_pop + tiny_city.lambda_phi
        np.testing.assert_allclose(
            tiny_city.w_H + tiny_city.w_L, total_mass, rtol=1e-10
        )

    def test_expweights_built(self, tiny_city_expw):
        assert tiny_city_expw.precompute_expweights
        assert tiny_city_expw.catch_Kexp_L is not None
        assert tiny_city_expw.catch_Kexp_H is not None
        assert tiny_city_expw.catch_Kexp_L.shape == tiny_city_expw.catch_indices.shape

    def test_expweights_positive(self, tiny_city_expw):
        assert float(tiny_city_expw.catch_Kexp_L.min()) > 0.0
        assert float(tiny_city_expw.catch_Kexp_H.min()) > 0.0


class TestStableCatchmentDemand:
    """Stable log-sum-exp kernel against dense reference."""

    N_SAMPLES = 50
    TOL       = 1e-4

    @pytest.fixture(autouse=True)
    def _rng(self):
        self.rng = np.random.default_rng(SEED + 1)

    def test_non_negative_demand(self, tiny_city):
        prices  = self.rng.uniform(0.5, 3.0, N_STORES)
        efforts = self.rng.uniform(0.0, 5.0, N_STORES)
        d = _stable_demand(tiny_city, prices, efforts)
        assert float(d.min()) >= 0.0

    def test_demand_sum_le_total_mass(self, tiny_city):
        """Total demand ≤ sum(cell_pop + lambda_phi) — consumers can't exceed mass."""
        prices  = self.rng.uniform(0.5, 3.0, N_STORES)
        efforts = self.rng.uniform(0.0, 5.0, N_STORES)
        d = _stable_demand(tiny_city, prices, efforts)
        max_mass = float((tiny_city.cell_pop + tiny_city.lambda_phi).sum())
        assert float(d.sum()) <= max_mass + 1e-8

    def test_demand_shape(self, tiny_city):
        d = _stable_demand(tiny_city, np.ones(N_STORES), np.zeros(N_STORES))
        assert d.shape == (N_STORES,)

    def test_relative_error_vs_dense(self, tiny_city):
        max_rel_err = 0.0
        for _ in range(self.N_SAMPLES):
            prices  = self.rng.uniform(0.5, 5.0, N_STORES)
            efforts = self.rng.uniform(0.0, 10.0, N_STORES)
            d_dense  = _dense_demand(tiny_city, prices, efforts)
            d_stable = _stable_demand(tiny_city, prices, efforts)
            norm_ref = float(np.linalg.norm(d_dense))
            if norm_ref > 0:
                rel_err = float(np.linalg.norm(d_dense - d_stable)) / norm_ref
                max_rel_err = max(max_rel_err, rel_err)

        assert max_rel_err <= self.TOL, (
            f"Stable catchment demand max rel-err={max_rel_err:.2e} exceeds {self.TOL:.0e}. "
            "The test fixture uses a full-coverage catchment (radius=1000 min, k_max=N_STORES) "
            "so the two paths should agree to < 1e-4.  This failure indicates a kernel bug."
        )

    def test_price_monotone_demand(self, tiny_city):
        """Demand for store 0 should fall when its price rises (ceteris paribus)."""
        base_p = np.ones(N_STORES) * 1.0
        high_p = base_p.copy(); high_p[0] = 5.0
        efforts = np.zeros(N_STORES)
        d_base = _stable_demand(tiny_city, base_p, efforts)
        d_high = _stable_demand(tiny_city, high_p, efforts)
        assert d_high[0] <= d_base[0], "Demand did not decrease when price rose"


class TestExpweightsCatchmentDemand:
    """Expweights fast path against dense reference."""

    N_SAMPLES = 50
    TOL       = 1e-3

    @pytest.fixture(autouse=True)
    def _rng(self):
        self.rng = np.random.default_rng(SEED + 2)

    def test_expweights_flag(self, tiny_city_expw):
        assert tiny_city_expw.precompute_expweights

    def test_relative_error_vs_dense(self, tiny_city_expw):
        max_rel_err = 0.0
        for _ in range(self.N_SAMPLES):
            prices  = self.rng.uniform(0.5, 5.0, N_STORES)
            efforts = self.rng.uniform(0.0, 10.0, N_STORES)
            d_dense = _dense_demand(tiny_city_expw, prices, efforts)
            d_expw  = _expw_demand(tiny_city_expw, prices, efforts)
            norm_ref = float(np.linalg.norm(d_dense))
            if norm_ref > 0:
                rel_err = float(np.linalg.norm(d_dense - d_expw)) / norm_ref
                max_rel_err = max(max_rel_err, rel_err)

        assert max_rel_err <= self.TOL, (
            f"Expweights catchment demand max rel-err={max_rel_err:.2e} "
            f"exceeds {self.TOL:.0e}."
        )

    def test_expw_vs_stable_agree(self, tiny_city_expw):
        """Expweights and stable kernels should agree to machine precision."""
        rng = np.random.default_rng(SEED + 99)
        prices  = rng.uniform(0.5, 3.0, N_STORES)
        efforts = rng.uniform(0.0, 5.0, N_STORES)
        d_stable = _stable_demand(tiny_city_expw, prices, efforts)
        d_expw   = _expw_demand(tiny_city_expw, prices, efforts)
        # They differ by the log-sum-exp stabilisation path; allow 1e-6 rel tol
        np.testing.assert_allclose(d_expw, d_stable, rtol=1e-5, atol=1e-10)


class TestCatchmentCellMass:
    def test_inside_sum_equals_demand(self, tiny_city):
        from hotelling.core.market import catchment_cell_mass
        prices  = np.ones(N_STORES) * 1.5
        efforts = np.zeros(N_STORES)
        inside, outside = catchment_cell_mass(
            tiny_city, prices=prices, efforts=efforts, transport_cost=TC
        )
        assert inside.shape  == (len(tiny_city.cell_pop), N_STORES)
        assert outside.shape == (len(tiny_city.cell_pop),)
        # Column sums == catchment demand
        d_from_mass   = inside.sum(axis=0)
        d_from_kernel = _stable_demand(tiny_city, prices, efforts)
        np.testing.assert_allclose(d_from_mass, d_from_kernel, rtol=1e-10)

    def test_outside_mass_nonneg(self, tiny_city):
        from hotelling.core.market import catchment_cell_mass
        prices  = np.ones(N_STORES) * 1.5
        efforts = np.zeros(N_STORES)
        inside, outside = catchment_cell_mass(
            tiny_city, prices=prices, efforts=efforts, transport_cost=TC
        )
        assert float(outside.min()) >= 0.0

    def test_row_sum_le_cell_mass(self, tiny_city):
        from hotelling.core.market import catchment_cell_mass
        prices  = np.ones(N_STORES) * 1.5
        efforts = np.zeros(N_STORES)
        inside, outside = catchment_cell_mass(
            tiny_city, prices=prices, efforts=efforts, transport_cost=TC
        )
        row_sum = inside.sum(axis=1) + outside
        total_w = tiny_city.cell_pop + tiny_city.lambda_phi
        np.testing.assert_allclose(row_sum, total_w, rtol=1e-9,
                                   err_msg="Row conservation violated")


class TestEquilibriumDispatch:
    """Bertrand-Nash and joint-monopoly agree between dense and catchment paths."""

    TOL = 1e-3

    def _sparse_city(self, city):
        c = copy.copy(city)
        c.dist2_km2 = None
        return c

    def test_bertrand_nash_prices_agree(self, tiny_city):
        from hotelling.core.equilibrium import bertrand_nash
        p_dense, _ = bertrand_nash(tiny_city, transport_cost=TC)
        p_catch, _ = bertrand_nash(self._sparse_city(tiny_city), transport_cost=TC)
        err = float(np.max(np.abs(p_dense - p_catch)))
        assert err <= self.TOL, (
            f"Bertrand-Nash price disagreement={err:.2e} > {self.TOL:.0e}"
        )

    def test_joint_monopoly_prices_agree(self, tiny_city):
        from hotelling.core.equilibrium import joint_monopoly
        p_dense, _ = joint_monopoly(tiny_city, transport_cost=TC)
        p_catch, _ = joint_monopoly(self._sparse_city(tiny_city), transport_cost=TC)
        err = float(np.max(np.abs(p_dense - p_catch)))
        assert err <= self.TOL, (
            f"Joint-monopoly price disagreement={err:.2e} > {self.TOL:.0e}"
        )

    def test_param_signature_includes_catchment(self, tiny_city):
        from hotelling.core.equilibrium import _param_signature
        sparse = self._sparse_city(tiny_city)
        sig = _param_signature(sparse, TC)
        assert isinstance(sig, str) and len(sig) == 12

    def test_param_signature_differs_from_dense(self, tiny_city):
        from hotelling.core.equilibrium import _param_signature
        sig_dense  = _param_signature(tiny_city, TC)
        sparse = self._sparse_city(tiny_city)
        sig_catch  = _param_signature(sparse, TC)
        # Dense and catchment cities have different distance representations
        # so their signatures should differ (dense includes dist_sum, catchment
        # includes nnz/tt_sum).
        assert sig_dense != sig_catch


class TestMarketClearingDispatch:
    """market_clearing_arrays dispatches to catchment path automatically."""

    def test_dispatch_uses_catchment(self, tiny_city):
        from hotelling.core.market import market_clearing_arrays, precompute_firm_arrays
        fa = precompute_firm_arrays(tiny_city.firms)
        prices  = np.ones(N_STORES) * 1.5
        efforts = np.zeros(N_STORES)
        d_mc, _ = market_clearing_arrays(prices, efforts, tiny_city, TC, fa)
        d_ref   = _stable_demand(tiny_city, prices, efforts)
        np.testing.assert_allclose(d_mc, d_ref, rtol=1e-10)

    def test_dispatch_dense_city_unchanged(self, tiny_city):
        """Dense city (dist2_km2 not None, no catchment) should use the dense kernel."""
        import copy as _copy
        from hotelling.core.market import market_clearing_arrays, precompute_firm_arrays, logit_demand
        city_dense = _copy.copy(tiny_city)
        city_dense.catch_indptr = None
        fa = precompute_firm_arrays(city_dense.firms)
        prices  = np.ones(N_STORES) * 1.5
        efforts = np.zeros(N_STORES)
        d_mc, _ = market_clearing_arrays(prices, efforts, city_dense, TC, fa)
        d_ref   = _dense_demand(city_dense, prices, efforts)
        np.testing.assert_allclose(d_mc, d_ref, rtol=1e-10)
