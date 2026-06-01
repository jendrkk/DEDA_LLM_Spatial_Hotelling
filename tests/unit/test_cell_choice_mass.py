"""Unit tests for cell_choice_mass and cell_metrics.

Tests exercise the public wrappers directly with hand-built array inputs —
no City fixture, no parquet I/O — so they run without any spatial data files.

References
----------
Anderson, de Palma & Thisse (1992) *Discrete Choice Theory of Product
Differentiation*, Ch. 3.
Calvano, E. et al. (2020) *Artificial Intelligence, Algorithmic Pricing,
and Collusion*, AER §II.A.
"""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.core.market import cell_choice_mass, cell_metrics, logit_demand
from hotelling.core.city import City
from hotelling.core.firm import Firm


# ---------------------------------------------------------------------------
# Shared random fixture
# ---------------------------------------------------------------------------

M, N = 20, 5
_RNG_SEED = 0


def _make_inputs(rng: np.random.Generator) -> dict:
    """Return a consistent set of random arrays for M=20 cells and N=5 stores."""
    prices = rng.uniform(1.0, 3.0, size=N)
    efforts = rng.uniform(0.0, 1.0, size=N)
    dist2_km2 = rng.uniform(0.01, 25.0, size=(M, N))
    cell_pop = rng.uniform(50.0, 500.0, size=M)
    lambda_phi = rng.uniform(5.0, 50.0, size=M)
    pi_H = rng.uniform(0.1, 0.9, size=M)
    pi_H_lambda_phi = rng.uniform(0.1, 0.9, size=M)
    alpha = np.array([0.5, 1.0])
    quality = rng.uniform(0.0, 2.0, size=N)
    beta = 0.3
    transport_cost = 0.1
    mu = 0.25
    a0 = 0.0
    transport_exponent = 1.0
    return dict(
        prices=prices,
        efforts=efforts,
        dist2_km2=dist2_km2,
        cell_pop=cell_pop,
        lambda_phi=lambda_phi,
        pi_H=pi_H,
        pi_H_lambda_phi=pi_H_lambda_phi,
        alpha=alpha,
        quality=quality,
        beta=beta,
        transport_cost=transport_cost,
        mu=mu,
        a0=a0,
        transport_exponent=transport_exponent,
    )


# ---------------------------------------------------------------------------
# Test 1: column sums of inside_mass reproduce logit_demand exactly
# ---------------------------------------------------------------------------

def test_column_sum_matches_logit_demand() -> None:
    """inside_mass.sum(axis=0) must equal logit_demand(...) within atol=1e-9."""
    rng = np.random.default_rng(_RNG_SEED)
    kw = _make_inputs(rng)

    inside, _outside = cell_choice_mass(**kw)

    expected = logit_demand(
        prices=kw["prices"],
        efforts=kw["efforts"],
        dist2_km2=kw["dist2_km2"],
        cell_pop=kw["cell_pop"],
        lambda_phi=kw["lambda_phi"],
        pi_H=kw["pi_H"],
        pi_H_lambda_phi=kw["pi_H_lambda_phi"],
        alpha=kw["alpha"],
        quality=kw["quality"],
        beta=kw["beta"],
        transport_cost=kw["transport_cost"],
        mu=kw["mu"],
        a0=kw["a0"],
        transport_exponent=kw["transport_exponent"],
    )

    np.testing.assert_allclose(
        inside.sum(axis=0),
        expected,
        atol=1e-9,
        err_msg="Column sums of inside_mass differ from logit_demand output.",
    )


# ---------------------------------------------------------------------------
# Test 2: inside + outside mass equals total consumer weight per cell
# ---------------------------------------------------------------------------

def test_total_mass_conservation() -> None:
    """inside_mass.sum() + outside_mass.sum() == (cell_pop + lambda_phi).sum()."""
    rng = np.random.default_rng(_RNG_SEED)
    kw = _make_inputs(rng)

    inside, outside = cell_choice_mass(**kw)

    total_weight = (kw["cell_pop"] + kw["lambda_phi"]).sum()
    total_allocated = inside.sum() + outside.sum()

    np.testing.assert_allclose(
        total_allocated,
        total_weight,
        atol=1e-6,
        err_msg=(
            f"Total allocated mass {total_allocated:.6f} does not equal "
            f"total consumer weight {total_weight:.6f}."
        ),
    )


# ---------------------------------------------------------------------------
# Test 3: all entries non-negative
# ---------------------------------------------------------------------------

def test_all_entries_non_negative() -> None:
    """inside_mass and outside_mass must be element-wise >= 0."""
    rng = np.random.default_rng(_RNG_SEED)
    kw = _make_inputs(rng)

    inside, outside = cell_choice_mass(**kw)

    assert np.all(inside >= 0.0), "inside_mass contains negative entries."
    assert np.all(outside >= 0.0), "outside_mass contains negative entries."


# ---------------------------------------------------------------------------
# Additional: shape checks
# ---------------------------------------------------------------------------

def test_output_shapes() -> None:
    rng = np.random.default_rng(_RNG_SEED)
    kw = _make_inputs(rng)

    inside, outside = cell_choice_mass(**kw)

    assert inside.shape == (M, N), f"Expected ({M}, {N}), got {inside.shape}"
    assert outside.shape == (M,), f"Expected ({M},), got {outside.shape}"


# ---------------------------------------------------------------------------
# Additional: outside-only market (very negative a0 suppresses outside option)
# ---------------------------------------------------------------------------

def test_very_negative_a0_suppresses_outside_mass() -> None:
    """With a0 = -1e6, essentially all mass goes to stores."""
    rng = np.random.default_rng(_RNG_SEED)
    kw = _make_inputs(rng)
    kw["a0"] = -1e6

    inside, outside = cell_choice_mass(**kw)

    total_weight = (kw["cell_pop"] + kw["lambda_phi"]).sum()
    assert outside.sum() < 1e-3 * total_weight, (
        "outside_mass should be negligible when a0 = -1e6"
    )


# ---------------------------------------------------------------------------
# cell_metrics smoke tests (uses a tiny City-like setup)
# ---------------------------------------------------------------------------

def _make_tiny_city(rng: np.random.Generator) -> tuple[City, np.ndarray, np.ndarray]:
    """Build a minimal City with 3 cells and 2 stores for smoke-testing cell_metrics."""
    M_s, N_s = 6, 3
    firms = [
        Firm(
            id=f"firm_{j}",
            location=(float(j), 0.5),
            marginal_cost=1.0,
            quality=float(j) * 0.5,
            kappa0=1.0,
            size=1.0,
            rent=0.0,
        )
        for j in range(N_s)
    ]
    city = City(
        boundary=(0.0, 0.0, 3.0, 1.0),
        population_grid=None,
        firms=firms,
        dist2_km2=rng.uniform(0.01, 4.0, size=(M_s, N_s)),
        cell_pop=rng.uniform(50.0, 200.0, size=M_s),
        lambda_phi=rng.uniform(5.0, 30.0, size=M_s),
        pi_H=rng.uniform(0.2, 0.8, size=M_s),
        pi_H_lambda_phi=rng.uniform(0.2, 0.8, size=M_s),
        alpha=np.array([0.4, 0.9]),
        beta=0.2,
        mu=0.25,
        a0=0.0,
        transport_exponent=1.0,
    )
    prices = rng.uniform(1.0, 2.5, size=N_s)
    efforts = rng.uniform(0.0, 1.0, size=N_s)
    return city, prices, efforts


@pytest.mark.parametrize("metric", ["expected_price", "served_demand", "dominant_chain", "consumer_surplus"])
def test_cell_metrics_shape_and_dtype(metric: str) -> None:
    """cell_metrics returns a (M,) float64 array for all supported metrics."""
    rng = np.random.default_rng(7)
    city, prices, efforts = _make_tiny_city(rng)

    result = cell_metrics(prices, efforts, city, transport_cost=0.05, metric=metric)

    assert result.shape == (len(city.cell_pop),), (
        f"metric={metric!r}: expected shape ({len(city.cell_pop)},), got {result.shape}"
    )
    assert result.dtype == np.float64, (
        f"metric={metric!r}: expected float64, got {result.dtype}"
    )


def test_cell_metrics_served_demand_matches_logit_demand() -> None:
    """served_demand summed over cells must equal logit_demand column sum."""
    rng = np.random.default_rng(13)
    city, prices, efforts = _make_tiny_city(rng)

    served = cell_metrics(prices, efforts, city, transport_cost=0.05, metric="served_demand")

    qualities = np.array([f.quality for f in city.firms], dtype=np.float64)
    expected = logit_demand(
        prices=prices,
        efforts=efforts,
        dist2_km2=city.dist2_km2,
        cell_pop=city.cell_pop,
        lambda_phi=city.lambda_phi,
        pi_H=city.pi_H,
        pi_H_lambda_phi=city.pi_H_lambda_phi,
        alpha=city.alpha,
        quality=qualities,
        beta=city.beta,
        transport_cost=0.05,
        mu=city.mu,
        a0=city.a0,
        transport_exponent=city.transport_exponent,
    )

    np.testing.assert_allclose(
        served.sum(),
        expected.sum(),
        atol=1e-9,
        err_msg="Sum of served_demand should equal total logit_demand.",
    )


def test_cell_metrics_invalid_metric_raises() -> None:
    rng = np.random.default_rng(99)
    city, prices, efforts = _make_tiny_city(rng)
    with pytest.raises(ValueError, match="Unknown metric"):
        cell_metrics(prices, efforts, city, transport_cost=0.0, metric="unicorn")
