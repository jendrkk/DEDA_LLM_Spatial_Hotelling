"""Unit tests for effort parameter calibration (ADR-031)."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.calibration.structural import compute_effort_params
from hotelling.core.city import City
from hotelling.core.firm import Firm


def _minimal_city(*, beta: float = 0.001, kappa0: float = 1.0) -> City:
    return City(
        boundary=(0.0, 0.0, 1.0, 1.0),
        population_grid=None,
        firms=[
            Firm(
                id="s0",
                location=(0.0, 0.0),
                marginal_cost=10.0,
                quality=0.0,
                kappa0=kappa0,
                size=500.0,
                rent=0.0,
                chain_type="discount",
            ),
            Firm(
                id="s1",
                location=(0.5, 0.5),
                marginal_cost=12.0,
                quality=6.0,
                kappa0=kappa0,
                size=500.0,
                rent=0.0,
                chain_type="standard",
            ),
            Firm(
                id="s2",
                location=(1.0, 1.0),
                marginal_cost=14.0,
                quality=18.0,
                kappa0=kappa0,
                size=500.0,
                rent=0.0,
                chain_type="bio",
            ),
        ],
        dist2_km2=np.ones((2, 3), dtype=float),
        cell_pop=np.array([100.0, 200.0]),
        lambda_phi=np.array([10.0, 20.0]),
        pi_H=np.array([0.3, 0.7]),
        pi_H_lambda_phi=np.array([0.3, 0.7]),
        alpha=np.array([0.5, 1.25]),
        beta=beta,
        mu=5.0,
        a0=-10.0,
    )


def _patch_nash_and_demand(monkeypatch, demands: np.ndarray) -> None:
    prices = np.array([30.0, 35.0, 40.0], dtype=np.float64)

    def fake_bertrand_nash(city, transport_cost=1.0, **kwargs):
        return prices, np.zeros(len(city.firms))

    def fake_logit_demand(p, e, *args, **kwargs):
        return np.asarray(demands, dtype=np.float64)

    monkeypatch.setattr(
        "hotelling.core.equilibrium.bertrand_nash", fake_bertrand_nash
    )
    monkeypatch.setattr(
        "hotelling.core.market.logit_demand", fake_logit_demand
    )


@pytest.fixture
def city() -> City:
    return _minimal_city()


@pytest.fixture
def uniform_demands() -> np.ndarray:
    return np.array([100.0, 100.0, 100.0], dtype=np.float64)


def test_beta_equals_X_times_basket_over_e_max(
    monkeypatch, city, uniform_demands
):
    _patch_nash_and_demand(monkeypatch, uniform_demands)
    basket = 40.0
    e_max = 1.0
    X = 0.10
    rho = 0.40

    result = compute_effort_params(
        city,
        transport_cost=0.5,
        basket_price_standard_eur=basket,
        e_max=e_max,
        X=X,
        rho=rho,
    )

    expected_beta = X * basket / e_max
    assert result["beta_effort"] == pytest.approx(expected_beta)
    assert result["wtp_full_pct"] == pytest.approx(X)


def test_mean_e_star_equals_rho_times_e_max_with_uniform_D(
    monkeypatch, city, uniform_demands
):
    _patch_nash_and_demand(monkeypatch, uniform_demands)
    basket = 40.0
    e_max = 1.0
    X = 0.10
    rho = 0.40

    result = compute_effort_params(
        city,
        transport_cost=0.5,
        basket_price_standard_eur=basket,
        e_max=e_max,
        X=X,
        rho=rho,
    )

    beta = X * basket / e_max
    d_bar = float(uniform_demands.mean())
    expected_kappa0 = beta * d_bar / (rho * e_max)

    assert result["kappa0"] == pytest.approx(expected_kappa0)
    assert result["e_star_mean"] == pytest.approx(rho * e_max, abs=1e-9)
    assert result["wtp_equil_pct"] == pytest.approx(X * rho, abs=1e-9)


def test_interior_fraction_in_unit_interval(monkeypatch, city):
    demands = np.array([50.0, 100.0, 150.0], dtype=np.float64)
    _patch_nash_and_demand(monkeypatch, demands)

    result = compute_effort_params(
        city,
        transport_cost=0.5,
        basket_price_standard_eur=40.0,
        e_max=1.0,
        X=0.10,
        rho=0.40,
    )

    assert 0.0 <= result["interior_fraction"] <= 1.0
    assert result["e_star_min"] >= 0.0
    assert result["e_star_max"] <= 1.0 + 1e-9


def test_larger_X_increases_beta(monkeypatch, city, uniform_demands):
    _patch_nash_and_demand(monkeypatch, uniform_demands)

    low = compute_effort_params(
        city, 0.5, 40.0, 1.0, X=0.05, rho=0.40
    )
    high = compute_effort_params(
        city, 0.5, 40.0, 1.0, X=0.15, rho=0.40
    )

    assert high["beta_effort"] > low["beta_effort"]
    assert high["kappa0"] > low["kappa0"]


def test_larger_rho_decreases_kappa0(monkeypatch, city, uniform_demands):
    _patch_nash_and_demand(monkeypatch, uniform_demands)

    low_rho = compute_effort_params(
        city, 0.5, 40.0, 1.0, X=0.10, rho=0.25
    )
    high_rho = compute_effort_params(
        city, 0.5, 40.0, 1.0, X=0.10, rho=0.50
    )

    assert high_rho["kappa0"] < low_rho["kappa0"]
    assert high_rho["e_star_mean"] > low_rho["e_star_mean"]
