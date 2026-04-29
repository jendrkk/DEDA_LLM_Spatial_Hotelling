"""Unit tests for Step-1 equilibrium solvers."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.core.city import City
from hotelling.core.equilibrium import bertrand_nash, joint_monopoly
from hotelling.core.firm import Firm
from hotelling.core.market import market_clearing


def _make_simple_city(marginal_cost: float = 1.0, a0: float = 0.0) -> City:
    firms = [
        Firm(
            id="firm_0",
            location=(0.25, 0.5),
            marginal_cost=marginal_cost,
            quality=0.0,
            kappa0=1.0,
            size=1.0,
            rent=0.0,
            chain="A",
        ),
        Firm(
            id="firm_1",
            location=(0.75, 0.5),
            marginal_cost=marginal_cost,
            quality=0.0,
            kappa0=1.0,
            size=1.0,
            rent=0.0,
            chain="B",
        ),
    ]
    return City(
        boundary=(0.0, 0.0, 1.0, 1.0),
        population_grid=None,
        firms=firms,
        dist2_km2=np.zeros((2, 2), dtype=float),
        cell_pop=np.array([100.0, 150.0], dtype=float),
        lambda_phi=np.array([10.0, 20.0], dtype=float),
        pi_H=np.array([0.4, 0.6], dtype=float),
        pi_H_lambda_phi=np.array([0.4, 0.6], dtype=float),
        alpha=np.array([0.0, 0.0], dtype=float),
        beta=0.0,
        mu=0.25,
        a0=a0,
    )


def test_bertrand_nash_prices_are_above_cost() -> None:
    city = _make_simple_city(marginal_cost=1.0)
    prices, _ = bertrand_nash(city=city, transport_cost=0.0, tol=1e-10, max_iter=500)
    costs = np.array([f.marginal_cost for f in city.firms], dtype=float)
    assert np.all(prices > costs)


def test_joint_monopoly_profits_are_at_least_nash_profits() -> None:
    city = _make_simple_city(marginal_cost=1.0)

    nash_prices, nash_efforts = bertrand_nash(city=city, transport_cost=0.0, tol=1e-10, max_iter=500)
    _, nash_profits = market_clearing(
        prices=nash_prices, efforts=nash_efforts, city=city, transport_cost=0.0
    )

    monopoly_prices, monopoly_efforts = joint_monopoly(city=city, transport_cost=0.0)
    _, monopoly_profits = market_clearing(
        prices=monopoly_prices, efforts=monopoly_efforts, city=city, transport_cost=0.0
    )

    assert monopoly_profits.sum() >= nash_profits.sum()


def test_bertrand_nash_symmetric_converges_to_known_price() -> None:
    city = _make_simple_city(marginal_cost=1.0, a0=-100.0)
    prices, _ = bertrand_nash(city=city, transport_cost=0.0, tol=1e-12, max_iter=500)
    expected_price = 1.5
    assert prices[0] == pytest.approx(expected_price, rel=1e-3)
    assert prices[1] == pytest.approx(expected_price, rel=1e-3)
