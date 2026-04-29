"""Unit tests for Step-1 market-clearing behavior."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.core.city import City
from hotelling.core.firm import Firm
from hotelling.core.market import market_clearing


def _make_simple_city(marginal_cost: float = 1.0) -> City:
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
        pi_H=np.array([0.3, 0.7], dtype=float),
        pi_H_lambda_phi=np.array([0.3, 0.7], dtype=float),
        alpha=np.array([0.0, 0.0], dtype=float),
        beta=0.0,
        mu=0.25,
        a0=-100.0,
    )


def test_demands_do_not_exceed_total_population_mass() -> None:
    city = _make_simple_city(marginal_cost=1.0)
    prices = np.array([1.2, 1.25], dtype=float)
    efforts = np.zeros(2, dtype=float)

    demands, _ = market_clearing(prices=prices, efforts=efforts, city=city, transport_cost=0.0)

    total_population_mass = float((city.cell_pop + city.lambda_phi).sum())
    assert demands.sum() <= total_population_mass + 1e-9
    assert np.all(demands >= 0.0)


def test_profits_non_negative_when_price_above_marginal_cost() -> None:
    city = _make_simple_city(marginal_cost=1.0)
    prices = np.array([1.5, 1.6], dtype=float)
    efforts = np.zeros(2, dtype=float)

    _, profits = market_clearing(prices=prices, efforts=efforts, city=city, transport_cost=0.0)

    assert np.all(profits >= 0.0)


def test_symmetric_duopoly_matches_closed_form_nash_price() -> None:
    city = _make_simple_city(marginal_cost=1.0)
    expected_price = 1.0 + 2.0 * city.mu  # Calvano-style symmetric 2-firm logit markup.
    prices = np.array([expected_price, expected_price], dtype=float)
    efforts = np.zeros(2, dtype=float)

    demands, _ = market_clearing(prices=prices, efforts=efforts, city=city, transport_cost=0.0)
    total_population_mass = float((city.cell_pop + city.lambda_phi).sum())

    assert demands[0] == pytest.approx(total_population_mass / 2.0, rel=1e-2)
    assert demands[1] == pytest.approx(total_population_mass / 2.0, rel=1e-2)
