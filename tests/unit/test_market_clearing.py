"""Unit tests for Step-1 market-clearing behavior."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.core.city import City
from hotelling.core.firm import Firm
from hotelling.core.market import market_clearing, profit


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


class TestProfit:
    """Unit tests for hotelling.core.market.profit()."""

    def test_basic_no_costs(self) -> None:
        """(p - c) * D with zero effort and zero fixed/rent costs."""
        result = profit(
            price=2.0,
            demand=100.0,
            marginal_cost=1.0,
            kappa0=1.0,
            effort=0.0,
            size=1.0,
            rent=0.0,
            fixed_cost=0.0,
        )
        assert result == pytest.approx(100.0)

    def test_fixed_cost_subtracted(self) -> None:
        """fixed_cost reduces profit by exactly its value."""
        base = profit(
            price=2.0,
            demand=100.0,
            marginal_cost=1.0,
            kappa0=1.0,
            effort=0.0,
            size=1.0,
            rent=0.0,
            fixed_cost=0.0,
        )
        with_fc = profit(
            price=2.0,
            demand=100.0,
            marginal_cost=1.0,
            kappa0=1.0,
            effort=0.0,
            size=1.0,
            rent=0.0,
            fixed_cost=5.0,
        )
        assert with_fc == pytest.approx(base - 5.0)

    def test_fixed_cost_array_subtracted_elementwise(self) -> None:
        """Per-firm fixed_cost array is subtracted element-wise."""
        prices = np.array([2.0, 3.0])
        demands = np.array([100.0, 80.0])
        mc = np.array([1.0, 1.0])
        fc = np.array([5.0, 10.0])
        result = profit(
            price=prices,
            demand=demands,
            marginal_cost=mc,
            kappa0=1.0,
            effort=np.zeros(2),
            size=np.ones(2),
            rent=np.zeros(2),
            fixed_cost=fc,
        )
        expected = np.array([
            (2.0 - 1.0) * 100.0 - 5.0,
            (3.0 - 1.0) * 80.0  - 10.0,
        ])
        np.testing.assert_allclose(result, expected)

    def test_fixed_cost_default_zero_backward_compat(self) -> None:
        """Omitting fixed_cost gives the same result as fixed_cost=0."""
        r_omitted = profit(
            price=2.0,
            demand=50.0,
            marginal_cost=1.0,
            kappa0=2.0,
            effort=1.0,
            size=1.0,
            rent=0.5,
        )
        r_explicit = profit(
            price=2.0,
            demand=50.0,
            marginal_cost=1.0,
            kappa0=2.0,
            effort=1.0,
            size=1.0,
            rent=0.5,
            fixed_cost=0.0,
        )
        assert r_omitted == pytest.approx(r_explicit)

    def test_effort_cost_and_rent_and_fixed_cost_combined(self) -> None:
        """All cost components reduce profit correctly."""
        result = profit(
            price=3.0,
            demand=10.0,
            marginal_cost=1.0,
            kappa0=2.0,
            effort=2.0,
            size=100.0,
            rent=0.01,
            fixed_cost=3.0,
        )
        # gross margin = (3-1)*10 = 20
        # effort cost  = 0.5 * 2 * 4 = 4
        # rent*size    = 0.01 * 100  = 1
        # fixed_cost   = 3
        assert result == pytest.approx(20.0 - 4.0 - 1.0 - 3.0)


def test_symmetric_duopoly_matches_closed_form_nash_price() -> None:
    city = _make_simple_city(marginal_cost=1.0)
    expected_price = 1.0 + 2.0 * city.mu  # Calvano-style symmetric 2-firm logit markup.
    prices = np.array([expected_price, expected_price], dtype=float)
    efforts = np.zeros(2, dtype=float)

    demands, _ = market_clearing(prices=prices, efforts=efforts, city=city, transport_cost=0.0)
    total_population_mass = float((city.cell_pop + city.lambda_phi).sum())

    assert demands[0] == pytest.approx(total_population_mass / 2.0, rel=1e-2)
    assert demands[1] == pytest.approx(total_population_mass / 2.0, rel=1e-2)


def test_market_clearing_fixed_cost_reduces_profits() -> None:
    """market_clearing passes Firm.fixed_cost through to reported profits.

    Demands must be identical regardless of fixed_cost (pricing FOC invariance),
    while profits must differ by exactly the fixed_cost value.
    """
    def _city_with_fc(fc0: float, fc1: float) -> City:
        firms = [
            Firm(
                id="firm_0",
                location=(0.25, 0.5),
                marginal_cost=1.0,
                quality=0.0,
                kappa0=1.0,
                size=1.0,
                rent=0.0,
                fixed_cost=fc0,
                chain="A",
            ),
            Firm(
                id="firm_1",
                location=(0.75, 0.5),
                marginal_cost=1.0,
                quality=0.0,
                kappa0=1.0,
                size=1.0,
                rent=0.0,
                fixed_cost=fc1,
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

    prices = np.array([1.5, 1.5], dtype=float)
    efforts = np.zeros(2, dtype=float)

    demands_no_fc, profits_no_fc = market_clearing(
        prices=prices,
        efforts=efforts,
        city=_city_with_fc(0.0, 0.0),
        transport_cost=0.0,
    )
    demands_fc, profits_fc = market_clearing(
        prices=prices,
        efforts=efforts,
        city=_city_with_fc(3.0, 7.0),
        transport_cost=0.0,
    )

    # Demands must be unchanged (fixed_cost is not in the utility / price FOC)
    np.testing.assert_allclose(demands_fc, demands_no_fc, rtol=1e-12)

    # Profits must decrease by exactly the respective fixed costs
    assert profits_fc[0] == pytest.approx(profits_no_fc[0] - 3.0)
    assert profits_fc[1] == pytest.approx(profits_no_fc[1] - 7.0)
