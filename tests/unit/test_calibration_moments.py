"""Unit tests for structural calibration model-moment functions."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.calibration import moments as moments_mod
from hotelling.calibration.moments import (
    all_model_moments,
    bio_income_gradient,
    chain_shares,
    mean_gross_margin,
    outside_share,
)
from hotelling.core.city import City
from hotelling.core.equilibrium import bertrand_nash
from hotelling.core.firm import Firm

_Q_S = 3.0
_Q_B = 15.0
_TRANSPORT_COST = 0.4
_RNG_SEED = 42


def _make_firm(
    firm_id: str,
    quality: float,
    *,
    x: float,
    marginal_cost: float,
    chain_type: str,
) -> Firm:
    return Firm(
        id=firm_id,
        location=(x, 0.0),
        marginal_cost=marginal_cost,
        quality=quality,
        kappa0=1.0,
        size=500.0,
        rent=0.0,
        fixed_cost=0.0,
        chain=firm_id,
        chain_type=chain_type,
    )


def _make_synthetic_city(
    rng: np.random.Generator,
    *,
    pi_H: np.ndarray | None = None,
    dist2_km2: np.ndarray | None = None,
) -> City:
    m_cells = 20
    n_firms = 6
    if pi_H is None:
        pi_H = np.linspace(0.1, 0.9, m_cells)
    if dist2_km2 is None:
        dist2_km2 = rng.uniform(1.0, 15.0, size=(m_cells, n_firms))

    firms = [
        _make_firm("d1", 0.0, x=0.0, marginal_cost=10.0, chain_type="discount"),
        _make_firm("d2", 0.0, x=1.0, marginal_cost=10.0, chain_type="discount"),
        _make_firm("s1", _Q_S, x=2.0, marginal_cost=25.0, chain_type="standard"),
        _make_firm("s2", _Q_S, x=3.0, marginal_cost=25.0, chain_type="standard"),
        _make_firm("b1", _Q_B, x=4.0, marginal_cost=35.0, chain_type="bio"),
        _make_firm("b2", _Q_B, x=5.0, marginal_cost=35.0, chain_type="bio"),
    ]

    cell_pop = rng.uniform(50.0, 200.0, size=m_cells)
    lambda_phi = rng.uniform(1.0, 10.0, size=m_cells)

    return City(
        boundary=(0.0, 0.0, 10.0, 10.0),
        population_grid=None,
        firms=firms,
        dist2_km2=dist2_km2,
        cell_pop=cell_pop,
        lambda_phi=lambda_phi,
        pi_H=pi_H,
        pi_H_lambda_phi=pi_H,
        alpha=np.array([1.0, 2.5]),
        beta=0.001,
        mu=5.0,
        a0=-5.0,
    )


@pytest.fixture
def synthetic_city() -> City:
    rng = np.random.default_rng(_RNG_SEED)
    return _make_synthetic_city(rng)


def test_chain_shares_sum_to_one(synthetic_city: City) -> None:
    shares = chain_shares(
        synthetic_city, _TRANSPORT_COST, q_S=_Q_S, q_B=_Q_B
    )
    assert set(shares) == {"discount", "standard", "bio"}
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-6)


def test_outside_share_in_unit_interval(synthetic_city: City) -> None:
    share = outside_share(synthetic_city, _TRANSPORT_COST)
    assert 0.0 < share < 1.0


def test_mean_gross_margin_positive_and_below_one(synthetic_city: City) -> None:
    margin = mean_gross_margin(synthetic_city, _TRANSPORT_COST)
    assert 0.0 < margin < 1.0


def test_bio_income_gradient_positive_when_bio_attractive() -> None:
    rng = np.random.default_rng(_RNG_SEED)
    m_cells = 20
    pi_H = np.concatenate(
        [
            np.full(5, 0.1),
            np.linspace(0.2, 0.8, 10),
            np.full(5, 0.9),
        ]
    )
    dist = rng.uniform(5.0, 15.0, size=(m_cells, 6))
    dist[15:, 4:] = rng.uniform(1.0, 3.0, size=(5, 2))

    city = _make_synthetic_city(rng, pi_H=pi_H, dist2_km2=dist)
    city = City(
        boundary=city.boundary,
        population_grid=city.population_grid,
        firms=city.firms,
        dist2_km2=city.dist2_km2,
        cell_pop=city.cell_pop,
        lambda_phi=city.lambda_phi,
        pi_H=city.pi_H,
        pi_H_lambda_phi=city.pi_H_lambda_phi,
        alpha=np.array([1.0, 4.0]),
        beta=city.beta,
        mu=city.mu,
        a0=city.a0,
    )

    gradient = bio_income_gradient(city, _TRANSPORT_COST, q_S=_Q_S, q_B=_Q_B)
    assert np.isfinite(gradient)
    assert gradient > 1.0


def test_all_model_moments_keys_and_single_nash_solve(
    synthetic_city: City, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_count = 0
    real_bertrand_nash = bertrand_nash

    def counting_bertrand_nash(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_bertrand_nash(*args, **kwargs)

    monkeypatch.setattr(
        moments_mod, "bertrand_nash", counting_bertrand_nash
    )

    result = all_model_moments(
        synthetic_city, _TRANSPORT_COST, q_S=_Q_S, q_B=_Q_B
    )

    expected_keys = {
        "mean_gross_margin",
        "outside_share",
        "chain_share_discount",
        "chain_share_standard",
        "chain_share_bio",
        "bio_income_gradient",
    }
    assert set(result) == expected_keys
    assert call_count == 1
