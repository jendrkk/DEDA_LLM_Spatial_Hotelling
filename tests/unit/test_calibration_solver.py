"""Unit tests for the structural calibration least-squares solver."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.calibration import structural
from hotelling.core.city import City
from hotelling.core.firm import Firm

_TRUE_MU = 4.2
_TRUE_A0 = -4.5


def _minimal_city(
    *,
    mu: float,
    a0: float,
    alpha_L: float,
    alpha_H: float,
) -> City:
    return City(
        boundary=(0.0, 0.0, 1.0, 1.0),
        population_grid=None,
        firms=[
            Firm(
                id="s0",
                location=(0.0, 0.0),
                marginal_cost=10.0,
                quality=0.0,
                kappa0=1.0,
                size=500.0,
                rent=0.0,
                chain_type="discount",
            )
        ],
        dist2_km2=np.ones((2, 1), dtype=float),
        cell_pop=np.array([100.0, 200.0]),
        lambda_phi=np.array([10.0, 20.0]),
        pi_H=np.array([0.3, 0.7]),
        pi_H_lambda_phi=np.array([0.3, 0.7]),
        alpha=np.array([alpha_L, alpha_H]),
        beta=0.001,
        mu=mu,
        a0=a0,
    )


def _fake_targets() -> dict:
    return {
        "basket_price_standard_eur": 40.0,
        "price_index": {"discount": 0.85, "standard": 1.0, "bio": 1.3},
        "gross_margin_common": _TRUE_MU * 0.05,
        "gross_margin_by_chain": {"discount": 0.18, "standard": 0.24, "bio": 0.30},
        "use_common_margin": True,
        "wage_monthly_gross_eur": 3955.0,
        "work_hours_per_month": 167.0,
        "vtt_wage_ratio": 0.5,
        "round_trip_factor": 2.0,
        "outside_share_target": -_TRUE_A0 * 0.01,
        "alpha_ratio": 2.5,
        "bio_share_income_gradient_target": 2.5,
    }


def _fake_env_cfg() -> dict:
    return {
        "grid_path": "unused.parquet",
        "stores_path": "unused.parquet",
        "travel_times_path": "unused.parquet",
        "lambda_val": 1200.0,
        "beta_effort": 0.001,
        "kappa0": 1.0,
        "store_size": 600.0,
        "nan_fill_minutes": 120.0,
        "rent_scale": 0.0,
        "rent_normalization": "mean_ratio",
    }


@pytest.fixture
def patched_solver(monkeypatch: pytest.MonkeyPatch):
    build_count = 0

    def fake_load_berlin_city(
        grid_path,
        stores_path,
        travel_times_path,
        *,
        lambda_val,
        q_S=0.8,
        q_B=1.5,
        alpha_L=0.5,
        alpha_H=1.5,
        mu=0.25,
        a0=-1.0,
        **kwargs,
    ):
        nonlocal build_count
        build_count += 1
        return _minimal_city(
            mu=mu, a0=a0, alpha_L=alpha_L, alpha_H=alpha_H
        ), []

    def fake_all_model_moments(city, transport_cost, q_S, q_B):
        mu = city.mu
        a0 = city.a0
        return {
            "mean_gross_margin": mu * 0.05,
            "outside_share": -a0 * 0.01,
            "chain_share_discount": 0.4,
            "chain_share_standard": 0.4,
            "chain_share_bio": 0.2,
            "bio_income_gradient": 2.5,
        }

    monkeypatch.setattr(structural, "load_berlin_city", fake_load_berlin_city)
    monkeypatch.setattr(structural, "all_model_moments", fake_all_model_moments)
    return {"build_count": lambda: build_count}


def test_calibrate_structural_recovers_mu_and_a0(
    patched_solver,
) -> None:
    targets = _fake_targets()
    env_cfg = _fake_env_cfg()

    result = structural.calibrate_structural(
        targets=targets,
        env_cfg=env_cfg,
        grid_path="unused.parquet",
        stores_path="unused.parquet",
        travel_times_path="unused.parquet",
        lambda_val=1200.0,
        x0={"mu": 3.0, "a0": -3.0},
        max_nfev=40,
    )

    rtol = 1e-3
    assert result["mu"] == pytest.approx(_TRUE_MU, rel=rtol)
    assert result["a0"] == pytest.approx(_TRUE_A0, rel=rtol)
    assert result["q_S"] == pytest.approx(6.0, rel=rtol)
    assert result["q_B"] == pytest.approx(18.0, rel=rtol)
    assert result["alpha_ratio"] == pytest.approx(2.5, rel=rtol)
    assert result["residual_norm"] == pytest.approx(0.0, abs=1e-2)
    assert patched_solver["build_count"]() == 1
