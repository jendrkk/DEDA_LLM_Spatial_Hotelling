"""Unit tests for data-only structural calibration functions."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hotelling.calibration.structural import compute_marginal_costs, compute_transport_cost

_TARGETS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "calibration" / "targets.yaml"
)


def _load_targets() -> dict:
    with _TARGETS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_transport_cost_matches_hand_calc():
    t = compute_transport_cost(
        wage_monthly_gross_eur=3955.0,
        work_hours_per_month=167.0,
        vtt_wage_ratio=0.5,
        round_trip_factor=2.0,
    )
    expected = 2 * 0.5 * (3955 / 167) / 60
    assert t == pytest.approx(expected, rel=1e-9)
    assert 0.3 < t < 0.5


def test_marginal_cost_ordering():
    targets = _load_targets()
    c = compute_marginal_costs(
        basket_price_standard_eur=targets["basket_price_standard_eur"],
        price_index=targets["price_index"],
        gross_margin_common=targets["gross_margin_common"],
        gross_margin_by_chain=targets["gross_margin_by_chain"],
        use_common_margin=True,
    )
    assert c["discount"] < c["standard"] < c["bio"]
    assert c["standard"] == pytest.approx(40.0 * 1.00 * (1 - 0.22))


def test_marginal_cost_chain_specific():
    targets = _load_targets()
    c = compute_marginal_costs(
        basket_price_standard_eur=targets["basket_price_standard_eur"],
        price_index=targets["price_index"],
        gross_margin_common=targets["gross_margin_common"],
        gross_margin_by_chain=targets["gross_margin_by_chain"],
        use_common_margin=False,
    )
    assert c["discount"] < c["standard"] < c["bio"]


def test_marginal_cost_raises_on_bad_ordering():
    with pytest.raises(ValueError):
        compute_marginal_costs(
            basket_price_standard_eur=40.0,
            price_index={"discount": 1.30, "standard": 1.00, "bio": 0.85},
            gross_margin_common=0.22,
            gross_margin_by_chain={
                "discount": 0.18,
                "standard": 0.24,
                "bio": 0.30,
            },
            use_common_margin=True,
        )
