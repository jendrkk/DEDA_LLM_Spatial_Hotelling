"""Unit tests for data-only quality calibration from the price ladder."""
from __future__ import annotations

import pytest

from hotelling.calibration.structural import compute_qualities


def test_compute_qualities() -> None:
    q_S, q_B = compute_qualities(
        basket_price_standard_eur=40.0,
        price_index={"discount": 0.85, "standard": 1.0, "bio": 1.3},
    )
    assert q_S == pytest.approx(6.0)
    assert q_B == pytest.approx(18.0)
    assert q_S < q_B
