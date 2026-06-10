"""Regression tests for Firm.chain_type."""
from __future__ import annotations

from hotelling.core.firm import Firm


def test_firm_accepts_chain_type() -> None:
    firm = Firm(
        id="0",
        location=(0.0, 0.0),
        marginal_cost=0.0,
        quality=0.0,
        kappa0=1.0,
        size=1.0,
        rent=0.0,
        chain_type="bio",
    )
    assert firm.chain_type == "bio"


def test_firm_chain_type_default_none() -> None:
    firm = Firm(
        id="0",
        location=(0.0, 0.0),
        marginal_cost=0.0,
        quality=0.0,
        kappa0=1.0,
        size=1.0,
        rent=0.0,
    )
    assert firm.chain_type is None
