"""Shared pytest fixtures for the hotelling test suite."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.core.city import City
from hotelling.core.firm import Firm


@pytest.fixture
def unit_city() -> City:
    """A simple 1×1 unit-square City with no population grid."""
    return City(boundary=(0.0, 0.0, 1.0, 1.0))


@pytest.fixture
def two_firms(unit_city: City) -> City:
    """Unit city pre-loaded with two symmetric duopoly firms."""
    firm_a = Firm(id="firm_0", location=(0.25, 0.5), marginal_cost=1.0, quality=1.0)
    firm_b = Firm(id="firm_1", location=(0.75, 0.5), marginal_cost=1.0, quality=1.0)
    unit_city.firms = [firm_a, firm_b]
    return unit_city


@pytest.fixture
def population_grid() -> np.ndarray:
    """A small 10×10 uniform population density array."""
    return np.ones((10, 10), dtype=float)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy Generator for reproducible tests."""
    return np.random.default_rng(42)
