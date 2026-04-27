"""Unit tests for hotelling.core – City and Firm."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.core.city import City
from hotelling.core.firm import Firm


class TestCity:
    def test_default_boundary(self):
        city = City()
        assert city.boundary == (0.0, 0.0, 1.0, 1.0)

    def test_width_height(self):
        city = City(boundary=(0.0, 0.0, 2.0, 3.0))
        assert city.width == pytest.approx(2.0)
        assert city.height == pytest.approx(3.0)

    def test_area(self):
        city = City(boundary=(0.0, 0.0, 4.0, 5.0))
        assert city.area == pytest.approx(20.0)

    def test_center(self):
        city = City(boundary=(0.0, 0.0, 4.0, 6.0))
        cx, cy = city.center
        assert cx == pytest.approx(2.0)
        assert cy == pytest.approx(3.0)

    def test_firms_default_empty(self):
        city = City()
        assert city.firms == []

    def test_population_grid_default_none(self):
        city = City()
        assert city.population_grid is None

    def test_population_grid_stored(self):
        grid = np.ones((10, 10))
        city = City(population_grid=grid)
        assert city.population_grid is not None
        assert city.population_grid.shape == (10, 10)

    def test_firms_mutable(self, unit_city):
        firm = Firm(id="A", location=(0.5, 0.5))
        unit_city.firms.append(firm)
        assert len(unit_city.firms) == 1


class TestFirm:
    def test_defaults(self):
        firm = Firm(id="X")
        assert firm.location == (0.5, 0.5)
        assert firm.marginal_cost == pytest.approx(0.0)
        assert firm.quality == pytest.approx(1.0)
        assert firm.chain is None

    def test_id_required(self):
        with pytest.raises(TypeError):
            Firm()  # type: ignore[call-arg]

    def test_immutable(self):
        firm = Firm(id="X")
        with pytest.raises((AttributeError, TypeError, Exception)):
            firm.id = "Y"  # frozen dataclass should not allow mutation

    def test_custom_values(self):
        firm = Firm(id="A", location=(0.1, 0.9), marginal_cost=0.5, quality=2.0, chain="Rewe")
        assert firm.id == "A"
        assert firm.location == (0.1, 0.9)
        assert firm.marginal_cost == pytest.approx(0.5)
        assert firm.quality == pytest.approx(2.0)
        assert firm.chain == "Rewe"
