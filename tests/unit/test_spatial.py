"""Unit tests for hotelling.spatial – SquareGrid."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.spatial.grid import SquareGrid


class TestSquareGrid:
    def test_default_shape(self):
        grid = SquareGrid()
        assert grid.width == 50
        assert grid.height == 50

    def test_uniform_population_default(self):
        grid = SquareGrid(width=5, height=4)
        assert grid.population is not None
        assert grid.population.shape == (4, 5)
        np.testing.assert_array_equal(grid.population, np.ones((4, 5)))

    def test_total_population(self):
        grid = SquareGrid(width=3, height=3)
        assert grid.total_population() == pytest.approx(9.0)

    def test_total_population_custom(self):
        pop = np.array([[1.0, 2.0], [3.0, 4.0]])
        grid = SquareGrid(width=2, height=2, population=pop)
        assert grid.total_population() == pytest.approx(10.0)

    def test_cell_size_stored(self):
        grid = SquareGrid(cell_size=50.0)
        assert grid.cell_size == pytest.approx(50.0)
