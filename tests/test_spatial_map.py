"""Smoke tests for hotelling.viz.spatial_map.

Tests are skipped automatically when:
- No finished run directory exists under ``results/runs/``.
- contextily is not installed.
- geopandas is not installed.

contextily.add_basemap is monkeypatched to a no-op so no network requests
are made during CI.

To run locally against real data:
    pytest tests/test_spatial_map.py -v
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / skip guards
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNS_DIR = _REPO_ROOT / "results" / "runs"


def _find_run_dir() -> Path | None:
    """Return the first complete run dir (config.yaml + dense_log_meta.json)."""
    if not _RUNS_DIR.exists():
        return None
    for candidate in sorted(_RUNS_DIR.iterdir()):
        if (
            candidate.is_dir()
            and (candidate / "config.yaml").exists()
            and (candidate / "dense_log_meta.json").exists()
        ):
            return candidate
    return None


def _contextily_available() -> bool:
    try:
        import contextily  # noqa: F401

        return True
    except ImportError:
        return False


def _geopandas_available() -> bool:
    try:
        import geopandas  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_dir() -> Path:
    """Path to a complete run directory; skip module if none found."""
    rd = _find_run_dir()
    if rd is None:
        pytest.skip(
            "No complete run directory found under results/runs/ "
            "(needs config.yaml + dense_log_meta.json). "
            "Run a simulation first."
        )
    if not _contextily_available():
        pytest.skip(
            "contextily is not installed. "
            "Install with: pip install 'hotelling[viz]'"
        )
    if not _geopandas_available():
        pytest.skip(
            "geopandas is not installed. "
            "Install with: pip install 'hotelling[spatial]'"
        )
    return rd


@pytest.fixture(scope="module")
def loaded_run(run_dir: Path):
    """Load run artefacts once for the whole module (expensive I/O).

    Does NOT use contextily, so no monkeypatch needed here.
    """
    from hotelling.viz.spatial_map import load_run

    return load_run(run_dir)


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch for contextily.add_basemap → no-op.

    Skips automatically if contextily is not installed.
    """
    try:
        import contextily
    except ImportError:
        pytest.skip("contextily not installed")

    original = contextily.add_basemap

    def _noop(*args: Any, **kwargs: Any) -> None:
        pass

    contextily.add_basemap = _noop
    yield
    contextily.add_basemap = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadRun:
    """Verify that load_run returns well-formed artefacts."""

    def test_returns_six_tuple(self, loaded_run: Any) -> None:
        assert len(loaded_run) == 6

    def test_dense_log_has_data(self, loaded_run: Any) -> None:
        dense_log = loaded_run[0]
        assert dense_log._t_written > 0, "DenseLog has no written steps."
        assert dense_log.N > 0, "DenseLog has zero agents."

    def test_city_dist_matrix_shape(self, loaded_run: Any) -> None:
        dense_log, city, firms = loaded_run[:3]
        M, N = city.dist2_km2.shape
        assert N == dense_log.N, (
            f"city.dist2_km2 columns ({N}) != DenseLog.N ({dense_log.N})"
        )
        assert M > 0

    def test_grid_gdf_crs_is_3857(self, loaded_run: Any) -> None:
        grid_gdf = loaded_run[3]
        assert grid_gdf.crs is not None
        assert grid_gdf.crs.to_epsg() == 3857

    def test_stores_gdf_crs_is_3857(self, loaded_run: Any) -> None:
        stores_gdf = loaded_run[4]
        assert stores_gdf.crs is not None
        assert stores_gdf.crs.to_epsg() == 3857

    def test_grid_row_count_matches_city(self, loaded_run: Any) -> None:
        dense_log, city, firms, grid_gdf = loaded_run[:4]
        assert len(grid_gdf) == city.dist2_km2.shape[0], (
            "grid_gdf row count must equal city.dist2_km2 rows (M)."
        )

    def test_stores_row_count_matches_firms(self, loaded_run: Any) -> None:
        dense_log, city, firms, grid_gdf, stores_gdf = loaded_run[:5]
        assert len(stores_gdf) == len(firms), (
            "stores_gdf row count must equal len(city.firms) (N)."
        )


class TestPricesEffortsAt:
    """Verify prices_efforts_at returns correct shapes and ranges."""

    def test_shape(self, loaded_run: Any) -> None:
        from hotelling.viz.spatial_map import prices_efforts_at

        dense_log = loaded_run[0]
        prices, efforts = prices_efforts_at(dense_log, 0)
        assert prices.shape == (dense_log.N,)
        assert efforts.shape == (dense_log.N,)

    def test_prices_within_grid(self, loaded_run: Any) -> None:
        from hotelling.viz.spatial_map import prices_efforts_at

        dense_log = loaded_run[0]
        prices, _ = prices_efforts_at(dense_log, 0)
        assert np.all(prices >= float(dense_log.price_grid.min()) - 1e-6)
        assert np.all(prices <= float(dense_log.price_grid.max()) + 1e-6)


class TestPlotMarketSnapshot:
    """Smoke-test that plot_market_snapshot returns a Figure without error."""

    @pytest.mark.parametrize(
        "metric",
        ["expected_price", "served_demand", "dominant_chain", "consumer_surplus"],
    )
    def test_returns_figure(
        self, run_dir: Path, monkeypatch_module: Any, metric: str
    ) -> None:
        """plot_market_snapshot returns a matplotlib Figure for all metrics."""
        import matplotlib.pyplot as plt

        from hotelling.viz.spatial_map import plot_market_snapshot

        fig = plot_market_snapshot(run_dir, t=0, metric=metric)
        assert fig is not None

        # Verify it is a matplotlib Figure
        assert hasattr(fig, "savefig"), (
            f"metric={metric!r}: return value is not a Figure"
        )
        plt.close(fig)

    def test_save_path_creates_file(
        self, run_dir: Path, monkeypatch_module: Any, tmp_path: Path
    ) -> None:
        import matplotlib.pyplot as plt

        from hotelling.viz.spatial_map import plot_market_snapshot

        out = tmp_path / "snapshot.png"
        fig = plot_market_snapshot(run_dir, t=0, save_path=out)
        assert out.exists(), "save_path file was not created."
        plt.close(fig)


class TestAnimateTraining:
    """Verify animate_training writes a GIF from synthetic figures."""

    def test_writes_gif(self, tmp_path: Path) -> None:
        pytest.importorskip("imageio")
        import matplotlib.pyplot as plt

        from hotelling.viz.animation import animate_training

        figs = []
        for _ in range(3):
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.plot([0, 1], [0, 1])
            figs.append(fig)

        out = tmp_path / "test.gif"
        result = animate_training(figs, out, fps=5)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0
        for f in figs:
            plt.close(f)

    def test_raises_on_empty_list(self, tmp_path: Path) -> None:
        pytest.importorskip("imageio")
        from hotelling.viz.animation import animate_training

        with pytest.raises(ValueError, match="at least one"):
            animate_training([], tmp_path / "empty.gif")
