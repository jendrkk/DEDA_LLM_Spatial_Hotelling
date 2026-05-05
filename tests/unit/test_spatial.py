"""Unit tests for hotelling.spatial – SquareGrid."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.spatial import SquareGrid


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


class TestSpatialPublicAPI:
    """Verify __all__ and lazy-loader consistency, no GIS deps needed."""

    def test_all_is_sorted(self):
        from hotelling.spatial import __all__ as spatial_all
        assert spatial_all == sorted(spatial_all), "__all__ must be alphabetically sorted"

    def test_lazy_geo_keys_match_all(self):
        import importlib
        import hotelling.spatial as sp

        src = importlib.import_module("hotelling.spatial")
        lazy_keys = set(src._LAZY_GEO.keys())
        # Every lazy key must be in __all__
        for key in lazy_keys:
            assert key in sp.__all__, f"_LAZY_GEO key '{key}' not listed in __all__"

    def test_dir_returns_all(self):
        import hotelling.spatial as sp

        d = dir(sp)
        for name in sp.__all__:
            assert name in d, f"'{name}' missing from dir(hotelling.spatial)"

    def test_unknown_attribute_raises(self):
        import hotelling.spatial as sp

        with pytest.raises(AttributeError):
            _ = sp.this_does_not_exist_xyz


class TestNormalizeChainName:
    """Tests for hotelling.spatial.osm.normalize_chain_name."""

    def test_known_qid_returns_canonical(self):
        from hotelling.spatial.osm import normalize_chain_name

        assert normalize_chain_name("Q151954") == "Rewe"
        assert normalize_chain_name("Q700965") == "Lidl"
        assert normalize_chain_name("Q685967") == "Edeka"

    def test_unknown_qid_returns_fallback(self):
        from hotelling.spatial.osm import normalize_chain_name

        assert normalize_chain_name("Q999999", fallback_name="MyStore") == "MyStore"

    def test_none_qid_returns_fallback(self):
        from hotelling.spatial.osm import normalize_chain_name

        assert normalize_chain_name(None, fallback_name="Aldi") == "Aldi"

    def test_none_qid_no_fallback_returns_none(self):
        from hotelling.spatial.osm import normalize_chain_name

        assert normalize_chain_name(None) is None

    def test_all_qids_in_chain_map(self):
        from hotelling.spatial.osm import CHAIN_QID_MAP

        assert len(CHAIN_QID_MAP) >= 10
        assert all(qid.startswith("Q") for qid in CHAIN_QID_MAP)


class TestBuildTagFilters:
    """Tests for hotelling.spatial.osm._build_tag_filters."""

    def test_exact_match(self):
        from hotelling.spatial.osm import _build_tag_filters

        result = _build_tag_filters({"shop": "supermarket"})
        assert result == '["shop"="supermarket"]'

    def test_list_match_produces_regex(self):
        from hotelling.spatial.osm import _build_tag_filters

        result = _build_tag_filters({"shop": ["supermarket", "convenience"]})
        assert "~" in result
        assert "supermarket" in result
        assert "convenience" in result

    def test_true_value_produces_existence_check(self):
        from hotelling.spatial.osm import _build_tag_filters

        result = _build_tag_filters({"healthcare": True})
        assert result == '["healthcare"]'

    def test_multiple_keys(self):
        from hotelling.spatial.osm import _build_tag_filters

        result = _build_tag_filters({"shop": "supermarket", "opening_hours": True})
        assert '["shop"="supermarket"]' in result
        assert '["opening_hours"]' in result


class TestLoadBoundary:
    """Tests for hotelling.spatial.boundaries.load_boundary."""

    def _write_feature_geojson(self, tmp_path, crs="EPSG:3035"):
        import json

        feature = {
            "type": "Feature",
            "properties": {"crs": crs, "city_name": "Test"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        }
        path = tmp_path / "test_boundary.geojson"
        path.write_text(json.dumps(feature))
        return path

    def test_loads_feature_geojson(self, tmp_path):
        from hotelling.spatial.boundaries import load_boundary

        path = self._write_feature_geojson(tmp_path)
        gdf = load_boundary(path)
        assert len(gdf) == 1
        assert gdf.crs.to_epsg() == 3035

    def test_returns_geodataframe(self, tmp_path):
        import geopandas as gpd
        from hotelling.spatial.boundaries import load_boundary

        path = self._write_feature_geojson(tmp_path)
        result = load_boundary(path)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_reads_crs_from_properties(self, tmp_path):
        from hotelling.spatial.boundaries import load_boundary

        path = self._write_feature_geojson(tmp_path, crs="EPSG:3035")
        gdf = load_boundary(path)
        assert gdf.crs.to_epsg() == 3035


class TestBuildFullGrid:
    """Tests for hotelling.spatial.census.build_full_grid with synthetic data."""

    @pytest.fixture
    def tiny_boundary(self):
        """A small square boundary in EPSG:3035."""
        import geopandas as gpd
        from shapely.geometry import box

        # 500m x 500m box in EPSG:3035 near Berlin (arbitrary coords)
        geom = box(4500000, 3300000, 4500500, 3300500)
        return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:3035")

    @pytest.fixture
    def tiny_zensus(self):
        """A handful of census points on a 100m aligned grid inside the boundary."""
        import geopandas as gpd
        import pandas as pd

        # Use realistic EPSG:3035 coords that are multiples of 100 + offset 50
        # (Zensus midpoints are at X = 50 + 100k)
        xs = [4500050, 4500150, 4500250, 4500350, 4500450]
        ys = [3300050, 3300150, 3300250, 3300350, 3300450]
        rows = []
        for x in xs:
            for y in ys:
                rows.append({"x_mp_100m": x, "y_mp_100m": y, "Einwohner": 10})
        df = pd.DataFrame(rows)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["x_mp_100m"], df["y_mp_100m"]),
            crs="EPSG:3035",
        )
        return gdf

    def test_returns_geodataframe(self, tiny_boundary, tiny_zensus):
        import geopandas as gpd
        from hotelling.spatial.census import build_full_grid

        result = build_full_grid(tiny_boundary, tiny_zensus)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_required_columns_present(self, tiny_boundary, tiny_zensus):
        from hotelling.spatial.census import build_full_grid

        result = build_full_grid(tiny_boundary, tiny_zensus)
        assert "x_mp_100m" in result.columns
        assert "y_mp_100m" in result.columns
        assert "Einwohner" in result.columns
        assert "geometry" in result.columns

    def test_all_cells_inside_boundary(self, tiny_boundary, tiny_zensus):
        from hotelling.spatial.census import build_full_grid

        result = build_full_grid(tiny_boundary, tiny_zensus)
        boundary_geom = tiny_boundary.geometry.unary_union
        assert result.geometry.within(boundary_geom.buffer(1)).all()

    def test_no_negative_einwohner(self, tiny_boundary, tiny_zensus):
        from hotelling.spatial.census import build_full_grid

        result = build_full_grid(tiny_boundary, tiny_zensus)
        assert (result["Einwohner"] >= 0).all()

    def test_empty_zensus_raises(self, tiny_boundary):
        import geopandas as gpd
        import pandas as pd
        from hotelling.spatial.census import build_full_grid

        empty = gpd.GeoDataFrame(
            {
                "x_mp_100m": pd.Series([], dtype=int),
                "y_mp_100m": pd.Series([], dtype=int),
                "Einwohner": pd.Series([], dtype=int),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:3035"),
        )
        with pytest.raises(ValueError, match="non-empty"):
            build_full_grid(tiny_boundary, empty)

    def test_populated_cells_match_zensus(self, tiny_boundary, tiny_zensus):
        from hotelling.spatial.census import build_full_grid

        result = build_full_grid(tiny_boundary, tiny_zensus)
        # All zensus points are inside the boundary, so populated cells == len(tiny_zensus)
        assert (result["Einwohner"] > 0).sum() == len(tiny_zensus)


class TestStubsRaiseNotImplementedError:
    """All pipeline stubs must raise NotImplementedError, not fail silently."""

    def test_euclidean_distance_matrix(self):
        import numpy as np
        from hotelling.spatial.distance import euclidean_distance_matrix

        with pytest.raises(NotImplementedError):
            euclidean_distance_matrix(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_network_distance_matrix(self):
        import numpy as np
        from hotelling.spatial.distance import network_distance_matrix

        with pytest.raises(NotImplementedError):
            network_distance_matrix(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_build_grid_polygons(self):
        import geopandas as gpd
        from hotelling.spatial.census import build_grid_polygons

        with pytest.raises(NotImplementedError):
            build_grid_polygons(gpd.GeoDataFrame())

    def test_clip_grid_to_boundary(self):
        import geopandas as gpd
        from hotelling.spatial.census import clip_grid_to_boundary

        with pytest.raises(NotImplementedError):
            clip_grid_to_boundary(gpd.GeoDataFrame(), gpd.GeoDataFrame())

    def test_load_lor(self):
        from hotelling.spatial.admin import load_lor

        with pytest.raises(NotImplementedError):
            load_lor()

    def test_select_ringbahn_lor(self):
        import geopandas as gpd
        from hotelling.spatial.admin import select_ringbahn_lor

        with pytest.raises(NotImplementedError):
            select_ringbahn_lor(gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame())

    def test_add_lor_attributes(self):
        import geopandas as gpd
        from hotelling.spatial.assembly import add_lor_attributes

        with pytest.raises(NotImplementedError):
            add_lor_attributes(gpd.GeoDataFrame(), gpd.GeoDataFrame())

    def test_add_poi_layer(self):
        import geopandas as gpd
        from hotelling.spatial.assembly import add_poi_layer

        with pytest.raises(NotImplementedError):
            add_poi_layer(gpd.GeoDataFrame(), gpd.GeoDataFrame())

    def test_assemble_simulation_grid(self):
        import geopandas as gpd
        from hotelling.spatial.assembly import assemble_simulation_grid

        with pytest.raises(NotImplementedError):
            assemble_simulation_grid(gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame())

    def test_process_ihk_data(self):
        from pathlib import Path

        import geopandas as gpd
        from hotelling.spatial.city_data import process_ihk_data

        with pytest.raises(NotImplementedError):
            process_ihk_data(gpd.GeoDataFrame(), Path("dummy.csv"))

    def test_process_esix_mss_data(self):
        import geopandas as gpd
        from hotelling.spatial.city_data import process_esix_mss_data

        with pytest.raises(NotImplementedError):
            process_esix_mss_data(gpd.GeoDataFrame())

    def test_identify_transport_hubs(self):
        import geopandas as gpd
        from hotelling.spatial.city_data import identify_transport_hubs

        with pytest.raises(NotImplementedError):
            identify_transport_hubs(gpd.GeoDataFrame())

    def test_identify_cbd(self):
        import geopandas as gpd
        from hotelling.spatial.city_data import identify_cbd

        with pytest.raises(NotImplementedError):
            identify_cbd(gpd.GeoDataFrame())

    def test_squaregrid_sample_locations(self):
        from hotelling.spatial import SquareGrid

        with pytest.raises(NotImplementedError):
            SquareGrid().sample_locations(5)

    def test_squaregrid_cell_to_metres(self):
        from hotelling.spatial import SquareGrid

        with pytest.raises(NotImplementedError):
            SquareGrid().cell_to_metres(0, 0)
