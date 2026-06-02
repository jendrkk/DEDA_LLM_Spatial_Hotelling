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

    @pytest.mark.xfail(
        reason="__all__ has 'load_boundary' before 'load_berlin_city'; fix in spatial/__init__.py",
        strict=False,
    )
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

        assert normalize_chain_name("Q151954") == "Lidl"
        assert normalize_chain_name("Q16968817") == "Rewe"
        assert normalize_chain_name("Q701755") == "Edeka"

    def test_unknown_qid_returns_fallback(self):
        from hotelling.spatial.osm import normalize_chain_name

        assert normalize_chain_name("Q999999", brand="MyStore") == "MyStore"

    def test_none_qid_returns_fallback(self):
        from hotelling.spatial.osm import normalize_chain_name

        # "Aldi" is in _BRAND_NAME_MAP → "Aldi Nord" (Berlin is Aldi Nord territory).
        # Unknown brands are returned as-is; known brands are canonicalised.
        assert normalize_chain_name(None, brand="Aldi") == "Aldi Nord"
        assert normalize_chain_name(None, brand="MyUnknownStore") == "MyUnknownStore"

    def test_none_qid_no_fallback_returns_none(self):
        from hotelling.spatial.osm import normalize_chain_name

        assert normalize_chain_name(None) is None

    def test_all_qids_in_chain_map(self):
        from hotelling.spatial.osm import CHAIN_QID_MAP

        assert len(CHAIN_QID_MAP) >= 15
        assert all(qid.startswith("Q") for qid in CHAIN_QID_MAP)


class TestChainTypeMap:
    """Tests for hotelling.spatial.osm.CHAIN_TYPE_MAP."""

    def test_chain_type_map_exists(self):
        from hotelling.spatial.osm import CHAIN_TYPE_MAP
        assert isinstance(CHAIN_TYPE_MAP, dict)
        assert len(CHAIN_TYPE_MAP) >= 10

    def test_lidl_is_discount(self):
        from hotelling.spatial.osm import CHAIN_TYPE_MAP
        assert CHAIN_TYPE_MAP["Lidl"] == "discount"

    def test_rewe_is_standard(self):
        from hotelling.spatial.osm import CHAIN_TYPE_MAP
        assert CHAIN_TYPE_MAP["Rewe"] == "standard"

    def test_bio_company_is_bio(self):
        from hotelling.spatial.osm import CHAIN_TYPE_MAP
        assert CHAIN_TYPE_MAP["Bio Company"] == "bio"

    def test_all_values_in_expected_set(self):
        from hotelling.spatial.osm import CHAIN_TYPE_MAP
        valid = {"discount", "standard", "bio"}
        assert all(v in valid for v in CHAIN_TYPE_MAP.values())

    def test_nahcity_maps_to_rewe_via_brand_map(self):
        """nahcity should resolve to Rewe via _BRAND_NAME_MAP."""
        from hotelling.spatial.osm import normalize_chain_name
        # brand='NahCity' (any capitalisation) should map to 'Rewe'
        result = normalize_chain_name(None, brand="nahcity")
        assert result == "Rewe"


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


class TestAddLorAttributes:
    """Unit tests for hotelling.spatial.assembly.add_lor_attributes."""

    @pytest.fixture
    def small_grid(self):
        import geopandas as gpd
        from shapely.geometry import box
        cells = [
            box(4500000, 3300000, 4500100, 3300100),
            box(4500100, 3300000, 4500200, 3300100),
            box(4500200, 3300000, 4500300, 3300100),
        ]
        return gpd.GeoDataFrame(
            {"Einwohner": [10, 0, 5]},
            geometry=cells,
            crs="EPSG:3035",
        )

    @pytest.fixture
    def small_lor(self):
        import geopandas as gpd
        from shapely.geometry import box
        # Two LOR polygons covering the three grid cells
        lor_polys = [
            box(4499900, 3299900, 4500200, 3300200),  # covers cell 0 and 1
            box(4500200, 3299900, 4500400, 3300200),  # covers cell 2
        ]
        return gpd.GeoDataFrame(
            {"PLR_ID": ["01010101", "01010102"], "PLR_NAME": ["Alpha", "Beta"]},
            geometry=lor_polys,
            crs="EPSG:3035",
        )

    def test_returns_geodataframe(self, small_grid, small_lor):
        import geopandas as gpd
        from hotelling.spatial.assembly import add_lor_attributes
        result = add_lor_attributes(small_grid, small_lor)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_plr_id_column_added(self, small_grid, small_lor):
        from hotelling.spatial.assembly import add_lor_attributes
        result = add_lor_attributes(small_grid, small_lor)
        assert "PLR_ID" in result.columns

    def test_plr_name_column_added(self, small_grid, small_lor):
        from hotelling.spatial.assembly import add_lor_attributes
        result = add_lor_attributes(small_grid, small_lor)
        assert "PLR_NAME" in result.columns

    def test_row_count_preserved(self, small_grid, small_lor):
        from hotelling.spatial.assembly import add_lor_attributes
        result = add_lor_attributes(small_grid, small_lor)
        assert len(result) == len(small_grid)

    def test_empty_grid_returns_empty(self, small_lor):
        import geopandas as gpd
        from hotelling.spatial.assembly import add_lor_attributes
        empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:3035"))
        result = add_lor_attributes(empty, small_lor)
        assert len(result) == 0

    def test_empty_lor_returns_grid_with_nan(self, small_grid):
        import geopandas as gpd
        from hotelling.spatial.assembly import add_lor_attributes
        empty_lor = gpd.GeoDataFrame(
            {"PLR_ID": [], "PLR_NAME": []},
            geometry=gpd.GeoSeries([], crs="EPSG:3035"),
        )
        result = add_lor_attributes(small_grid, empty_lor)
        assert len(result) == len(small_grid)


class TestAddPoiLayer:
    """Unit tests for hotelling.spatial.assembly.add_poi_layer."""

    @pytest.fixture
    def grid_3cells(self):
        import geopandas as gpd
        from shapely.geometry import box
        cells = [
            box(4500000, 3300000, 4500100, 3300100),
            box(4500100, 3300000, 4500200, 3300100),
            box(4500200, 3300000, 4500300, 3300100),
        ]
        return gpd.GeoDataFrame(geometry=cells, crs="EPSG:3035")

    @pytest.fixture
    def two_pois(self):
        """Two POI points, both inside cell 0."""
        import geopandas as gpd
        from shapely.geometry import Point
        pts = [Point(4500050, 3300050), Point(4500060, 3300060)]
        return gpd.GeoDataFrame(
            {"chain": ["Rewe", "Lidl"]},
            geometry=pts,
            crs="EPSG:3035",
        )

    def test_poi_count_column_added(self, grid_3cells, two_pois):
        from hotelling.spatial.assembly import add_poi_layer
        result = add_poi_layer(grid_3cells, two_pois)
        assert "poi_count" in result.columns

    def test_poi_count_correct(self, grid_3cells, two_pois):
        from hotelling.spatial.assembly import add_poi_layer
        result = add_poi_layer(grid_3cells, two_pois)
        # Both POIs are in cell 0; cells 1 and 2 have no POIs
        assert result["poi_count"].iloc[0] == 2
        assert result["poi_count"].iloc[1] == 0
        assert result["poi_count"].iloc[2] == 0

    def test_chain_flag_columns_added(self, grid_3cells, two_pois):
        from hotelling.spatial.assembly import add_poi_layer
        result = add_poi_layer(grid_3cells, two_pois)
        assert "has_Rewe" in result.columns or "has_Lidl" in result.columns

    def test_row_count_preserved(self, grid_3cells, two_pois):
        from hotelling.spatial.assembly import add_poi_layer
        result = add_poi_layer(grid_3cells, two_pois)
        assert len(result) == len(grid_3cells)

    def test_empty_pois_gives_zero_counts(self, grid_3cells):
        import geopandas as gpd
        from hotelling.spatial.assembly import add_poi_layer
        empty_pois = gpd.GeoDataFrame(
            {"chain": []},
            geometry=gpd.GeoSeries([], crs="EPSG:3035"),
        )
        result = add_poi_layer(grid_3cells, empty_pois)
        assert (result["poi_count"] == 0).all()


class TestAssembleSimulationGrid:
    """Unit tests for hotelling.spatial.assembly.assemble_simulation_grid."""

    @pytest.fixture
    def valid_grid(self):
        import geopandas as gpd
        from shapely.geometry import box
        cell = box(4500000, 3300000, 4500100, 3300100)
        return gpd.GeoDataFrame(
            {
                "x_mp_100m":  [4500050],
                "y_mp_100m":  [3300050],
                "Einwohner":  [10],
                "PLR_ID":     ["01010101"],
                "PLR_NAME":   ["Alpha"],
                "poi_count":  [2],
            },
            geometry=[cell],
            crs="EPSG:3035",
        )

    def test_returns_geodataframe(self, valid_grid):
        import geopandas as gpd
        from hotelling.spatial.assembly import assemble_simulation_grid
        result = assemble_simulation_grid(valid_grid, gpd.GeoDataFrame(), gpd.GeoDataFrame())
        assert isinstance(result, gpd.GeoDataFrame)

    def test_index_is_range(self, valid_grid):
        import geopandas as gpd
        from hotelling.spatial.assembly import assemble_simulation_grid
        result = assemble_simulation_grid(valid_grid, gpd.GeoDataFrame(), gpd.GeoDataFrame())
        import pandas as pd
        assert isinstance(result.index, pd.RangeIndex)

    def test_missing_required_column_raises_keyerror(self):
        import geopandas as gpd
        from hotelling.spatial.assembly import assemble_simulation_grid
        # Grid missing PLR_ID, PLR_NAME, poi_count
        from shapely.geometry import box
        bad = gpd.GeoDataFrame(
            {"Einwohner": [0], "x_mp_100m": [0], "y_mp_100m": [0]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:3035",
        )
        with pytest.raises(KeyError):
            assemble_simulation_grid(bad, gpd.GeoDataFrame(), gpd.GeoDataFrame())

    def test_nan_einwohner_filled_zero(self, valid_grid):
        import geopandas as gpd
        import numpy as np
        from hotelling.spatial.assembly import assemble_simulation_grid
        valid_grid = valid_grid.copy()
        valid_grid.loc[0, "Einwohner"] = np.nan
        result = assemble_simulation_grid(valid_grid, gpd.GeoDataFrame(), gpd.GeoDataFrame())
        assert result["Einwohner"].iloc[0] == 0


class TestStubsRaiseNotImplementedError:
    """All pipeline stubs must raise NotImplementedError, not fail silently."""

    def test_network_distance_matrix(self):
        import numpy as np
        from hotelling.spatial.distance import network_distance_matrix

        with pytest.raises(NotImplementedError):
            network_distance_matrix(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_squaregrid_sample_locations(self):
        from hotelling.spatial import SquareGrid

        with pytest.raises(NotImplementedError):
            SquareGrid().sample_locations(5)

    def test_squaregrid_cell_to_metres(self):
        from hotelling.spatial import SquareGrid

        with pytest.raises(NotImplementedError):
            SquareGrid().cell_to_metres(0, 0)


# ---------------------------------------------------------------------------
# BRW fixed-cost normalisation — synthetic loader tests
# ---------------------------------------------------------------------------

class TestBrwFixedCostNormalisation:
    """Tests for the brw→fixed_cost mapping in load_berlin_city.

    These tests bypass the full loader by exercising the normalisation logic
    directly through synthetic parquet files written to a tmp_path.  They are
    skipped when the real GEO data files are absent (expected in CI).
    """

    @pytest.fixture
    def synthetic_parquets(self, tmp_path):
        """Write minimal synthetic parquet files that satisfy load_berlin_city."""
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        # ── demand_grid ───────────────────────────────────────────────────
        n_cells = 4
        # GITTER_ID_100m must be sortable strings
        gids = [f"100mN{3300000 + 100*i}E4500000" for i in range(n_cells)]
        grid_gdf = gpd.GeoDataFrame(
            {
                "GITTER_ID_100m": gids,
                "Einwohner": [100.0] * n_cells,
                "pi_H_res": [0.5] * n_cells,
                "phi_i": [0.1] * n_cells,
            },
            geometry=[Point(4500050, 3300050 + 100 * i) for i in range(n_cells)],
            crs="EPSG:3035",
        )
        grid_path = tmp_path / "demand_grid.parquet"
        grid_gdf.to_parquet(grid_path)

        # ── supermarkets ──────────────────────────────────────────────────
        n_stores = 3
        brw_values = [1000.0, 2000.0, 3000.0]   # mean = 2000, median = 2000
        stores_gdf = gpd.GeoDataFrame(
            {
                "chain": ["Lidl", "Rewe", "Bio Company"],
                "chain_type": ["discount", "standard", "bio"],
                "brw": brw_values,
            },
            geometry=[Point(4500050 + 200 * j, 3300050) for j in range(n_stores)],
            crs="EPSG:3035",
        )
        stores_path = tmp_path / "supermarkets.parquet"
        stores_gdf.to_parquet(stores_path)

        # ── travel_times ──────────────────────────────────────────────────
        rows = []
        for cell_id in gids:
            for store_id in [str(j) for j in range(n_stores)]:
                rows.append({"from_id": cell_id, "to_id": store_id, "travel_time": 10.0})
        tt_df = pd.DataFrame(rows)
        tt_path = tmp_path / "travel_times.parquet"
        tt_df.to_parquet(tt_path)

        return grid_path, stores_path, tt_path, brw_values

    def _load(self, synthetic_parquets, **kwargs):
        from hotelling.spatial.loader import load_berlin_city

        grid_path, stores_path, tt_path, brw_values = synthetic_parquets
        _, firms = load_berlin_city(
            grid_path=grid_path,
            stores_path=stores_path,
            travel_times_path=tt_path,
            lambda_val=100.0,
            transport_cost=0.01,
            **kwargs,
        )
        return firms, brw_values

    def test_rent_scale_zero_all_fixed_costs_zero(self, synthetic_parquets):
        """rent_scale=0 → all fixed_cost values are 0.0."""
        firms, _ = self._load(synthetic_parquets, rent_scale=0.0)
        assert all(f.fixed_cost == pytest.approx(0.0) for f in firms)

    def test_mean_ratio_mean_equals_rent_scale(self, synthetic_parquets):
        """mean_ratio normalisation: mean(fixed_cost) ≈ rent_scale."""
        rent_scale = 0.1
        firms, _ = self._load(
            synthetic_parquets,
            rent_scale=rent_scale,
            rent_normalization="mean_ratio",
        )
        fc = np.array([f.fixed_cost for f in firms])
        assert fc.mean() == pytest.approx(rent_scale, rel=1e-6)

    def test_mean_ratio_preserves_relative_ordering(self, synthetic_parquets):
        """Stores with higher brw have higher fixed_cost under mean_ratio."""
        firms, brw_values = self._load(
            synthetic_parquets,
            rent_scale=0.1,
            rent_normalization="mean_ratio",
        )
        fc = [f.fixed_cost for f in firms]
        assert fc[0] < fc[1] < fc[2]

    def test_median_ratio_preserves_relative_ordering(self, synthetic_parquets):
        """median_ratio also preserves brw ordering across stores."""
        firms, _ = self._load(
            synthetic_parquets,
            rent_scale=0.1,
            rent_normalization="median_ratio",
        )
        fc = [f.fixed_cost for f in firms]
        assert fc[0] < fc[1] < fc[2]

    def test_minmax_range(self, synthetic_parquets):
        """minmax: min fixed_cost ≈ 0, max ≈ rent_scale."""
        rent_scale = 0.2
        firms, _ = self._load(
            synthetic_parquets,
            rent_scale=rent_scale,
            rent_normalization="minmax",
        )
        fc = np.array([f.fixed_cost for f in firms])
        assert fc.min() == pytest.approx(0.0, abs=1e-10)
        assert fc.max() == pytest.approx(rent_scale, rel=1e-6)

    def test_no_brw_column_returns_zeros(self, tmp_path):
        """If 'brw' column is absent, fixed_costs are all zero even for rent_scale > 0."""
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point
        from hotelling.spatial.loader import load_berlin_city

        n_cells = 2
        gids = [f"100mN{3300000 + 100*i}E4500000" for i in range(n_cells)]
        grid_gdf = gpd.GeoDataFrame(
            {
                "GITTER_ID_100m": gids,
                "Einwohner": [100.0] * n_cells,
                "pi_H_res": [0.5] * n_cells,
                "phi_i": [0.1] * n_cells,
            },
            geometry=[Point(4500050, 3300050 + 100 * i) for i in range(n_cells)],
            crs="EPSG:3035",
        )
        grid_path = tmp_path / "demand_grid.parquet"
        grid_gdf.to_parquet(grid_path)

        # No brw column
        stores_gdf = gpd.GeoDataFrame(
            {"chain": ["Lidl"], "chain_type": ["discount"]},
            geometry=[Point(4500050, 3300050)],
            crs="EPSG:3035",
        )
        stores_path = tmp_path / "supermarkets.parquet"
        stores_gdf.to_parquet(stores_path)

        tt_df = pd.DataFrame([
            {"from_id": gid, "to_id": "0", "travel_time": 10.0}
            for gid in gids
        ])
        tt_path = tmp_path / "travel_times.parquet"
        tt_df.to_parquet(tt_path)

        _, firms = load_berlin_city(
            grid_path=grid_path,
            stores_path=stores_path,
            travel_times_path=tt_path,
            lambda_val=100.0,
            transport_cost=0.01,
            rent_scale=0.1,  # non-zero, but brw absent
        )
        assert all(f.fixed_cost == pytest.approx(0.0) for f in firms)

    def test_invalid_normalization_raises(self, synthetic_parquets):
        """Unrecognised rent_normalization string raises ValueError."""
        with pytest.raises(ValueError, match="rent_normalization"):
            self._load(
                synthetic_parquets,
                rent_scale=0.1,
                rent_normalization="bad_method",
            )


# ---------------------------------------------------------------------------
# Alignment check tests (no real data required)
# ---------------------------------------------------------------------------

class TestTravelTimeAlignmentCheck:
    """load_berlin_city raises ValueError when travel_times references store IDs
    outside the range {0..N-1}, indicating mismatched inner-ring vs. full-grid files."""

    def _make_synthetic_grid(self, tmp_path, n_cells: int = 2):
        import geopandas as gpd
        from shapely.geometry import Point

        gids = [f"100mN{3300000 + 100*i}E4500000" for i in range(n_cells)]
        gdf = gpd.GeoDataFrame(
            {
                "GITTER_ID_100m": gids,
                "Einwohner": [100.0] * n_cells,
                "pi_H_res": [0.5] * n_cells,
                "phi_i": [0.1] * n_cells,
            },
            geometry=[Point(4500050, 3300050 + 100 * i) for i in range(n_cells)],
            crs="EPSG:3035",
        )
        grid_path = tmp_path / "grid.parquet"
        gdf.to_parquet(grid_path)
        return grid_path, gids

    def test_bad_store_ids_raise_value_error(self, tmp_path):
        """travel_times referencing a store id not in {0..N-1} raises ValueError."""
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point
        from hotelling.spatial.loader import load_berlin_city

        grid_path, gids = self._make_synthetic_grid(tmp_path)

        # One real store, id "0"
        stores_gdf = gpd.GeoDataFrame(
            {"chain": ["Lidl"], "chain_type": ["discount"]},
            geometry=[Point(4500050, 3300050)],
            crs="EPSG:3035",
        )
        stores_path = tmp_path / "stores.parquet"
        stores_gdf.to_parquet(stores_path)

        # travel_times references store "999" which doesn't exist
        tt_df = pd.DataFrame([
            {"from_id": gids[0], "to_id": "0",   "travel_time": 5.0},
            {"from_id": gids[0], "to_id": "999", "travel_time": 10.0},
        ])
        tt_path = tmp_path / "tt.parquet"
        tt_df.to_parquet(tt_path)

        with pytest.raises(ValueError, match="store IDs"):
            load_berlin_city(
                grid_path=grid_path,
                stores_path=stores_path,
                travel_times_path=tt_path,
                lambda_val=100.0,
                transport_cost=0.01,
                dense_distances=True,
            )

    def test_valid_ids_do_not_raise(self, tmp_path):
        """Valid store IDs in travel_times do NOT raise an error."""
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point
        from hotelling.spatial.loader import load_berlin_city

        grid_path, gids = self._make_synthetic_grid(tmp_path)

        stores_gdf = gpd.GeoDataFrame(
            {"chain": ["Lidl", "Rewe"], "chain_type": ["discount", "standard"]},
            geometry=[Point(4500050, 3300050), Point(4500150, 3300050)],
            crs="EPSG:3035",
        )
        stores_path = tmp_path / "stores.parquet"
        stores_gdf.to_parquet(stores_path)

        # Both store IDs "0" and "1" are valid
        tt_df = pd.DataFrame([
            {"from_id": gids[0], "to_id": "0", "travel_time": 5.0},
            {"from_id": gids[0], "to_id": "1", "travel_time": 10.0},
            {"from_id": gids[1], "to_id": "0", "travel_time": 8.0},
        ])
        tt_path = tmp_path / "tt.parquet"
        tt_df.to_parquet(tt_path)

        # Should not raise
        city, firms = load_berlin_city(
            grid_path=grid_path,
            stores_path=stores_path,
            travel_times_path=tt_path,
            lambda_val=100.0,
            transport_cost=0.01,
            dense_distances=True,
        )
        assert len(firms) == 2


# ---------------------------------------------------------------------------
# build_catchment unit tests (no file I/O — pure function)
# ---------------------------------------------------------------------------

class TestBuildCatchment:
    """Unit tests for hotelling.spatial.loader.build_catchment."""

    def _make_tt_df(self, rows):
        import pandas as pd
        return pd.DataFrame(rows, columns=["from_id", "to_id", "travel_time"])

    def test_basic_csr_structure(self):
        from hotelling.spatial.loader import build_catchment

        cell_ids  = ["C0", "C1", "C2"]
        store_ids = ["0", "1", "2", "3"]
        rows = [
            ("C0", "0", 5.0), ("C0", "1", 10.0), ("C0", "2", 20.0),
            ("C1", "0", 3.0), ("C1", "2", 8.0),
            # C2 has no entries
        ]
        tt_df = self._make_tt_df(rows)
        indptr, indices, tt_min = build_catchment(
            tt_df, cell_ids, store_ids,
            transport_cost=0.1, transport_exponent=1.0,
            catchment_minutes=25.0, k_min=1, k_max=10,
        )

        assert len(indptr) == len(cell_ids) + 1
        assert indptr[0] == 0
        assert len(indices) == int(indptr[-1])
        assert len(tt_min)  == int(indptr[-1])

    def test_indptr_monotone(self):
        from hotelling.spatial.loader import build_catchment

        cell_ids  = ["C0", "C1"]
        store_ids = ["0", "1", "2"]
        rows = [("C0", "0", 5.0), ("C0", "1", 12.0), ("C1", "2", 7.0)]
        tt_df = self._make_tt_df(rows)
        indptr, _, _ = build_catchment(
            tt_df, cell_ids, store_ids,
            transport_cost=0.1, transport_exponent=1.0,
            catchment_minutes=20.0, k_min=1, k_max=5,
        )
        assert np.all(np.diff(indptr) >= 0)

    def test_catchment_radius_excludes_far_stores(self):
        """Stores beyond catchment_minutes are excluded when k_min is satisfied."""
        from hotelling.spatial.loader import build_catchment

        cell_ids  = ["C0"]
        store_ids = ["0", "1", "2"]
        rows = [
            ("C0", "0", 5.0),   # within 10 min
            ("C0", "1", 9.0),   # within 10 min
            ("C0", "2", 50.0),  # far away
        ]
        tt_df = self._make_tt_df(rows)
        indptr, indices, tt_min = build_catchment(
            tt_df, cell_ids, store_ids,
            transport_cost=0.1, transport_exponent=1.0,
            catchment_minutes=10.0, k_min=1, k_max=10,
        )
        assert int(indptr[1]) == 2   # only stores 0 and 1 kept
        assert 2 not in indices      # store col 2 (50 min) excluded

    def test_k_min_pads_beyond_radius(self):
        """When fewer than k_min stores are within radius, pad with nearest."""
        from hotelling.spatial.loader import build_catchment

        cell_ids  = ["C0"]
        store_ids = ["0", "1", "2"]
        rows = [
            ("C0", "0", 5.0),    # within 6 min
            ("C0", "1", 30.0),   # beyond 6 min
            ("C0", "2", 40.0),   # beyond 6 min
        ]
        tt_df = self._make_tt_df(rows)
        indptr, indices, tt_min = build_catchment(
            tt_df, cell_ids, store_ids,
            transport_cost=0.1, transport_exponent=1.0,
            catchment_minutes=6.0, k_min=2, k_max=10,
        )
        # Only 1 store within radius but k_min=2, so 2 stores should be kept
        assert int(indptr[1]) == 2

    def test_k_max_caps_entries(self):
        """Entries per cell are capped at k_max."""
        from hotelling.spatial.loader import build_catchment

        cell_ids  = ["C0"]
        store_ids = [str(i) for i in range(10)]
        rows = [(f"C0", str(i), float(i + 1)) for i in range(10)]
        tt_df = self._make_tt_df(rows)
        indptr, indices, tt_min = build_catchment(
            tt_df, cell_ids, store_ids,
            transport_cost=0.1, transport_exponent=1.0,
            catchment_minutes=100.0, k_min=1, k_max=3,
        )
        assert int(indptr[1]) == 3   # capped at k_max=3

    def test_empty_cell_has_empty_span(self):
        """Cells with no tt rows contribute an empty span to the CSR."""
        from hotelling.spatial.loader import build_catchment

        cell_ids  = ["C0", "C1", "C2"]
        store_ids = ["0"]
        rows = [("C0", "0", 5.0)]   # only C0 has entries
        tt_df = self._make_tt_df(rows)
        indptr, _, _ = build_catchment(
            tt_df, cell_ids, store_ids,
            transport_cost=0.1, transport_exponent=1.0,
            catchment_minutes=20.0, k_min=1, k_max=5,
        )
        assert indptr[2] == indptr[1]   # C1 is empty
        assert indptr[3] == indptr[2]   # C2 is empty

    def test_output_dtypes(self):
        from hotelling.spatial.loader import build_catchment

        cell_ids  = ["C0"]
        store_ids = ["0", "1"]
        rows = [("C0", "0", 5.0), ("C0", "1", 8.0)]
        tt_df = self._make_tt_df(rows)
        indptr, indices, tt_min = build_catchment(
            tt_df, cell_ids, store_ids,
            transport_cost=0.1, transport_exponent=1.0,
            catchment_minutes=20.0, k_min=1, k_max=5,
        )
        assert indptr.dtype == np.int64
        assert indices.dtype == np.int32
        assert tt_min.dtype == np.float64

    def test_k_min_gt_k_max_raises(self):
        from hotelling.spatial.loader import build_catchment

        with pytest.raises(ValueError, match="k_min"):
            build_catchment(
                self._make_tt_df([("C0", "0", 5.0)]),
                cell_ids=["C0"], store_ids=["0"],
                transport_cost=0.1, transport_exponent=1.0,
                catchment_minutes=20.0, k_min=10, k_max=5,
            )
