"""Unit tests for hotelling.spatial.osm Overpass query construction and fetch_pois."""
from __future__ import annotations

from unittest.mock import MagicMock


class TestBuildOverpassQuery:
    def test_build_overpass_query_single_block(self):
        from hotelling.spatial.osm import _build_overpass_query

        block = '["shop"="supermarket"]'
        expected = (
            "[out:json][timeout:180];\n"
            "area(42)->.searchArea;\n"
            "(\n"
            '  node["shop"="supermarket"](area.searchArea);\n'
            '  way["shop"="supermarket"](area.searchArea);\n'
            '  relation["shop"="supermarket"](area.searchArea);\n'
            ");\n"
            "out geom tags;\n"
        )
        assert _build_overpass_query(42, [block]) == expected

    def test_build_overpass_query_multiple_blocks(self):
        from hotelling.spatial.osm import _build_overpass_query

        b1 = '["shop"="mall"]'
        b2 = '["landuse"="retail"]'
        q = _build_overpass_query(99, [b1, b2], timeout=60)
        assert q.count("node") == 2
        assert q.count("way") == 2
        assert q.count("relation") == 2
        assert '[timeout:60];' in q
        assert "area(99)->.searchArea;" in q
        assert f"  node{b1}(area.searchArea);" in q
        assert f"  way{b1}(area.searchArea);" in q
        assert f"  relation{b1}(area.searchArea);" in q
        assert f"  node{b2}(area.searchArea);" in q
        assert f"  way{b2}(area.searchArea);" in q
        assert f"  relation{b2}(area.searchArea);" in q


def test_fetch_pois_tags_list_form(tmp_path):
    from unittest.mock import patch

    from hotelling.spatial.osm import fetch_pois

    captured: dict[str, bytes] = {}

    def fake_post(url, data, timeout, max_attempts=3):
        captured["data"] = data
        r = MagicMock()
        r.status_code = 200
        r.raise_for_status = MagicMock()
        r.json.return_value = {"elements": []}
        return r

    with patch("hotelling.spatial.osm._post_with_retry", side_effect=fake_post), patch(
        "hotelling.spatial.osm._get_area_id", return_value=3600062422
    ):
        gdf = fetch_pois(
            city="TestCity",
            tags=[{"shop": "mall"}, {"landuse": "retail"}],
            name="test",
            cache_dir=tmp_path,
            timeout=90,
        )

    assert gdf.crs.to_string() == "EPSG:4326"
    query = captured["data"].decode("utf-8")
    assert '["shop"="mall"]' in query
    assert '["landuse"="retail"]' in query
    assert query.count('["shop"="mall"]') >= 3
    assert query.count('["landuse"="retail"]') >= 3
