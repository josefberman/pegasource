"""
Tests for pegasource.geo — distance, projection, and vectorizer functions.
"""

import math
import numpy as np
import pytest

from pegasource.geo.distance import haversine, vincenty, bearing
from pegasource.geo.projection import wgs84_to_itm, itm_to_wgs84, wgs84_to_utm, meters_offset
from pegasource.geo.vectorizer import build_graph, compute_road_coverage


# ---------------------------------------------------------------------------
# Distance tests
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point(self):
        assert haversine(32.0, 34.8, 32.0, 34.8) == pytest.approx(0.0, abs=1e-3)

    def test_tlv_to_jerusalem(self):
        # Tel Aviv Yafo → Jerusalem (~54 km)
        dist = haversine(32.0853, 34.7818, 31.7683, 35.2137)
        assert 53_000 < dist < 56_000, f"Expected ~54 km, got {dist:.0f} m"

    def test_symmetry(self):
        d1 = haversine(32.0, 34.8, 31.5, 35.0)
        d2 = haversine(31.5, 35.0, 32.0, 34.8)
        assert d1 == pytest.approx(d2, rel=1e-6)

    def test_positive_distance(self):
        assert haversine(31.0, 34.0, 33.0, 36.0) > 0


class TestVincenty:
    def test_same_point(self):
        assert vincenty(32.0, 34.8, 32.0, 34.8) == pytest.approx(0.0, abs=1e-3)

    def test_close_to_haversine(self):
        d_h = haversine(32.0853, 34.7818, 31.7683, 35.2137)
        d_v = vincenty(32.0853, 34.7818, 31.7683, 35.2137)
        # Should agree within 0.1%
        assert abs(d_h - d_v) / d_v < 0.001


class TestBearing:
    def test_north(self):
        # Moving straight north → bearing 0°
        b = bearing(30.0, 35.0, 31.0, 35.0)
        assert b == pytest.approx(0.0, abs=0.5)

    def test_east(self):
        # Moving east → bearing ~90°
        b = bearing(32.0, 34.0, 32.0, 36.0)
        assert b == pytest.approx(90.0, abs=1.0)

    def test_range(self):
        b = bearing(30.0, 34.0, 33.0, 36.0)
        assert 0.0 <= b < 360.0


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------

class TestITMRoundtrip:
    def test_jerusalem_roundtrip(self):
        lat, lon = 31.7683, 35.2137
        e, n = wgs84_to_itm(lat, lon)
        lat2, lon2 = itm_to_wgs84(e, n)
        assert lat2 == pytest.approx(lat, abs=1e-5)
        assert lon2 == pytest.approx(lon, abs=1e-5)

    def test_tlv_roundtrip(self):
        lat, lon = 32.0853, 34.7818
        e, n = wgs84_to_itm(lat, lon)
        lat2, lon2 = itm_to_wgs84(e, n)
        assert lat2 == pytest.approx(lat, abs=1e-5)
        assert lon2 == pytest.approx(lon, abs=1e-5)

    def test_itm_easting_range(self):
        # Israel is roughly easting 100k–300k in ITM
        e, n = wgs84_to_itm(32.0, 35.0)
        assert 100_000 < e < 300_000
        assert 500_000 < n < 800_000


class TestUTM:
    def test_israel_zone(self):
        # Israel is in UTM zone 36N
        e, n, zone = wgs84_to_utm(32.0, 35.0)
        assert zone == 36

    def test_easting_reasonable(self):
        e, n, zone = wgs84_to_utm(32.0, 35.0)
        assert 100_000 < e < 900_000


class TestMetersOffset:
    def test_north_offset(self):
        lat, lon = 32.0, 35.0
        lat2, lon2 = meters_offset(lat, lon, dx_m=0, dy_m=1000)
        assert lat2 > lat
        assert lon2 == pytest.approx(lon, abs=1e-6)

    def test_east_offset(self):
        lat, lon = 32.0, 35.0
        lat2, lon2 = meters_offset(lat, lon, dx_m=1000, dy_m=0)
        assert lon2 > lon
        assert lat2 == pytest.approx(lat, abs=1e-6)

    def test_1km_distance(self):
        """Moving 1 km north should result in ~1 km distance."""
        from pegasource.geo.distance import haversine
        lat, lon = 32.0, 35.0
        lat2, lon2 = meters_offset(lat, lon, dx_m=0, dy_m=1000)
        d = haversine(lat, lon, lat2, lon2)
        assert d == pytest.approx(1000, rel=0.01)


# ---------------------------------------------------------------------------
# Vectorizer tests
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def _make_cross(self, size=20):
        """Create a plus/cross shaped density map for testing."""
        d = np.zeros((size, size))
        mid = size // 2
        d[mid, :] = 1.0   # horizontal bar
        d[:, mid] = 1.0   # vertical bar
        return d

    def test_returns_graph(self):
        import networkx as nx
        d = self._make_cross()
        G = build_graph(d)
        assert isinstance(G, nx.Graph)

    def test_has_nodes_and_edges(self):
        d = self._make_cross()
        G = build_graph(d)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_edge_attributes(self):
        d = self._make_cross()
        G = build_graph(d)
        for u, v, data in G.edges(data=True):
            assert "weight" in data
            assert "length" in data
            assert "path" in data

    def test_empty_map_returns_empty_graph(self):
        d = np.zeros((10, 10))
        G = build_graph(d)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0


class TestRoadCoverage:
    def test_full_coverage(self):
        d = np.zeros((20, 20))
        d[10, :] = 1.0
        G = build_graph(d)
        result = compute_road_coverage(G, G)
        assert result["coverage"] == pytest.approx(1.0)

    def test_partial_coverage(self):
        d_full = np.zeros((30, 30))
        d_full[15, :] = 1.0
        d_full[:, 15] = 1.0

        d_partial = np.zeros((30, 30))
        d_partial[15, :] = 1.0   # only horizontal bar

        G_full = build_graph(d_full)
        G_partial = build_graph(d_partial)

        result = compute_road_coverage(G_full, G_partial)
        assert 0.0 < result["coverage"] < 1.0
        assert "edges" in result
