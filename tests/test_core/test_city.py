"""Tests for hotelling.core – City, Firm, Consumer, Market."""

import pytest
import numpy as np

from hotelling.core.city import City
from hotelling.core.consumer import Consumer
from hotelling.core.firm import Firm
from hotelling.core.market import Market


# ---------------------------------------------------------------------------
# City
# ---------------------------------------------------------------------------


class TestCity:
    def test_default_city(self):
        city = City()
        assert city.width == 1.0
        assert city.height == 1.0
        assert city.n_consumers == 1000
        assert city.consumer_locations.shape == (1000, 2)

    def test_consumer_locations_within_bounds(self):
        city = City(width=2.0, height=3.0, n_consumers=500, seed=0)
        locs = city.consumer_locations
        assert np.all(locs[:, 0] >= 0.0) and np.all(locs[:, 0] <= 2.0)
        assert np.all(locs[:, 1] >= 0.0) and np.all(locs[:, 1] <= 3.0)

    def test_area(self):
        city = City(width=2.0, height=5.0)
        assert city.area == pytest.approx(10.0)

    def test_center(self):
        city = City(width=4.0, height=6.0)
        assert city.center == (2.0, 3.0)

    def test_is_valid_location(self):
        city = City(width=1.0, height=1.0)
        assert city.is_valid_location(0.5, 0.5) is True
        assert city.is_valid_location(1.5, 0.5) is False
        assert city.is_valid_location(-0.1, 0.5) is False

    def test_random_location_within_bounds(self):
        city = City(width=1.0, height=1.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            x, y = city.random_location(rng)
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_reproducibility(self):
        c1 = City(seed=123)
        c2 = City(seed=123)
        np.testing.assert_array_equal(c1.consumer_locations, c2.consumer_locations)


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------


class TestConsumer:
    def test_distance_to(self):
        consumer = Consumer(location=(0.0, 0.0))
        assert consumer.distance_to(3.0, 4.0) == pytest.approx(5.0)

    def test_total_cost(self):
        consumer = Consumer(location=(0.0, 0.0), transport_cost=1.0)
        firm = Firm(firm_id="A", location=(3.0, 4.0), price=2.0)
        assert consumer.total_cost(firm) == pytest.approx(7.0)  # 2 + 5

    def test_preferred_firm_empty(self):
        consumer = Consumer(location=(0.0, 0.0))
        assert consumer.preferred_firm([]) is None

    def test_preferred_firm_chooses_cheapest_delivered(self):
        consumer = Consumer(location=(0.0, 0.0), transport_cost=1.0)
        firm_far_cheap = Firm(firm_id="far", location=(10.0, 0.0), price=0.0)
        firm_near_expensive = Firm(firm_id="near", location=(1.0, 0.0), price=0.5)
        # Total cost far: 0 + 10 = 10; near: 0.5 + 1 = 1.5
        assert consumer.preferred_firm([firm_far_cheap, firm_near_expensive]).firm_id == "near"


# ---------------------------------------------------------------------------
# Firm
# ---------------------------------------------------------------------------


class TestFirm:
    def test_defaults(self):
        firm = Firm(firm_id="X")
        assert firm.location == (0.5, 0.5)
        assert firm.price == 1.0
        assert firm.market_share == 0.0
        assert firm.name == "X"

    def test_custom_name(self):
        firm = Firm(firm_id="X", name="MyFirm")
        assert firm.name == "MyFirm"

    def test_decide_location_default(self):
        firm = Firm(firm_id="X", location=(0.3, 0.7))
        assert firm.decide_location(None, None) == (0.3, 0.7)

    def test_decide_price_default(self):
        firm = Firm(firm_id="X", price=2.5)
        assert firm.decide_price(None, None) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


class TestMarket:
    def _make_market(self, n_consumers=200, seed=0):
        city = City(n_consumers=n_consumers, seed=seed)
        firm_a = Firm(firm_id="A", location=(0.25, 0.5), price=1.0)
        firm_b = Firm(firm_id="B", location=(0.75, 0.5), price=1.0)
        market = Market(city=city, firms=[firm_a, firm_b], transport_cost=1.0)
        return market

    def test_resolve_shares_sum_to_one(self):
        market = self._make_market()
        shares = market.resolve()
        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-6)

    def test_resolve_symmetric_equal_shares(self):
        """Two symmetric firms should get approximately equal market shares."""
        market = self._make_market(n_consumers=2000, seed=42)
        shares = market.resolve()
        assert shares["A"] == pytest.approx(shares["B"], abs=0.05)

    def test_add_remove_firm(self):
        city = City(n_consumers=100, seed=0)
        market = Market(city=city, firms=[])
        assert len(market.firms) == 0

        firm = Firm(firm_id="C")
        market.add_firm(firm)
        assert len(market.firms) == 1

        market.remove_firm("C")
        assert len(market.firms) == 0

    def test_empty_market_resolve(self):
        city = City(n_consumers=100, seed=0)
        market = Market(city=city, firms=[])
        shares = market.resolve()
        assert shares == {}

    def test_market_share_stored_on_firm(self):
        market = self._make_market()
        market.resolve()
        for firm in market.firms:
            assert 0.0 <= firm.market_share <= 1.0
