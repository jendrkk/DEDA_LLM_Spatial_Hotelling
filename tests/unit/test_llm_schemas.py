"""Unit tests for hotelling.llm – schemas and client skeleton."""
from __future__ import annotations

import pytest

from hotelling.llm.schemas import EntryDecision, FirmDecision


class TestFirmDecision:
    def test_valid(self):
        d = FirmDecision(price_index=7, rationale="test")
        assert d.price_index == 7
        assert d.rationale == "test"

    def test_price_index_zero(self):
        d = FirmDecision(price_index=0)
        assert d.price_index == 0

    def test_negative_price_index_raises(self):
        with pytest.raises(Exception):
            FirmDecision(price_index=-1)

    def test_default_rationale(self):
        d = FirmDecision(price_index=3)
        assert d.rationale == ""


class TestEntryDecision:
    def test_valid(self):
        d = EntryDecision(location_index=2, price_index=5, rationale="enter here")
        assert d.location_index == 2
        assert d.price_index == 5

    def test_negative_location_raises(self):
        with pytest.raises(Exception):
            EntryDecision(location_index=-1, price_index=0)

    def test_negative_price_index_raises(self):
        with pytest.raises(Exception):
            EntryDecision(location_index=0, price_index=-1)

    def test_default_rationale(self):
        d = EntryDecision(location_index=1, price_index=1)
        assert d.rationale == ""
