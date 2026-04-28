"""Unit tests for the group division system and store-group assignment logic.

Verifies that CompetitionDivision correctly classifies stores by rival count,
NeighbourhoodDivision by LOR social-status index, and that activating two
divisions produces the expected four composite group labels.
"""
from __future__ import annotations

import pytest

from hotelling.envelope.divisions.competition import CompetitionDivision  # noqa: F401
from hotelling.envelope.divisions.neighbourhood import NeighbourhoodDivision  # noqa: F401
from hotelling.envelope.groups import assign_groups  # noqa: F401


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_competition_division_assigns_heavy_to_dense_store() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_competition_division_assigns_easy_to_isolated_store() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_neighbourhood_division_assigns_rich_above_threshold() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_two_divisions_produce_four_groups() -> None:
    pass
