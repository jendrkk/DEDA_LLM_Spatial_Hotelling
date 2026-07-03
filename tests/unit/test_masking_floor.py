"""Unit tests for the ≥2-action floor in build_action_mask_and_epsilon.

Grid: np.linspace(50, 90, 11) → step = 4.0 €
  indices: 0→50, 1→54, 2→58, 3→62, 4→66, 5→70, 6→74, 7→78, 8→82, 9→86, 10→90
"""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.envelope.masking import (
    _allowed_grid_indices_asym,
    build_action_mask_and_epsilon,
)
from hotelling.llm.schemas import ChainEnvelopeOutput, GroupEnvelope


GRID = np.linspace(50.0, 90.0, 11)  # step 4.0 €
EFFORT_GRID = np.array([0.0])


def _env(p_bar: float, dp_minus: float, dp_plus: float, eps: float = 0.05) -> ChainEnvelopeOutput:
    return ChainEnvelopeOutput(
        chain_id="A", epoch=0, deliberation="test", rationale="test",
        groups={"default": GroupEnvelope(
            p_bar=p_bar, dp_minus=dp_minus, dp_plus=dp_plus, epsilon=eps,
        )},
    )


def _mask_row(p_bar: float, dp_minus: float, dp_plus: float) -> np.ndarray:
    env = _env(p_bar, dp_minus, dp_plus)
    mask, _ = build_action_mask_and_epsilon(
        {"A": env}, ["A"], ["default"], GRID, EFFORT_GRID, 1, False,
    )
    return mask[0]


class TestMinTwoActionFloor:
    def test_narrow_band_yields_at_least_two_actions(self):
        """Band (±0.5 €) < one step (4 €) → still ≥ 2 allowed columns."""
        row = _mask_row(p_bar=60.0, dp_minus=0.5, dp_plus=0.5)
        assert row.sum() >= 2

    def test_normal_band_yields_at_least_two_actions(self):
        """Band (±5 €) > one step → should include ≥ 2 grid points naturally."""
        row = _mask_row(p_bar=60.0, dp_minus=5.0, dp_plus=5.0)
        assert row.sum() >= 2

    def test_shifted_centre_changes_allowed_set(self):
        """Shifting p_bar by one full step (4 €) must change the allowed index set."""
        row_60 = _mask_row(p_bar=60.0, dp_minus=0.5, dp_plus=0.5)
        row_64 = _mask_row(p_bar=64.0, dp_minus=0.5, dp_plus=0.5)
        idx_60 = set(np.nonzero(row_60)[0].tolist())
        idx_64 = set(np.nonzero(row_64)[0].tolist())
        assert idx_60 != idx_64, (
            f"p_bar shift of one grid step did not change allowed indices: "
            f"{idx_60} vs {idx_64}"
        )

    def test_asymmetric_band_respected_when_wide_enough(self):
        """Asymmetric band ±8/±2: lower side spans 2 steps, upper only 0.5."""
        row = _mask_row(p_bar=66.0, dp_minus=8.0, dp_plus=2.0)
        allowed_prices = GRID[np.nonzero(row)[0]]
        # All allowed prices must be within [66-8, 66+2] = [58, 68]
        assert allowed_prices.min() >= 58.0 - 1e-9
        assert allowed_prices.max() <= 68.0 + 1e-9
        assert row.sum() >= 2


class TestAllowedGridIndicesAsym:
    def test_returns_at_least_two_interior(self):
        idx = _allowed_grid_indices_asym(GRID, 60.0, 0.1, 0.1)
        assert idx.size >= 2

    def test_returns_at_least_two_at_left_edge(self):
        idx = _allowed_grid_indices_asym(GRID, 50.0, 0.1, 0.1)
        assert idx.size >= 2
        assert 0 in idx

    def test_returns_at_least_two_at_right_edge(self):
        idx = _allowed_grid_indices_asym(GRID, 90.0, 0.1, 0.1)
        assert idx.size >= 2
        assert 10 in idx

    def test_wide_band_returns_multiple(self):
        idx = _allowed_grid_indices_asym(GRID, 70.0, 10.0, 10.0)
        # [60, 80] → indices 2,3,4,5,6,7
        assert idx.size >= 4

    def test_degenerate_single_element_grid(self):
        g = np.array([42.0])
        idx = _allowed_grid_indices_asym(g, 42.0, 0.1, 0.1)
        assert idx.tolist() == [0]


class TestWidenerIntent:
    """Widener bias toward the CEO's deliberately wider side (Task 6.4).

    Grid: np.linspace(50, 90, 11), step=4.0 €
    centre=60.0 → i_star=2 (grid[2]=58 is nearest? No: grid[2]=58, grid[3]=62;
    |60-58|=2, |60-62|=2 — tie → index 2 or 3. Let's check:
    grid = [50,54,58,62,66,70,74,78,82,86,90]
    |60-58|=2, |60-62|=2 → argmin picks 2 (first tie).
    i_star=2; neighbours: grid[1]=54, grid[3]=62.
    """

    def test_symmetric_sub_step_yields_two_actions(self):
        """dp_minus=dp_plus=0.5 < step → widener fires, returns ≥2."""
        idx = _allowed_grid_indices_asym(GRID, 60.0, 0.5, 0.5)
        assert idx.size >= 2

    def test_upper_bias_when_dp_plus_larger(self):
        """dp_plus=1.5 > dp_minus=0.5, both sub-step → widener fires, picks upper pair.

        centre=60.0; band [59.5, 61.5] contains no grid point (58 and 62 outside).
        i_star=2 (|60-58|=2 == |60-62|=2, argmin picks first → grid[2]=58).
        dp_plus > dp_minus → pair = (i_star, i_star+1) = (2, 3).
        """
        idx = _allowed_grid_indices_asym(GRID, 60.0, 0.5, 1.5)
        i_star = int(np.argmin(np.abs(GRID - 60.0)))
        expected = np.array([i_star, i_star + 1], dtype=np.int64)
        assert np.array_equal(idx, expected), (
            f"Expected upper pair {expected.tolist()}, got {idx.tolist()}"
        )

    def test_lower_bias_when_dp_minus_larger(self):
        """dp_minus=1.5 > dp_plus=0.5, both sub-step → widener fires, picks lower pair.

        centre=60.0; band [58.5, 60.5] contains no grid point.
        i_star=2; dp_minus > dp_plus → pair = (i_star-1, i_star) = (1, 2).
        """
        idx = _allowed_grid_indices_asym(GRID, 60.0, 1.5, 0.5)
        i_star = int(np.argmin(np.abs(GRID - 60.0)))
        expected = np.array([i_star - 1, i_star], dtype=np.int64)
        assert np.array_equal(idx, expected), (
            f"Expected lower pair {expected.tolist()}, got {idx.tolist()}"
        )

    def test_wide_symmetric_band_not_widened(self):
        """dp_minus=dp_plus=5.0 → band [55,65] covers grid[2]=58, grid[3]=62 naturally."""
        idx = _allowed_grid_indices_asym(GRID, 60.0, 5.0, 5.0)
        assert idx.size >= 2
        # Widener not triggered: all returned indices should be within [55, 65]
        assert GRID[idx].min() >= 55.0 - 1e-9
        assert GRID[idx].max() <= 65.0 + 1e-9

    def test_upper_bias_via_mask_helper(self):
        """End-to-end: build_action_mask_and_epsilon with dp_plus > dp_minus (sub-step)."""
        row = _mask_row(p_bar=60.0, dp_minus=0.5, dp_plus=1.5)
        assert row.sum() >= 2
        i_star = int(np.argmin(np.abs(GRID - 60.0)))
        assert row[i_star], f"i_star={i_star} should be set"
        assert row[i_star + 1], f"i_star+1={i_star+1} should be set (upper bias)"

    def test_lower_bias_via_mask_helper(self):
        """End-to-end: build_action_mask_and_epsilon with dp_minus > dp_plus (sub-step)."""
        row = _mask_row(p_bar=60.0, dp_minus=1.5, dp_plus=0.5)
        assert row.sum() >= 2
        i_star = int(np.argmin(np.abs(GRID - 60.0)))
        assert row[i_star - 1], f"i_star-1={i_star-1} should be set (lower bias)"
        assert row[i_star], f"i_star={i_star} should be set"


class TestEpsilonClamp:
    def test_epsilon_clamped_to_eps_lo(self):
        """epsilon below _EPS_LO (1e-3) is clamped up; but schema rejects < 0 anyway."""
        env = ChainEnvelopeOutput(
            chain_id="A", epoch=0, deliberation="test", rationale="t",
            groups={"default": GroupEnvelope(p_bar=60.0, delta_p=4.0, epsilon=0.001)},
        )
        _, eps = build_action_mask_and_epsilon(
            {"A": env}, ["A"], ["default"], GRID, EFFORT_GRID, 1, False,
        )
        assert eps[0] >= 1e-3

    def test_epsilon_passthrough_within_bounds(self):
        env = ChainEnvelopeOutput(
            chain_id="A", epoch=0, deliberation="test", rationale="t",
            groups={"default": GroupEnvelope(p_bar=60.0, delta_p=4.0, epsilon=0.1)},
        )
        _, eps = build_action_mask_and_epsilon(
            {"A": env}, ["A"], ["default"], GRID, EFFORT_GRID, 1, False,
        )
        assert abs(eps[0] - 0.1) < 1e-12
