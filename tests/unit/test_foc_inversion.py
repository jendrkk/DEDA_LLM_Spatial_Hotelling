"""Unit tests for the FOC-inversion calibration module."""
from __future__ import annotations

import numpy as np
import pytest

from hotelling.calibration import foc_inversion as foc


# ──────────────────────────────────────────────────────────────────────────
# compute_absolute_shares
# ──────────────────────────────────────────────────────────────────────────

class TestComputeAbsoluteShares:
    def test_basic(self):
        out = foc.compute_absolute_shares(
            s_outside=0.04, s_B_over_s_D=0.5, s_S_over_s_D=1.0
        )
        # s_D + s_S + s_B = 1 − 0.04 = 0.96
        assert out["discount"] + out["standard"] + out["bio"] == pytest.approx(0.96)
        # ratios preserved
        assert out["bio"] / out["discount"] == pytest.approx(0.5)
        assert out["standard"] / out["discount"] == pytest.approx(1.0)

    def test_store_count_default(self):
        out = foc.compute_absolute_shares(
            s_outside=0.04, s_B_over_s_D=0.5, s_S_over_s_D=None
        )
        # default s_S/s_D = 207/196 = 1.0561...
        assert out["standard"] / out["discount"] == pytest.approx(207 / 196)

    def test_invalid_outside(self):
        with pytest.raises(ValueError):
            foc.compute_absolute_shares(
                s_outside=-0.01, s_B_over_s_D=0.5, s_S_over_s_D=1.0
            )
        with pytest.raises(ValueError):
            foc.compute_absolute_shares(
                s_outside=1.5, s_B_over_s_D=0.5, s_S_over_s_D=1.0
            )

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            foc.compute_absolute_shares(
                s_outside=0.04, s_B_over_s_D=-0.1, s_S_over_s_D=1.0
            )
        with pytest.raises(ValueError):
            foc.compute_absolute_shares(
                s_outside=0.04, s_B_over_s_D=0.5, s_S_over_s_D=-0.5
            )


# ──────────────────────────────────────────────────────────────────────────
# compute_mu_from_foc
# ──────────────────────────────────────────────────────────────────────────

class TestComputeMuFromFoc:
    def test_known_mu_recovery(self):
        # Construct prices, costs, shares consistent with μ = 5.0.
        # μ_τ = (p − c)(1 − s) = 5.0
        # Pick s_D=0.4, s_S=0.4, s_B=0.16 (sum = 0.96 → outside 0.04)
        # Then (p − c)_D = 5/(1−0.4) = 8.333
        #      (p − c)_S = 5/(1−0.4) = 8.333
        #      (p − c)_B = 5/(1−0.16) = 5.952
        prices = {"discount": 34.0, "standard": 40.0, "bio": 52.0}
        costs = {
            "discount": 34.0 - 5.0/(1-0.4),
            "standard": 40.0 - 5.0/(1-0.4),
            "bio":      52.0 - 5.0/(1-0.16),
        }
        shares = {"discount": 0.4, "standard": 0.4, "bio": 0.16}
        out = foc.compute_mu_from_foc(prices, costs, shares)
        assert out["mu_by_type"]["discount"] == pytest.approx(5.0)
        assert out["mu_by_type"]["standard"] == pytest.approx(5.0)
        assert out["mu_by_type"]["bio"] == pytest.approx(5.0)
        assert out["mu_share_weighted"] == pytest.approx(5.0)
        assert out["spread_absolute"] == pytest.approx(0.0, abs=1e-9)

    def test_spread_diagnostic(self):
        # μ_D = 6.0, μ_S = 5.0, μ_B = 4.0 → spread = 2.0, mean = 5.0
        prices = {"discount": 30.0, "standard": 40.0, "bio": 50.0}
        shares = {"discount": 0.5, "standard": 0.5, "bio": 0.5}
        costs = {
            "discount": 30.0 - 6.0/0.5,   # margin = 12 → μ = 6.0
            "standard": 40.0 - 5.0/0.5,   # margin = 10 → μ = 5.0
            "bio":      50.0 - 4.0/0.5,   # margin =  8 → μ = 4.0
        }
        out = foc.compute_mu_from_foc(prices, costs, shares)
        assert out["spread_absolute"] == pytest.approx(2.0)
        # share-weighted with equal weights = simple mean = 5.0
        assert out["mu_share_weighted"] == pytest.approx(5.0)

    def test_invalid_margin(self):
        prices = {"discount": 10.0, "standard": 40.0, "bio": 50.0}
        costs = {"discount": 15.0, "standard": 20.0, "bio": 30.0}  # neg
        shares = {"discount": 0.4, "standard": 0.4, "bio": 0.2}
        with pytest.raises(ValueError, match="Non-positive markup"):
            foc.compute_mu_from_foc(prices, costs, shares)


# ──────────────────────────────────────────────────────────────────────────
# compute_accessibility_by_type
# ──────────────────────────────────────────────────────────────────────────

class TestComputeAccessibility:
    def test_uniform_cell_uniform_dist(self):
        # 2 cells, 3 stores (one per type), uniform 5-min distance.
        # A_{τ,i} = exp(−t·5/μ)·1 = exp(-5*0.5/4) = exp(-0.625) ≈ 0.5353
        dist = np.full((2, 3), 5.0)
        chain_types = np.array(["discount", "standard", "bio"], dtype=object)
        mass = np.array([100.0, 100.0])
        a = foc.compute_accessibility_by_type(
            dist_minutes=dist, chain_types=chain_types,
            cell_mass=mass, transport_cost=0.5, mu=4.0,
        )
        expected = float(np.exp(-0.5 * 5.0 / 4.0))
        assert a["discount"] == pytest.approx(expected)
        assert a["standard"] == pytest.approx(expected)
        assert a["bio"] == pytest.approx(expected)

    def test_proximity_increases_accessibility(self):
        # bio store at d=2, discount at d=20 → A_B > A_D
        dist = np.array([[20.0, 5.0, 2.0]])
        chain_types = np.array(["discount", "standard", "bio"], dtype=object)
        mass = np.array([100.0])
        a = foc.compute_accessibility_by_type(
            dist_minutes=dist, chain_types=chain_types,
            cell_mass=mass, transport_cost=0.5, mu=4.0,
        )
        assert a["bio"] > a["standard"] > a["discount"]

    def test_missing_type_raises(self):
        dist = np.full((1, 2), 5.0)
        chain_types = np.array(["discount", "standard"], dtype=object)
        mass = np.array([100.0])
        with pytest.raises(ValueError, match="No stores of chain type 'bio'"):
            foc.compute_accessibility_by_type(
                dist_minutes=dist, chain_types=chain_types,
                cell_mass=mass, transport_cost=0.5, mu=4.0,
            )


# ──────────────────────────────────────────────────────────────────────────
# compute_q_closed_form
# ──────────────────────────────────────────────────────────────────────────

class TestComputeQClosedForm:
    def test_identity_recovery(self):
        # Construct synthetic data such that q_S, q_B are known.
        # Then verify the closed-form recovers them.
        mu = 5.0
        prices = {"discount": 34.0, "standard": 40.0, "bio": 52.0}
        accessibility = {"discount": 0.8, "standard": 0.9, "bio": 0.6}
        # Pick q_S = 7.0, q_B = 18.0. Solve for shares that satisfy the
        # log-share-ratio identity:
        #   ln(s_S/s_D) = (q_S − 0 − (p_S − p_D))/μ + ln(A_S/A_D)
        #              = (7 − 6)/5 + ln(0.9/0.8)
        ln_sS_sD = (7.0 - (40.0 - 34.0)) / 5.0 + np.log(0.9 / 0.8)
        ln_sB_sD = (18.0 - (52.0 - 34.0)) / 5.0 + np.log(0.6 / 0.8)
        # Build shares satisfying the ratios; normalise to sum to 0.96.
        s_D = 1.0
        s_S = s_D * np.exp(ln_sS_sD)
        s_B = s_D * np.exp(ln_sB_sD)
        total = s_D + s_S + s_B
        shares = {
            "discount": 0.96 * s_D / total,
            "standard": 0.96 * s_S / total,
            "bio":      0.96 * s_B / total,
        }
        out = foc.compute_q_closed_form(
            mu=mu, shares=shares, accessibility=accessibility, prices=prices,
        )
        assert out["q_D"] == 0.0
        assert out["q_S"] == pytest.approx(7.0, rel=1e-9)
        assert out["q_B"] == pytest.approx(18.0, rel=1e-9)

    def test_vertical_ordering_violation(self):
        # If demand inputs force q_B ≤ q_S, raise.
        mu = 5.0
        prices = {"discount": 34.0, "standard": 40.0, "bio": 52.0}
        accessibility = {"discount": 1.0, "standard": 1.0, "bio": 1.0}
        # Bio share very low relative to standard → q_B < q_S despite price premium.
        shares = {"discount": 0.5, "standard": 0.35, "bio": 0.01}
        with pytest.raises(ValueError, match="Vertical ordering violated"):
            foc.compute_q_closed_form(
                mu=mu, shares=shares,
                accessibility=accessibility, prices=prices,
            )
