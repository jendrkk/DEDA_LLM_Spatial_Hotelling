import numpy as np

from hotelling.envelope.groups import (
    assign_groups, composite_group_keys, build_store_metadata,
)
from hotelling.envelope.masking import build_action_mask_and_epsilon
from hotelling.llm.schemas import ChainEnvelopeOutput, GroupEnvelope


class _F:  # minimal Firm stand-in
    def __init__(self, j, chain, ct, xy):
        self.id, self.chain, self.chain_type, self.location = str(j), chain, ct, xy


def test_default_group():
    md = [{"store_id": "0", "n_rivals_within_R": 5, "social_index": 0.6}]
    assert composite_group_keys([]) == ["default"]
    assert assign_groups(md, []) == {"0": "default"}


def test_two_divisions_keys_and_labels():
    keys = composite_group_keys(
        ["DIVISION_COMPETITION", "DIVISION_NEIGHBOURHOOD"],
        {"threshold_n_rivals": 3, "status_threshold": 0.5},
    )
    assert set(keys) == {"HEAVY_RICH", "HEAVY_POOR", "EASY_RICH", "EASY_POOR"}
    md = [
        {"store_id": "0", "n_rivals_within_R": 5, "social_index": 0.7},
        {"store_id": "1", "n_rivals_within_R": 1, "social_index": 0.2},
    ]
    labels = assign_groups(
        md, ["DIVISION_COMPETITION", "DIVISION_NEIGHBOURHOOD"],
        {"threshold_n_rivals": 3, "status_threshold": 0.5},
    )
    assert labels == {"0": "HEAVY_RICH", "1": "EASY_POOR"}


def test_build_store_metadata_rival_count():
    firms = [_F(0, "A", "discount", (0.0, 0.0)),
             _F(1, "A", "discount", (10.0, 0.0)),
             _F(2, "B", "bio", (1000.0, 0.0))]
    md = build_store_metadata(firms, grid_gdf=None, radius_m=100.0)
    assert md[0]["n_rivals_within_R"] == 1   # store 1 is within 100 m
    assert md[2]["n_rivals_within_R"] == 0
    assert all(m["social_index"] == 0.5 for m in md)  # no grid -> default


def test_mask_and_epsilon_price_only():
    price_grid = np.linspace(20.0, 55.0, 25)
    effort_grid = np.array([0.0])
    env = ChainEnvelopeOutput(
        chain_id="A", epoch=0, deliberation="test", rationale="t",
        groups={"default": GroupEnvelope(p_bar=30.0, delta_p=3.0, e_bar=0.5,
                                         delta_e=0.2, epsilon=0.05)},
    )
    mask, eps = build_action_mask_and_epsilon(
        {"A": env}, store_chain=["A", "A"], store_group_labels=["default", "default"],
        price_grid=price_grid, effort_grid=effort_grid, m_effort=1, mask_effort=False,
    )
    assert mask.shape == (2, 25)
    allowed_prices = price_grid[np.nonzero(mask[0])[0]]
    assert allowed_prices.min() >= 27.0 - 1e-9 and allowed_prices.max() <= 33.0 + 1e-9
    assert mask[0].any()  # non-empty
    np.testing.assert_allclose(eps, 0.05)


def test_mask_snap_to_nearest_when_band_too_narrow():
    price_grid = np.linspace(20.0, 55.0, 25)  # step ~1.46
    env = ChainEnvelopeOutput(
        chain_id="A", epoch=0, deliberation="test", rationale="t",
        groups={"default": GroupEnvelope(p_bar=30.0, delta_p=0.1, e_bar=0.5,
                                         delta_e=0.2, epsilon=0.05)},
    )
    mask, _ = build_action_mask_and_epsilon(
        {"A": env}, ["A"], ["default"], price_grid, np.array([0.0]), 1, False,
    )
    # ≥2 guaranteed by the snap-and-widen floor (band narrower than one grid step)
    assert mask[0].sum() >= 2
