import numpy as np

from hotelling.llm.group_analytics import compute_group_briefs


def test_briefs_basic():
    # 4 stores: 2 Edeka (standard), 2 Rewe (standard). One group "default".
    W, N = 6, 4
    price = np.full((W, N), 50.0)
    price[:, 0] = 45.0   # Edeka store 0 cheaper
    demand = np.full((W, N), 100.0)
    aw = {"price": price, "effort": np.zeros((W, N)),
          "demand": demand, "profit": np.zeros((W, N))}
    meta = [{"n_rivals_within_R": r} for r in (3, 4, 2, 5)]
    briefs = compute_group_briefs(
        aw, chain_id="Edeka",
        store_chain=["Edeka", "Edeka", "Rewe", "Rewe"],
        store_chain_type=["standard"] * 4,
        store_group_labels=["default"] * 4,
        store_metadata=meta, group_keys=["default"], marginal_cost=31.2,
    )
    b = briefs["default"]
    assert b["margin_eur"] > 0
    # Edeka mean (47.5) undercuts same-tier Rewe (50.0)
    assert b["price_gap_vs_same_tier"] < 0
    assert b["position_label"] == "undercut"
    assert abs(b["mean_local_competition"] - 3.5) < 1e-9  # mean of rivals 3,4


def test_build_ceo_state_enrich_groups():
    import numpy as np
    from hotelling.llm.ceo_state import RollingWindow, build_ceo_state

    N = 4
    w = RollingWindow(N, window=5)
    for _ in range(5):
        w.push(
            prices=np.array([45.0, 50.0, 50.0, 50.0]),
            efforts=np.zeros(N),
            demands=np.full(N, 100.0),
            profits=np.zeros(N),
        )
    meta = [{"n_rivals_within_R": r, "store_id": str(i)} for i, r in enumerate((3, 4, 2, 5))]
    state_off = build_ceo_state(
        w, chain_id="Edeka",
        store_chain=["Edeka", "Edeka", "Rewe", "Rewe"],
        store_chain_type=["standard"] * 4,
        store_group_labels=["default"] * 4,
        group_keys=["default"],
        zones={"total_population": 1.0, "high_status_share_pct": 40.0, "zones": []},
        history=[], epoch=0, T_ceo=5, marginal_cost=31.2,
        min_delta_p=1.5, min_delta_e=0.1,
    )
    assert "position_label" not in state_off["own"]["group_performance"][0]

    state_on = build_ceo_state(
        w, chain_id="Edeka",
        store_chain=["Edeka", "Edeka", "Rewe", "Rewe"],
        store_chain_type=["standard"] * 4,
        store_group_labels=["default"] * 4,
        group_keys=["default"],
        zones={"total_population": 1.0, "high_status_share_pct": 40.0, "zones": []},
        history=[], epoch=0, T_ceo=5, marginal_cost=31.2,
        min_delta_p=1.5, min_delta_e=0.1,
        store_metadata=meta, enrich_groups=True,
    )
    gp = state_on["own"]["group_performance"][0]
    assert gp["position_label"] == "undercut"
    assert "margin_eur" in gp
    assert "demand_index" in gp
