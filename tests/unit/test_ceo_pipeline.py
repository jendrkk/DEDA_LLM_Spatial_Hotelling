import numpy as np

from hotelling.llm.ceo_state import RollingWindow, build_ceo_state


def test_build_ceo_state_default_group():
    N = 4
    w = RollingWindow(N, window=5)
    for _ in range(5):
        w.push(
            prices=np.array([30.0, 31.0, 26.0, 41.0]),
            efforts=np.zeros(N),
            demands=np.array([100.0, 120.0, 200.0, 50.0]),
            profits=np.array([300.0, 360.0, 700.0, 400.0]),
        )
    state = build_ceo_state(
        w, chain_id="Edeka",
        store_chain=["Edeka", "Edeka", "Aldi", "BioCo"],
        store_chain_type=["standard", "standard", "discount", "bio"],
        store_group_labels=["default"] * N,
        group_keys=["default"], zones={"total_population": 1.0,
                                        "high_status_share_pct": 40.0, "zones": []},
        history=[], epoch=0, T_ceo=5, marginal_cost=31.2,
        min_delta_p=1.5, min_delta_e=0.1,
    )
    assert state["own"]["n_stores"] == 2
    assert abs(state["own"]["mean_price_last_T"] - 30.5) < 1e-9
    assert {r["id"] for r in state["rivals"]} == {"Aldi", "BioCo"}
    assert state["own"]["group_performance"][0]["group_key"] == "default"


def test_decide_fallback_without_api_key(monkeypatch):
    # Force the LLM call to fail; decide() must return the safe-default envelope.
    from hotelling.agents.chain_ceo import ChainCEO
    from hotelling.llm.client import LLMClient

    client = LLMClient(model="gemini/gemini-2.5-flash", log_path=None)

    def _boom(*a, **k):
        raise RuntimeError("no api key in test")

    monkeypatch.setattr(client, "complete", _boom)
    ceo = ChainCEO(
        chain_id="Edeka", chain_type="standard", marginal_cost=31.2, client=client,
        active_divisions=[], division_params=None, group_keys=["default"],
        min_delta_p=1.5, min_delta_e=0.1, T_ceo=100,
    )
    state = {"own": {"mean_price_last_T": 35.0}}
    out = ceo.decide(state, epoch=0, previous=None)
    assert set(out.groups) == {"default"}
    assert out.groups["default"].p_bar > 31.2
    assert "FALLBACK" in out.rationale
