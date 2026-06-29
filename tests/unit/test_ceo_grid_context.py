"""Tests for ChainCEO grid/graph spec injection (Step 3)."""
from __future__ import annotations

import pytest

from hotelling.agents.chain_ceo import ChainCEO, build_chain_ceos
from hotelling.llm.client import LLMClient


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_GRID_SPEC_CS = {
    "regime": "CS",
    "m": 21,
    "lo": 50.0,
    "hi": 90.0,
    "step": 2.0,
    "grid": list(range(50, 91, 2)),  # 21 values
    "other_chain_grids": {
        "standard": {"lo": 55.0, "hi": 95.0, "step": 2.0},
    },
}

_GRAPH_SPEC_SC = {
    "mode": "graph_states",
    "match": "SC",
    "grid_regime": "CS",
    "k": 2,
    "n_stores": 40,
    "mean_observed": 1.7,
    "max_observed": 2,
    "n_isolated": 5,
}

_GRAPH_SPEC_A = {
    "mode": "graph_states",
    "match": "A",
    "grid_regime": "G",
    "k": 3,
    "n_stores": 20,
    "mean_observed": 2.1,
    "max_observed": 3,
    "n_isolated": 0,
}


def _dummy_client() -> LLMClient:
    return LLMClient(model="gemini/gemini-2.5-flash", log_path=None)


def _make_ceo(grid_spec=None, graph_spec=None, min_delta_p=1.5, mc=31.2) -> ChainCEO:
    return ChainCEO(
        chain_id="Edeka",
        chain_type="standard",
        marginal_cost=mc,
        client=_dummy_client(),
        active_divisions=[],
        division_params=None,
        group_keys=["default"],
        min_delta_p=min_delta_p,
        min_delta_e=0.1,
        grid_spec=grid_spec,
        graph_spec=graph_spec,
        T_ceo=100,
    )


# ---------------------------------------------------------------------------
# AC: acceptance-criteria assertions from the task
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    def test_grid_step_in_system_ctx(self):
        ceo = _make_ceo(grid_spec=_GRID_SPEC_CS, graph_spec=_GRAPH_SPEC_SC)
        assert ceo._system_ctx["grid"]["step"] == 2.0

    def test_graph_k_in_system_ctx(self):
        ceo = _make_ceo(grid_spec=_GRID_SPEC_CS, graph_spec=_GRAPH_SPEC_SC)
        assert ceo._system_ctx["graph"]["k"] == 2

    def test_match_explainer_sc_contains_same_chain_type(self):
        ceo = _make_ceo(grid_spec=_GRID_SPEC_CS, graph_spec=_GRAPH_SPEC_SC)
        explainer = ceo._system_ctx["graph"]["match_explainer"]
        assert "SAME CHAIN TYPE" in explainer.upper()

    def test_match_explainer_a_contains_any(self):
        ceo = _make_ceo(grid_spec=_GRID_SPEC_CS, graph_spec=_GRAPH_SPEC_A)
        explainer = ceo._system_ctx["graph"]["match_explainer"]
        assert "ANY" in explainer.upper()


# ---------------------------------------------------------------------------
# grid key in _system_ctx
# ---------------------------------------------------------------------------

class TestGridContext:
    def test_grid_key_present(self):
        ceo = _make_ceo(grid_spec=_GRID_SPEC_CS)
        assert "grid" in ceo._system_ctx

    def test_grid_regime(self):
        ceo = _make_ceo(grid_spec=_GRID_SPEC_CS)
        assert ceo._system_ctx["grid"]["regime"] == "CS"

    def test_grid_m(self):
        ceo = _make_ceo(grid_spec=_GRID_SPEC_CS)
        assert ceo._system_ctx["grid"]["m"] == 21

    def test_grid_lo_hi(self):
        ceo = _make_ceo(grid_spec=_GRID_SPEC_CS)
        g = ceo._system_ctx["grid"]
        assert g["lo"] == 50.0
        assert g["hi"] == 90.0

    def test_other_chain_grids_forwarded(self):
        ceo = _make_ceo(grid_spec=_GRID_SPEC_CS)
        others = ceo._system_ctx["grid"]["other_chain_grids"]
        assert "standard" in others
        assert others["standard"]["lo"] == 55.0

    def test_placeholder_when_no_spec(self):
        ceo = _make_ceo(grid_spec=None)
        g = ceo._system_ctx["grid"]
        assert g["regime"] == "G"
        assert g["m"] == 1  # placeholder: int(round(1.0))
        assert g["other_chain_grids"] == {}


# ---------------------------------------------------------------------------
# graph key in _system_ctx
# ---------------------------------------------------------------------------

class TestGraphContext:
    def test_graph_key_present(self):
        ceo = _make_ceo(graph_spec=_GRAPH_SPEC_SC)
        assert "graph" in ceo._system_ctx

    def test_match_forwarded(self):
        ceo = _make_ceo(graph_spec=_GRAPH_SPEC_SC)
        assert ceo._system_ctx["graph"]["match"] == "SC"

    def test_n_isolated(self):
        ceo = _make_ceo(graph_spec=_GRAPH_SPEC_SC)
        assert ceo._system_ctx["graph"]["n_isolated"] == 5

    def test_mean_observed_rounded(self):
        ceo = _make_ceo(graph_spec=_GRAPH_SPEC_SC)
        assert abs(ceo._system_ctx["graph"]["mean_observed"] - 1.7) < 1e-6

    def test_placeholder_match_explainer(self):
        ceo = _make_ceo(graph_spec=None)
        # default match is "n/a" → "nearest competitors"
        assert "nearest" in ceo._system_ctx["graph"]["match_explainer"]


# ---------------------------------------------------------------------------
# _min_step_delta
# ---------------------------------------------------------------------------

class TestMinStepDelta:
    def test_min_step_delta_from_spec(self):
        ceo = _make_ceo(grid_spec=_GRID_SPEC_CS)
        assert ceo._min_step_delta == 2.0

    def test_min_step_delta_zero_when_no_spec(self):
        ceo = _make_ceo(grid_spec=None)
        assert ceo._min_step_delta == 0.0


# ---------------------------------------------------------------------------
# _safe_default: grid-feasible asymmetric widths
# ---------------------------------------------------------------------------

class TestSafeDefault:
    def _fallback(self, step=2.0, mc=31.2, min_delta_p=1.5):
        spec = dict(_GRID_SPEC_CS, step=step) if step else None
        ceo = _make_ceo(grid_spec=spec, mc=mc, min_delta_p=min_delta_p)
        state = {"own": {"mean_price_last_T": 35.0}}
        return ceo, ceo._safe_default(state, epoch=0)

    def test_dp_minus_positive(self):
        _, out = self._fallback()
        assert out.groups["default"].dp_minus > 0

    def test_dp_plus_positive(self):
        _, out = self._fallback()
        assert out.groups["default"].dp_plus > 0

    def test_dp_minus_at_least_step(self):
        step = 2.0
        _, out = self._fallback(step=step)
        assert out.groups["default"].dp_minus >= step

    def test_dp_minus_at_least_min_delta_p(self):
        min_dp = 3.0
        _, out = self._fallback(step=0.5, min_delta_p=min_dp)
        assert out.groups["default"].dp_minus >= min_dp

    def test_fallback_rationale(self):
        _, out = self._fallback()
        assert "FALLBACK" in out.rationale

    def test_p_bar_above_mc(self):
        mc = 31.2
        _, out = self._fallback(mc=mc)
        assert out.groups["default"].p_bar > mc

    def test_zero_step_still_positive(self):
        """Even with step=0 (placeholder), dp > 0 from min_delta_p/0.1*p floor."""
        _, out = self._fallback(step=0.0)
        assert out.groups["default"].dp_minus > 0


# ---------------------------------------------------------------------------
# build_chain_ceos: new params forwarded
# ---------------------------------------------------------------------------

class TestBuildChainCeos:
    def _firms(self):
        class _F:
            def __init__(self, chain, ct, mc):
                self.chain = chain
                self.chain_type = ct
                self.marginal_cost = mc
        return [_F("Edeka", "standard", 31.2), _F("Aldi", "discount", 20.0)]

    def test_no_specs_still_builds(self):
        ceos = build_chain_ceos(
            self._firms(),
            client=_dummy_client(),
            active_divisions=[],
            division_params=None,
            group_keys=["default"],
            min_delta_p=1.5,
            min_delta_e=0.1,
            T_ceo=100,
        )
        assert set(ceos) == {"Edeka", "Aldi"}

    def test_grid_specs_forwarded(self):
        ceos = build_chain_ceos(
            self._firms(),
            client=_dummy_client(),
            active_divisions=[],
            division_params=None,
            group_keys=["default"],
            min_delta_p=1.5,
            min_delta_e=0.1,
            T_ceo=100,
            grid_specs={"Edeka": _GRID_SPEC_CS},
        )
        assert ceos["Edeka"]._system_ctx["grid"]["step"] == 2.0
        # Aldi got no spec → placeholder
        assert ceos["Aldi"]._system_ctx["grid"]["regime"] == "G"

    def test_graph_specs_forwarded(self):
        ceos = build_chain_ceos(
            self._firms(),
            client=_dummy_client(),
            active_divisions=[],
            division_params=None,
            group_keys=["default"],
            min_delta_p=1.5,
            min_delta_e=0.1,
            T_ceo=100,
            graph_specs={"Aldi": _GRAPH_SPEC_A},
        )
        assert ceos["Aldi"]._system_ctx["graph"]["k"] == 3
        # Edeka got no spec → placeholder k=1
        assert ceos["Edeka"]._system_ctx["graph"]["k"] == 1


# ---------------------------------------------------------------------------
# Existing test still passes (grid/graph keys now also present)
# ---------------------------------------------------------------------------

def test_existing_system_ctx_keys_still_present():
    ceo = _make_ceo()
    ctx = ceo._system_ctx
    for key in ("chain_id", "chain_type", "chain_type_label", "T_ceo",
                "marginal_cost", "active_divisions", "n_groups", "group_keys",
                "min_delta_p", "min_delta_e", "with_effort", "with_comm",
                "grid", "graph"):
        assert key in ctx, f"missing key: {key!r}"
