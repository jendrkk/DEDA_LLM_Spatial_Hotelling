"""LLM-backed chain CEO agent that sets strategy envelopes each epoch.

Responsibility: query the LLM for a ChainEnvelopeOutput every T_CEO periods,
log the call to JSONL, and return the validated envelope. CEO calls are never
batched across chains (ADR-007).

Public API: ChainCEO

Key dependencies: hotelling.llm.client, hotelling.llm.schemas, hotelling.envelope

References: docs/agent_simulation_technical_report.md §6; ADR-007.
"""
from __future__ import annotations

import logging
from pathlib import Path

from hotelling.llm.ceo_state import ct_code, ct_label, division_context
from hotelling.llm.client import LLMClient
from hotelling.llm.schemas import ChainEnvelopeOutput, CoordinationSignal, GroupEnvelope

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


def _jinja_env():
    from jinja2 import Environment, FileSystemLoader

    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        autoescape=False, trim_blocks=True, lstrip_blocks=True,
    )


class ChainCEO:
    """LLM-backed chain CEO; outputs a per-group strategy envelope each epoch.

    Renders system_ceo.jinja + state_ceo.jinja, calls the LLM for a validated
    ChainEnvelopeOutput, checks group-key completeness and the marginal-cost
    floor, and on any failure retains the previous envelope (ADR-007; report 6.3).
    """

    def __init__(
        self,
        *,
        chain_id: str,
        chain_type: str,                 # "discount" | "standard" | "bio"
        marginal_cost: float,
        client: LLMClient,
        active_divisions: list[str],
        division_params: dict | None,
        group_keys: list[str],
        min_delta_p: float,
        min_delta_e: float,
        grid_spec: dict | None = None,
        graph_spec: dict | None = None,
        T_ceo: int,
        merge_system: bool = False,
        capture_comm: bool = False,
        with_effort: bool = True,
        with_comm: bool = False,
    ) -> None:
        self.chain_id = chain_id
        self.chain_type = chain_type
        self.marginal_cost = float(marginal_cost)
        self.client = client
        self.group_keys = list(group_keys)
        self.min_delta_p = float(min_delta_p)
        self.min_delta_e = float(min_delta_e)
        self.T_ceo = int(T_ceo)
        self.merge_system = bool(merge_system)
        self.capture_comm = bool(capture_comm)
        self.with_effort = bool(with_effort)
        self.with_comm = bool(with_comm)
        self.transcripts: list[dict] = []
        self.n_success = 0
        self.n_fail = 0
        self.last_error: str | None = None
        self._env = _jinja_env()
        self._system_tmpl = self._env.get_template("system_ceo.jinja")
        self._state_tmpl = self._env.get_template("state_ceo.jinja")
        self._div_ctx = division_context(active_divisions, division_params)

        # Action-grid + rival-observation context for the prompt (Step 2 specs).
        # Safe placeholders keep the CEO constructible if the runner has not yet
        # been wired (mid-migration); Step 4 always supplies real specs.
        self.grid_spec = grid_spec or {
            "regime": "G", "m": int(round(1.0)), "lo": 0.0, "hi": 0.0,
            "step": 0.0, "grid": [], "other_chain_grids": {},
        }
        self.graph_spec = graph_spec or {
            "mode": "neighbors", "match": "n/a", "grid_regime": "G", "k": 1,
            "n_stores": 0, "mean_observed": 0.0, "max_observed": 0, "n_isolated": 0,
        }
        self._min_step_delta = float(self.grid_spec.get("step", 0.0))

        _gs = self.grid_spec
        _gr = self.graph_spec
        _match = str(_gr.get("match", "n/a"))
        if _match == "SC":
            _match_explainer = ("the most pivotal rivals OF THE SAME CHAIN TYPE as the "
                                "store (same-tier competitors)")
        elif _match == "A":
            _match_explainer = ("the most pivotal rivals of ANY chain type (could be "
                                "discount, standard, or bio)")
        else:
            _match_explainer = "its nearest competitors"
        _grid_ctx = {
            "regime": str(_gs.get("regime", "G")),
            "m": int(_gs.get("m", 0)),
            "lo": round(float(_gs.get("lo", 0.0)), 2),
            "hi": round(float(_gs.get("hi", 0.0)), 2),
            "step": round(float(_gs.get("step", 0.0)), 3),
            "other_chain_grids": {
                ct: {"lo": round(float(v["lo"]), 2),
                     "hi": round(float(v["hi"]), 2),
                     "step": round(float(v["step"]), 3)}
                for ct, v in (_gs.get("other_chain_grids", {}) or {}).items()
            },
        }
        _graph_ctx = {
            "mode": str(_gr.get("mode", "neighbors")),
            "match": _match,
            "match_explainer": _match_explainer,
            "k": int(_gr.get("k", 1)),
            "mean_observed": round(float(_gr.get("mean_observed", 0.0)), 2),
            "max_observed": int(_gr.get("max_observed", 0)),
            "n_isolated": int(_gr.get("n_isolated", 0)),
            "n_stores": int(_gr.get("n_stores", 0)),
            "grid_regime": str(_gr.get("grid_regime", "G")),
        }
        self._system_ctx = {
            "chain_id": chain_id,
            "chain_type": ct_code(chain_type),
            "chain_type_label": ct_label(chain_type),
            "T_ceo": self.T_ceo,
            "marginal_cost": round(self.marginal_cost, 2),
            "active_divisions": self._div_ctx,
            "n_groups": len(self.group_keys),
            "group_keys": self.group_keys,
            "min_delta_p": self.min_delta_p,
            "min_delta_e": self.min_delta_e,
            "with_effort": self.with_effort,
            "with_comm": self.with_comm,
            "grid": _grid_ctx,
            "graph": _graph_ctx,
        }

    def decide(
        self,
        state: dict,
        epoch: int,
        previous: ChainEnvelopeOutput | None = None,
    ) -> ChainEnvelopeOutput:
        """Query the LLM for a new envelope; fall back on any failure."""
        messages = None
        try:
            system_prompt = self._system_tmpl.render(**self._system_ctx)
            state_prompt = self._state_tmpl.render(
                active_divisions=self._div_ctx, grid=self._system_ctx["grid"], **state
            )
            if self.merge_system:
                messages = [
                    {"role": "user", "content": system_prompt + "\n\n" + state_prompt},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": state_prompt},
                ]
            if self.capture_comm:
                out, transcript = self.client.complete_with_transcript(
                    messages, response_model=ChainEnvelopeOutput
                )
            else:
                out = self.client.complete(messages, response_model=ChainEnvelopeOutput)
                transcript = None
            self._validate(out)
            self.n_success += 1
            if self.capture_comm:
                self._record_transcript(epoch, messages, transcript)
            logger.info("CEO %s epoch %d OK: %s", self.chain_id, epoch, out.rationale[:80])
            return out
        except Exception as exc:  # noqa: BLE001 — never crash the simulation on a bad call
            self.n_fail += 1
            self.last_error = repr(exc)[:200]
            if self.capture_comm:
                self._record_transcript(epoch, messages, f"FAILED: {repr(exc)[:1500]}")
            logger.warning(
                "CEO %s epoch %d failed (%s); retaining previous envelope.",
                self.chain_id, epoch, exc,
            )
            return previous if previous is not None else self._safe_default(state, epoch)

    def _record_transcript(self, epoch: int, messages, response_text) -> None:
        prompt_text = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in (messages or [])
        )
        self.transcripts.append({
            "epoch": int(epoch),
            "chain": self.chain_id,
            "prompt": prompt_text,
            "response": response_text or "",
        })

    def _validate(self, out: ChainEnvelopeOutput) -> None:
        if set(out.groups) != set(self.group_keys):
            raise ValueError(
                f"group keys {sorted(out.groups)} != expected {sorted(self.group_keys)}"
            )
        for key, g in out.groups.items():
            if g.p_bar <= self.marginal_cost:
                raise ValueError(
                    f"group {key}: p_bar {g.p_bar} <= marginal_cost {self.marginal_cost}"
                )
        if self.with_comm:
            cs = out.coordination_signal
            if cs is None:
                raise ValueError("with_comm is on but coordination_signal is missing")
            if cs.proposed_tier_price <= self.marginal_cost:
                raise ValueError(
                    f"proposed_tier_price {cs.proposed_tier_price} <= "
                    f"marginal_cost {self.marginal_cost}"
                )

    def _safe_default(self, state: dict, epoch: int) -> ChainEnvelopeOutput:
        """Neutral envelope centred on the chain's recent mean price.

        Emits asymmetric widths that are grid-feasible: at least one grid step
        (or min_delta_p) of room on each side, so the masking floor never has to
        kick in for the fallback.
        """
        p = max(float(state.get("own", {}).get("mean_price_last_T", 0.0)),
                self.marginal_cost * 1.05, self.marginal_cost + 1.0)
        step = max(self._min_step_delta, 0.0)
        dp = max(self.min_delta_p, step, 0.1 * p)
        de = max(self.min_delta_e, 0.1)
        e_bar = 0.5 if self.with_effort else 0.0
        groups = {
            k: GroupEnvelope(
                p_bar=p, dp_minus=dp, dp_plus=dp, delta_p=dp,
                e_bar=e_bar, delta_e=de, epsilon=0.05,
            )
            for k in self.group_keys
        }
        signal = (
            CoordinationSignal(willing=False, proposed_tier_price=p)
            if self.with_comm else None
        )
        return ChainEnvelopeOutput(
            chain_id=self.chain_id, epoch=epoch, groups=groups,
            coordination_signal=signal,
            deliberation=("FALLBACK: the LLM call failed or returned an invalid "
                          "envelope; retaining a neutral status-quo band centred on "
                          "the chain's recent mean price."),
            rationale="FALLBACK: neutral envelope (LLM call failed or invalid).",
        )


def build_chain_ceos(
    firms: list,
    *,
    client: LLMClient,
    active_divisions: list[str],
    division_params: dict | None,
    group_keys: list[str],
    min_delta_p: float,
    min_delta_e: float,
    T_ceo: int,
    merge_system: bool = False,
    capture_comm: bool = False,
    with_effort: bool = True,
    with_comm: bool = False,
    grid_specs: dict[str, dict] | None = None,
    graph_specs: dict[str, dict] | None = None,
) -> dict[str, ChainCEO]:
    """Group firms by brand and build one ChainCEO per chain.

    Marginal cost / chain_type are taken from any store of the brand (they share
    chain_type within a brand). ``grid_specs`` / ``graph_specs`` map brand -> the
    per-chain grid / rival-observation spec (from env.grid_spec / graph_degree_spec);
    when omitted the CEO uses safe placeholder context. Returns brand -> ChainCEO.
    """
    by_brand: dict[str, list] = {}
    for f in firms:
        by_brand.setdefault(str(f.chain), []).append(f)
    grid_specs = grid_specs or {}
    graph_specs = graph_specs or {}
    ceos: dict[str, ChainCEO] = {}
    for brand, members in by_brand.items():
        rep = members[0]
        ceos[brand] = ChainCEO(
            chain_id=brand,
            chain_type=str(rep.chain_type),
            marginal_cost=float(rep.marginal_cost),
            client=client,
            active_divisions=active_divisions,
            division_params=division_params,
            group_keys=group_keys,
            min_delta_p=min_delta_p,
            min_delta_e=min_delta_e,
            grid_spec=grid_specs.get(brand),
            graph_spec=graph_specs.get(brand),
            T_ceo=T_ceo,
            merge_system=merge_system,
            capture_comm=capture_comm,
            with_effort=with_effort,
            with_comm=with_comm,
        )
    return ceos
