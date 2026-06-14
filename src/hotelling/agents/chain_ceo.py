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
from hotelling.llm.schemas import ChainEnvelopeOutput, GroupEnvelope

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
        T_ceo: int,
    ) -> None:
        self.chain_id = chain_id
        self.chain_type = chain_type
        self.marginal_cost = float(marginal_cost)
        self.client = client
        self.group_keys = list(group_keys)
        self.min_delta_p = float(min_delta_p)
        self.min_delta_e = float(min_delta_e)
        self.T_ceo = int(T_ceo)
        self._env = _jinja_env()
        self._system_tmpl = self._env.get_template("system_ceo.jinja")
        self._state_tmpl = self._env.get_template("state_ceo.jinja")
        self._div_ctx = division_context(active_divisions, division_params)
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
        }

    def decide(
        self,
        state: dict,
        epoch: int,
        previous: ChainEnvelopeOutput | None = None,
    ) -> ChainEnvelopeOutput:
        """Query the LLM for a new envelope; fall back on any failure."""
        try:
            system_prompt = self._system_tmpl.render(**self._system_ctx)
            state_prompt = self._state_tmpl.render(
                active_divisions=self._div_ctx, **state
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state_prompt},
            ]
            out = self.client.complete(messages, response_model=ChainEnvelopeOutput)
            self._validate(out)
            logger.info("CEO %s epoch %d OK: %s", self.chain_id, epoch, out.rationale[:80])
            return out
        except Exception as exc:  # noqa: BLE001 — never crash the simulation on a bad call
            logger.warning(
                "CEO %s epoch %d failed (%s); retaining previous envelope.",
                self.chain_id, epoch, exc,
            )
            return previous if previous is not None else self._safe_default(state, epoch)

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

    def _safe_default(self, state: dict, epoch: int) -> ChainEnvelopeOutput:
        """Neutral envelope centred on the chain's recent mean price."""
        p = max(float(state.get("own", {}).get("mean_price_last_T", 0.0)),
                self.marginal_cost * 1.05, self.marginal_cost + 1.0)
        dp = max(self.min_delta_p, 0.1 * p)
        de = max(self.min_delta_e, 0.1)
        groups = {
            k: GroupEnvelope(p_bar=p, delta_p=dp, e_bar=0.5, delta_e=de, epsilon=0.05)
            for k in self.group_keys
        }
        return ChainEnvelopeOutput(
            chain_id=self.chain_id, epoch=epoch, groups=groups,
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
) -> dict[str, ChainCEO]:
    """Group firms by brand and build one ChainCEO per chain.

    Marginal cost / chain_type are taken from any store of the brand (they share
    chain_type within a brand). Returns brand -> ChainCEO.
    """
    by_brand: dict[str, list] = {}
    for f in firms:
        by_brand.setdefault(str(f.chain), []).append(f)
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
            T_ceo=T_ceo,
        )
    return ceos
