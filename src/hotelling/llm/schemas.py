"""Pydantic schemas for LLM structured outputs.

Responsibility: define validated data models for all LLM agent decisions —
CEO strategy envelopes, entrant entry decisions, entrant reassessment outputs,
and the response functions / Q-table init choices they embed.  Used with
Instructor to enforce structured JSON responses from the LLM backends.

Public API:
    GroupEnvelope, ChainEnvelopeOutput, CoordinationSignal,
    RivalUnderCutResponse, ProfitDistressResponse,
    ReassessTrigger, ResponseFunction,
    QtableInitChoice, EntrantEntryDecision, EntrantReassessOutput

Key dependencies: pydantic >= 2.7

References:
    Calvano et al. (2020, AER) — Q-learning store agent design.
    Fish, Gonczarowski & Shorrer (2024) — LLM pricing behaviour.
    docs/agent_simulation_technical_report.md §6.3 — schema specification.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class GroupEnvelope(BaseModel):
    """Strategy envelope set by the CEO for one store group.

    Defines the price/effort target midpoint and the asymmetric price half-widths
    (`dp_minus` below target, `dp_plus` above target) that bound the Q-learning
    store agents in this group, plus the exploration rate (epsilon) the RL agents
    use within the envelope.

    Backward compatibility: `delta_p` is retained as a field. When an envelope is
    built with only `delta_p` (legacy), `dp_minus`/`dp_plus` are filled from it.
    When built with `dp_minus`/`dp_plus`, `delta_p` is normalised to their max so
    any legacy reader still sees a symmetric-equivalent half-width.
    """

    p_bar: float = Field(..., description="Target price midpoint (€)")
    delta_p: float = Field(
        default=1.0,
        description="Legacy symmetric price half-width (€). Retained for "
                    "compatibility; auto-set to max(dp_minus, dp_plus).",
    )
    dp_minus: float = Field(
        default=-1.0,
        description="Price wiggle room BELOW target (€); band lower edge = p_bar - dp_minus. "
                    "Must be > 0. Defaults to delta_p when omitted.",
    )
    dp_plus: float = Field(
        default=-1.0,
        description="Price wiggle room ABOVE target (€); band upper edge = p_bar + dp_plus. "
                    "Must be > 0. Defaults to delta_p when omitted.",
    )
    e_bar: float = Field(
        default=0.0,
        description="Target effort midpoint [0, 1]; ignored/omitted in price-only mode",
    )
    delta_e: float = Field(
        default=0.1,
        description="Effort half-width; ignored/omitted in price-only mode",
    )
    epsilon: float = Field(..., description="RL exploration rate for this group (0, 0.25)")

    @field_validator("p_bar")
    @classmethod
    def p_bar_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("p_bar must be positive")
        return v

    @field_validator("delta_e")
    @classmethod
    def delta_e_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("delta_e must be positive")
        return v

    @field_validator("e_bar")
    @classmethod
    def e_bar_unit_interval(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("e_bar must be in [0, 1]")
        return v

    @field_validator("epsilon")
    @classmethod
    def epsilon_range(cls, v: float) -> float:
        # Tightened from (0, 0.5): within a small in-envelope action set, eps > 0.25
        # is pure noise with no coordination benefit and inflates Δ measurement variance.
        if not (0.0 < v < 0.25):
            raise ValueError("epsilon must be in (0.0, 0.25)")
        return v

    @model_validator(mode="after")
    def _resolve_asymmetric_widths(self) -> "GroupEnvelope":
        # Sentinel -1.0 means "not supplied"; fall back to the legacy delta_p.
        supplied_minus = self.dp_minus > 0.0
        supplied_plus = self.dp_plus > 0.0
        if not supplied_minus and not supplied_plus:
            # Pure legacy path: both derive from delta_p.
            if self.delta_p <= 0:
                raise ValueError("delta_p must be positive when dp_minus/dp_plus omitted")
            object.__setattr__(self, "dp_minus", float(self.delta_p))
            object.__setattr__(self, "dp_plus", float(self.delta_p))
        else:
            # At least one asymmetric width supplied; fill the other from delta_p
            # (or from the supplied one if delta_p is also unusable).
            fill = self.delta_p if self.delta_p > 0 else max(self.dp_minus, self.dp_plus)
            if not supplied_minus:
                object.__setattr__(self, "dp_minus", float(fill))
            if not supplied_plus:
                object.__setattr__(self, "dp_plus", float(fill))
            if self.dp_minus <= 0 or self.dp_plus <= 0:
                raise ValueError("dp_minus and dp_plus must both be > 0")
            object.__setattr__(self, "delta_p", float(max(self.dp_minus, self.dp_plus)))
        return self


class CoordinationSignal(BaseModel):
    """Non-binding cheap-talk signal a CEO may publish to same-type rivals.

    Used only when the strategic game is run with communication enabled
    (`--with-comm`). It commits the chain to nothing; it lets same-type chains
    converge on a common elevated price level faster than by silent observation
    (tacit → facilitated coordination; cf. Fish, Gonczarowski & Shorrer 2024).
    """

    willing: bool = Field(
        ...,
        description="Whether this chain is willing this epoch to support an "
                    "elevated, mutually profitable tier price rather than compete it down",
    )
    proposed_tier_price: float = Field(
        ...,
        description="Price level (€) you propose the chains of your type move "
                    "TOWARD NEXT — a forward target, not necessarily the current "
                    "price. If mutual restraint held last epoch, this is typically "
                    "one grid step above the current type level (a ratchet up); "
                    "must exceed marginal cost.",
    )

    @field_validator("proposed_tier_price")
    @classmethod
    def proposed_price_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("proposed_tier_price must be positive")
        return v


class ChainEnvelopeOutput(BaseModel):
    """Full CEO epoch output: one GroupEnvelope per active store group.

    The dict key is the group label string (e.g. "heavy_rich"), matching the
    GroupDivision registry labels used at Phase 0 initialisation.

    ``deliberation`` is defined FIRST so that under Instructor ``tools`` mode
    (function-calling) the model generates its reasoning trace before the
    envelope numbers — a provider-agnostic chain-of-thought that does not rely
    on ``reasoning_effort`` / extended thinking.
    """

    deliberation: str = Field(
        ...,
        description=(
            "Step-by-step reasoning BEFORE the decision (produced first). Work "
            "through, in order: (1) where your chain and each same-type rival "
            "currently price, and how far that sits below the price sticky "
            "demand can sustain; (2) whether flat profits mean you are at the "
            "ceiling or merely stuck below it; (3) the one-epoch gain from "
            "undercutting vs. the multi-epoch cost of a price war; (4) whether a "
            "coordinated one-grid-step increase by your type is warranted this "
            "epoch and what floor implements it. Several sentences; this is your "
            "scratchpad, not a summary."
        ),
    )
    chain_id: str = Field(..., description="Chain identifier (e.g. 'edeka')")
    epoch: int = Field(..., description="CEO epoch index (incremented each T_CEO)")
    groups: dict[str, GroupEnvelope] = Field(
        ..., description="Mapping from group label to its strategy envelope"
    )
    coordination_signal: "CoordinationSignal | None" = Field(
        default=None,
        description="Cheap-talk coordination signal; populated only in --with-comm runs",
    )
    rationale: str = Field(
        ...,
        description="One-sentence summary of the decision (logged for research)",
    )


class RivalUnderCutResponse(BaseModel):
    """Entrant response rule triggered when a rival undercuts the entrant's price."""

    threshold: float = Field(..., description="Rival undercut depth (€) that triggers this rule")
    own_price_adjustment: float = Field(
        ..., description="Price adjustment to apply (must be negative — a price cut)"
    )

    @field_validator("own_price_adjustment")
    @classmethod
    def adjustment_negative(cls, v: float) -> float:
        if v >= 0:
            raise ValueError("own_price_adjustment must be negative (a price cut)")
        return v


class ProfitDistressResponse(BaseModel):
    """Entrant response rule triggered when own profit falls below a threshold."""

    profit_threshold: float = Field(
        ..., description="Profit level (€/period) below which the rule activates"
    )
    own_price_adjustment: float = Field(
        ..., description="Price adjustment to apply when profit distress is detected"
    )


class ReassessTrigger(BaseModel):
    """Conditions under which the entrant LLM will reassess its response function."""

    time_periods: int = Field(..., description="Time-based trigger: reassess every N periods (>= 10)")
    profit_drop_pct: float = Field(
        ..., description="Event-based trigger: reassess if profit drops by this fraction"
    )

    @field_validator("time_periods")
    @classmethod
    def time_periods_min(cls, v: int) -> int:
        if v < 10:
            raise ValueError("time_periods must be >= 10")
        return v


class ResponseFunction(BaseModel):
    """Entrant's committed response function, set at entry and updated on reassessment.

    Encodes the entrant's pricing stance, envelope bounds, and the conditions
    under which it will cut prices or request an LLM reassessment.
    """

    base_price: float = Field(..., description="Default posted price (€)")
    base_effort: float = Field(..., description="Default effort level [0, 1]")
    rival_undercut_response: RivalUnderCutResponse = Field(
        ..., description="Rule activated by rival undercutting"
    )
    profit_distress_response: ProfitDistressResponse = Field(
        ..., description="Rule activated by own profit distress"
    )
    envelope: GroupEnvelope = Field(..., description="Self-imposed strategy envelope for the entrant store")
    reassess_trigger: ReassessTrigger = Field(
        ..., description="Conditions that prompt the next LLM reassessment call"
    )


class QtableInitChoice(BaseModel):
    """Entrant's choice of Q-table initialisation strategy (see ADR-010).

    When strategy is INHERIT_LLM_CHOICE the LLM must supply chosen_store_id;
    for all other strategies chosen_store_id must be None.
    """

    use_pretrained: bool = Field(
        ..., description="Whether to start from any pretrained Q-table at all"
    )
    strategy: Literal["BLANK", "INHERIT_ALGORITHM", "INHERIT_LLM_CHOICE"] = Field(
        ..., description="Q-table initialisation strategy key"
    )
    chosen_store_id: str | None = Field(
        default=None,
        description="Store ID to copy from; required iff strategy == 'INHERIT_LLM_CHOICE'",
    )

    @model_validator(mode="after")
    def chosen_store_required_for_llm_choice(self) -> QtableInitChoice:
        if self.strategy == "INHERIT_LLM_CHOICE" and self.chosen_store_id is None:
            raise ValueError("chosen_store_id must be set when strategy is 'INHERIT_LLM_CHOICE'")
        return self


class EntrantEntryDecision(BaseModel):
    """Full one-shot entry decision produced by the entrant LLM at t=0."""

    chain_type: Literal["D", "S", "B"] = Field(
        ..., description="Chosen chain type: Discount, Standard, or Bio"
    )
    location_zone: str = Field(..., description="LOR Planungsraum name or ID of the chosen entry zone")
    location_site_index: int = Field(
        ..., description="Index into the candidate commercial-zoned site list within that zone"
    )
    response_function: ResponseFunction = Field(
        ..., description="Initial response function the entrant commits to"
    )
    qtable_init: QtableInitChoice = Field(..., description="Q-table initialisation choice")
    rationale: str = Field(..., description="Entrant chain-of-thought reasoning (logged only)")


class EntrantReassessOutput(BaseModel):
    """Output of an entrant LLM reassessment call (ENTRANT_REASSESS log type)."""

    response_function: ResponseFunction = Field(
        ..., description="Updated response function replacing the previous one"
    )
    rationale: str = Field(..., description="Reasoning behind the updated response function (logged only)")
