"""Pydantic schemas for LLM structured outputs.

Responsibility: define validated data models for all LLM agent decisions —
CEO strategy envelopes, entrant entry decisions, entrant reassessment outputs,
and the response functions / Q-table init choices they embed.  Used with
Instructor to enforce structured JSON responses from the LLM backends.

Public API:
    GroupEnvelope, ChainEnvelopeOutput,
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

    Defines the price/effort target midpoints and half-widths that bound the
    Q-learning store agents operating in this group, plus the exploration rate
    the RL agents should use within the envelope.
    """

    p_bar: float = Field(..., description="Target price midpoint (€)")
    delta_p: float = Field(..., description="Price half-width (€); minimum 0.05")
    e_bar: float = Field(..., description="Target effort midpoint [0, 1]")
    delta_e: float = Field(..., description="Effort half-width; minimum 0.05")
    epsilon: float = Field(..., description="RL exploration rate for this group (0, 0.5)")

    @field_validator("p_bar")
    @classmethod
    def p_bar_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("p_bar must be positive")
        return v

    @field_validator("delta_p")
    @classmethod
    def delta_p_min(cls, v: float) -> float:
        if v < 0.05:
            raise ValueError("delta_p must be >= 0.05")
        return v

    @field_validator("delta_e")
    @classmethod
    def delta_e_min(cls, v: float) -> float:
        if v < 0.05:
            raise ValueError("delta_e must be >= 0.05")
        return v

    @field_validator("epsilon")
    @classmethod
    def epsilon_range(cls, v: float) -> float:
        if not (0.0 < v < 0.5):
            raise ValueError("epsilon must be in (0.0, 0.5)")
        return v


class ChainEnvelopeOutput(BaseModel):
    """Full CEO epoch output: one GroupEnvelope per active store group.

    The dict key is the group label string (e.g. "heavy_rich"), matching the
    GroupDivision registry labels used at Phase 0 initialisation.
    """

    chain_id: str = Field(..., description="Chain identifier (e.g. 'edeka')")
    epoch: int = Field(..., description="CEO epoch index (incremented each T_CEO)")
    groups: dict[str, GroupEnvelope] = Field(
        ..., description="Mapping from group label to its strategy envelope"
    )
    rationale: str = Field(..., description="CEO chain-of-thought reasoning (logged only)")


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
