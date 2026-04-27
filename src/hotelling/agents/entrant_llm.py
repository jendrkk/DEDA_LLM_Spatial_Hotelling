"""LLM entrant agent: entry decision, response function, and reassessment.

Responsibility: (a) make the one-shot entry decision at Phase 1 (location,
chain type, initial response function, Q-table init strategy); (b) execute
the committed response function deterministically each period; (c) reassess
the response function on time- or event-triggered calls (see ADR-011).

Public API: EntrantLLM

Key dependencies: hotelling.llm.client, hotelling.llm.schemas

References: ADR-010, ADR-011; docs/agent_simulation_technical_report.md §7.
"""
from __future__ import annotations

from hotelling.llm.schemas import (
    EntrantEntryDecision,
    EntrantReassessOutput,
    ResponseFunction,
)


class EntrantLLM:
    """LLM-backed entrant: entry decision and response function execution.

    Parameters
    ----------
    model : str  Pinned LLM model snapshot string.
    sunk_costs : dict[str, float]  Entry sunk cost by chain type (D/S/B).
    marginal_costs : dict[str, float]  Marginal cost by chain type (D/S/B).
    """

    def __init__(
        self,
        model: str,
        sunk_costs: dict[str, float],
        marginal_costs: dict[str, float],
    ) -> None:
        self.model = model
        self.sunk_costs = sunk_costs
        self.marginal_costs = marginal_costs

    def decide_entry(self, market_state: dict) -> EntrantEntryDecision:
        """Query the LLM for the one-shot entry decision at Phase 1."""
        raise NotImplementedError

    def execute_response_function(
        self,
        own_state: dict,
        response_fn: ResponseFunction,
    ) -> tuple[float, float]:
        """Apply the committed response function; return (price, effort)."""
        raise NotImplementedError

    def reassess(
        self,
        own_state: dict,
        market_state: dict,
        trigger_reason: str,
    ) -> EntrantReassessOutput:
        """Query the LLM to update the response function on trigger."""
        raise NotImplementedError
