"""Three-phase simulation controller stubs.

Responsibility: orchestrate Phase 0 (burn-in), Phase 1 (entry), and
Phase 2 (strategic game), delegating to stores, CEOs, and the entrant.

Public API: Phase0BurnIn, Phase1Entry, Phase2StrategicGame

Key dependencies: hotelling.agents, hotelling.llm.schemas

References: ADR-006; docs/agent_simulation_technical_report.md §8.
"""
from __future__ import annotations

from hotelling.agents.entrant_llm import EntrantLLM
from hotelling.llm.schemas import EntrantEntryDecision


class Phase0BurnIn:
    """Burn-in phase: incumbent Q-learners converge without CEO or entrant.

    Parameters
    ----------
    config : dict  Phase 0 config slice (T_burnin, convergence threshold).
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self, stores: list, env: object) -> dict:
        """Run burn-in until convergence; return convergence statistics."""
        raise NotImplementedError


class Phase1Entry:
    """Entry phase: entrant LLM makes the one-shot entry decision.

    Parameters
    ----------
    config : dict  Phase 1 config slice.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self, entrant: EntrantLLM, market_state: dict) -> EntrantEntryDecision:
        """Invoke the entrant entry decision and return the validated output."""
        raise NotImplementedError


class Phase2StrategicGame:
    """Strategic game phase: CEOs + entrant reassess; stores learn tactically.

    Parameters
    ----------
    config : dict  Phase 2 config slice (T_game, T_CEO, T_entrant).
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(
        self,
        stores: list,
        ceos: list,
        entrant: EntrantLLM,
        env: object,
    ) -> dict:
        """Run the strategic game; return per-period history dict."""
        raise NotImplementedError
