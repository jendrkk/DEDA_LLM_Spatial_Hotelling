"""Competition-based group division: HEAVY vs EASY by rival store count.

Responsibility: classify each store as HEAVY or EASY competition based on
the number of rival stores within radius R of the store's location.

Public API: CompetitionDivision

Key dependencies: hotelling.envelope.groups

References: ADR-009; docs/agent_simulation_technical_report.md §4.
"""
from __future__ import annotations

from hotelling.envelope.groups import REGISTRY, GroupDivision


class CompetitionDivision(GroupDivision):
    """Divides stores by competitive pressure: HEAVY (many rivals) vs EASY."""

    name = "DIVISION_COMPETITION"
    categories = ("HEAVY", "EASY")

    def assign(self, store_metadata: dict) -> str:
        raise NotImplementedError

    def description(self) -> str:
        raise NotImplementedError


REGISTRY["DIVISION_COMPETITION"] = CompetitionDivision
