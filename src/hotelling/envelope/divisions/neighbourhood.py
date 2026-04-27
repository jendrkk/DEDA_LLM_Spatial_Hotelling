"""Neighbourhood-based group division: RICH vs POOR by LOR social-status index.

Responsibility: classify each store as RICH or POOR based on the standardised
LOR Planungsraum social-status index S_r of the store's neighbourhood.

Public API: NeighbourhoodDivision

Key dependencies: hotelling.envelope.groups

References: ADR-009; docs/agent_simulation_technical_report.md §4.
"""
from __future__ import annotations

from hotelling.envelope.groups import REGISTRY, GroupDivision


class NeighbourhoodDivision(GroupDivision):
    """Divides stores by neighbourhood income: RICH (high S_r) vs POOR."""

    name = "DIVISION_NEIGHBOURHOOD"
    categories = ("RICH", "POOR")

    def assign(self, store_metadata: dict) -> str:
        raise NotImplementedError

    def description(self) -> str:
        raise NotImplementedError


REGISTRY["DIVISION_NEIGHBOURHOOD"] = NeighbourhoodDivision
