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
        threshold = float(self.params.get("status_threshold", 0.5))
        return "RICH" if store_metadata.get("social_index", 0.5) >= threshold else "POOR"

    def description(self) -> str:
        threshold = float(self.params.get("status_threshold", 0.5))
        return (
            f"Neighbourhood income: stores whose LOR social-status index S_r >= "
            f"{threshold:.2f} are RICH, otherwise POOR."
        )


REGISTRY["DIVISION_NEIGHBOURHOOD"] = NeighbourhoodDivision
