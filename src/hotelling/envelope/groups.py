"""Group division abstract base class and store-group assignment logic.

Responsibility: define the GroupDivision interface, maintain the division
registry, and assign composite group labels to stores at Phase 0 init.

Public API: GroupDivision, REGISTRY, assign_groups

Key dependencies: abc

References: ADR-009; docs/agent_simulation_technical_report.md §4.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

REGISTRY: dict[str, type[GroupDivision]] = {}


class GroupDivision(ABC):
    """Abstract base for group-division strategies.

    Each concrete division classifies a store into one of exactly two
    categories based on static store metadata assigned at Phase 0 init.
    Register concrete subclasses in REGISTRY using the class's ``name`` key.
    """

    name: ClassVar[str]
    categories: ClassVar[tuple[str, str]]

    @abstractmethod
    def assign(self, store_metadata: dict) -> str:
        """Return the category label for a single store."""
        raise NotImplementedError

    @abstractmethod
    def description(self) -> str:
        """Human-readable description included in the CEO system prompt."""
        raise NotImplementedError


def assign_groups(
    stores: list[dict],
    active_divisions: list[str],
) -> dict[str, str]:
    """Assign a composite group label to every store.

    Parameters
    ----------
    stores : list[dict]
        Store metadata dicts; each must include a ``store_id`` key.
    active_divisions : list[str]
        Registry keys of the active divisions (at most 2, per ADR-009).

    Returns
    -------
    dict[str, str]
        Mapping from store_id to composite group label, e.g. ``"HEAVY_RICH"``.
    """
    raise NotImplementedError
