"""Firm dataclass for 2-D Hotelling competition.

Responsibility: represent a single market actor with location, marginal cost,
quality, and chain attributes.

Public API: Firm

Key dependencies: dataclasses, typing

References: Calvano et al. (2020 AER) §II.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Firm:
    """Immutable firm descriptor.

    Parameters
    ----------
    id: str - unique firm identifier
    location: Tuple[float, float] - (x, y) in metres (EPSG:25833 for Berlin)
    marginal_cost: float - constant marginal cost c >= 0
    quality: float - vertical quality parameter a >= 0
    chain: Optional[str] - brand/chain label (e.g. "Rewe", "Lidl")
    """

    id: str
    location: Tuple[float, float]
    marginal_cost: float
    quality: float
    kappa0: float
    size: float
    rent: float
    chain: Optional[str] = None
