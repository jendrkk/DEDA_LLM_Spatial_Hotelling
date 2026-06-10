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
    kappa0: float - quadratic effort cost coefficient
    size: float - store floor area in m²
    rent: float - per-m² rent (multiplied by size in the cost function)
    fixed_cost: float - Per-period fixed operating cost (lump sum, size-independent),
        e.g. brw-derived rent.  As an additive constant in the profit function it
        does NOT enter the price FOC, so Bertrand-Nash / joint-monopoly prices, the
        Q-table price grid, and Calvano Δ are all invariant to it.  It shifts every
        realised profit level by a constant and becomes behaviourally relevant only
        at the entry/exit margin (Phase 1+).  Included here for correct profit
        accounting and forward-compatibility.  See ADR-022.
    chain: Optional[str] - brand/chain label (e.g. "Rewe", "Lidl")
    chain_type: Optional[str] - chain TYPE classification
        ('discount' | 'standard' | 'bio'), stored explicitly so demand /
        calibration code never re-infers it from the quality value.
    """

    id: str
    location: Tuple[float, float]
    marginal_cost: float
    quality: float
    kappa0: float
    size: float
    rent: float
    fixed_cost: float = 0.0
    chain: Optional[str] = None
    chain_type: Optional[str] = None
