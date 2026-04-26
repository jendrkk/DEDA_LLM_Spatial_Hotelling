"""Pydantic schemas for LLM structured outputs.

Responsibility: define validated data models for LLM pricing and entry
decisions.  Used with Instructor to enforce structured JSON responses.

Public API: FirmDecision, EntryDecision

Key dependencies: pydantic >= 2.7

References:
    Instructor https://python.useinstructor.com/;
    Pydantic v2 https://docs.pydantic.dev/latest/.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class FirmDecision(BaseModel):
    """LLM pricing decision for one simulation period.

    Attributes
    ----------
    price_index : int - discrete price level index in [0, m-1].
        Maps to an actual price via the environment's price grid.
    rationale : str - chain-of-thought reasoning.
        Not used in the simulation itself; logged for analysis.
    """

    price_index: int = Field(
        ...,
        ge=0,
        description="Discrete price level index (0 = lowest, m-1 = highest)",
    )
    rationale: str = Field(
        default="",
        description="Chain-of-thought reasoning (not used in simulation)",
    )


class EntryDecision(BaseModel):
    """LLM entry decision: choose a candidate location and initial price.

    Attributes
    ----------
    location_index : int - index into the list of candidate entry locations
        provided in the observation.
    price_index : int - initial price level index in [0, m-1].
    rationale : str - chain-of-thought reasoning.
    """

    location_index: int = Field(
        ...,
        ge=0,
        description="Candidate entry location index",
    )
    price_index: int = Field(
        ...,
        ge=0,
        description="Initial price level index (0 = lowest, m-1 = highest)",
    )
    rationale: str = Field(
        default="",
        description="Chain-of-thought reasoning (not used in simulation)",
    )
