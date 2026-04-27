"""LLM agent via LiteLLM + Instructor for provider-agnostic structured outputs.

Responsibility: implement an LLM-backed pricing agent with structured output
validation and JSONL call logging.

Public API: LLMAgent

Key dependencies: hotelling.llm.client, hotelling.llm.schemas

References:
    LiteLLM;
    Instructor library.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from hotelling.agents.base import Action, Observation, Transition


class LLMAgent:
    """LLM-backed market agent using LiteLLM + Instructor.

    Parameters
    ----------
    firm_id: str
    model: str - pinned model snapshot (e.g. "gpt-4o-2024-08-06")
    temperature: float - sampling temperature (use 0 for deterministic)
    max_tokens: int
    max_retries: int
    log_path: Optional[Path] - JSONL file to log all LLM calls
    """

    def __init__(
        self,
        firm_id: str,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0,
        max_tokens: int = 512,
        max_retries: int = 3,
        log_path: Optional[Path] = None,
    ) -> None:
        self.firm_id = firm_id
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.log_path = log_path
        self._client: Any = None

    def reset(self, info: Dict[str, Any]) -> None:
        """Initialize LiteLLM + Instructor client."""
        raise NotImplementedError

    def act(self, observation: Observation) -> Action:
        """Query LLM for pricing decision, log call to JSONL."""
        raise NotImplementedError

    def update(self, transition: Transition) -> None:
        pass

    def _log_call(
        self,
        prompt: str,
        response: str,
        tokens: int,
        elapsed: float,
    ) -> None:
        """Append a JSONL record to log_path."""
        raise NotImplementedError
