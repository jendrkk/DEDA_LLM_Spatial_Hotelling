"""
LLM-powered agent for the Hotelling simulation.

:class:`LLMAgent` queries a large-language model (via OpenAI-compatible API)
to decide its next location and price.  The prompt includes the current market
state so the model can reason strategically.

The class is designed to be backend-agnostic: any provider that supports the
``openai`` Python SDK (OpenAI, Azure OpenAI, local Ollama, etc.) works out of
the box by adjusting ``base_url`` and ``api_key``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

from hotelling.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a strategic firm competing in a 2-D Hotelling spatial market. "
    "Your objective is to maximise your market share and profit by choosing "
    "your location (x, y) and price. "
    "Respond ONLY with valid JSON: "
    '{"location": [x, y], "price": p} '
    "where x and y are floats in [0, 1] and p > 0."
)


class LLMAgent(BaseAgent):
    """LLM-backed market actor.

    Parameters
    ----------
    firm_id:
        Unique identifier.
    location:
        Initial *(x, y)* position.
    price:
        Initial price.
    model:
        LLM model identifier (e.g. ``"gpt-4o"``).
    api_key:
        API key for the LLM provider.  If *None*, the ``OPENAI_API_KEY``
        environment variable is used.
    base_url:
        Override the base URL (useful for local Ollama or Azure endpoints).
    system_prompt:
        System prompt to prepend to every request.
    max_tokens:
        Maximum tokens in the LLM response.
    temperature:
        Sampling temperature.
    """

    def __init__(
        self,
        firm_id: str,
        location: Tuple[float, float] = (0.5, 0.5),
        price: float = 1.0,
        name: Optional[str] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(firm_id=firm_id, location=location, price=price, name=name)
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = self._build_client(api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------
    # Client factory (lazy import so openai is optional at import time)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(api_key: Optional[str], base_url: Optional[str]):  # noqa: ANN205
        try:
            import openai  # noqa: PLC0415

            kwargs: Dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            return openai.OpenAI(**kwargs)
        except ImportError:
            logger.warning(
                "openai package not installed. LLMAgent will return fallback actions."
            )
            return None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self._client is None:
            # Fallback: stay put
            return {"location": self.location, "price": self.price}

        user_message = self._format_user_message(state)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content or ""
            return self._parse_response(content, state)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM call failed for %s: %s", self.firm_id, exc)
            return {"location": self.location, "price": self.price}

    # ------------------------------------------------------------------
    # Prompt / parsing helpers
    # ------------------------------------------------------------------

    def _format_user_message(self, state: Dict[str, Any]) -> str:
        city = state["city"]
        me = state["self"]
        competitors = state.get("firms", [])
        comp_str = json.dumps(competitors, indent=2)
        return (
            f"Market step {state['step']}.\n"
            f"City size: width={city['width']}, height={city['height']}.\n"
            f"Your current location: {me['location']}, price: {me['price']:.3f}, "
            f"market share: {me['market_share']:.3%}.\n"
            f"Competitors:\n{comp_str}\n"
            "Choose your next location and price to maximise profit."
        )

    def _parse_response(self, content: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract JSON from the LLM response, falling back gracefully."""
        city = state["city"]
        try:
            # Strip markdown code fences if present
            content = re.sub(r"```[a-z]*\n?", "", content).strip()
            data = json.loads(content)
            loc = data["location"]
            x = float(np.clip(loc[0], 0.0, city["width"]))  # type: ignore[name-defined]
            y = float(np.clip(loc[1], 0.0, city["height"]))  # type: ignore[name-defined]
            price = float(max(data["price"], 0.01))
            return {"location": (x, y), "price": price}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse LLM response (%s): %s", exc, content[:200])
            return {"location": self.location, "price": self.price}


# Lazy numpy import removed – numpy is now imported at the top of the module.
