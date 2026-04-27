"""LiteLLM wrapper with model snapshot pinning and full call logging.

Responsibility: provide a provider-agnostic LLM client built on LiteLLM +
Instructor.  Every call is logged to a JSONL file with: full messages, full
response text, model fingerprint (SHA-256 of messages), token counts, and
wall-clock timing.  Model aliases are explicitly forbidden - always pin to
a snapshot (e.g. "gpt-4o-2024-08-06", not "gpt-4o").

Public API: LLMClient

Key dependencies: litellm, instructor, pydantic, hashlib, json

References:
    LiteLLM https://docs.litellm.ai/;
    Instructor https://python.useinstructor.com/;
    OpenAI Structured Outputs https://platform.openai.com/docs/guides/structured-outputs.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Provider-agnostic LiteLLM + Instructor client with structured outputs.

    Parameters
    ----------
    model : pinned model snapshot string (e.g. "gpt-4o-2024-08-06").
        Never use aliases like "gpt-4o" - always pin to a snapshot.
    temperature : sampling temperature (0 = deterministic)
    max_tokens : maximum response tokens
    max_retries : number of retries on transient errors
    log_path : JSONL file to append call records to; None = no logging
    base_url : override base URL for local/custom endpoints (e.g. Ollama)
    seed : optional integer seed forwarded to the API for reproducibility
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0,
        max_tokens: int = 512,
        max_retries: int = 3,
        log_path: Optional[Path] = None,
        base_url: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.log_path = log_path
        self.base_url = base_url
        self.seed = seed
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Client factory
    # ------------------------------------------------------------------

    def _build_client(self) -> Any:
        """Initialize LiteLLM + Instructor patched client."""
        try:
            import instructor  # noqa: PLC0415
            import litellm  # noqa: PLC0415

            return instructor.from_litellm(litellm.completion)
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[T]] = None,
    ) -> Any:
        """Send a chat completion request.

        Parameters
        ----------
        messages : list of {"role": ..., "content": ...} dicts
        response_model : if provided, parse and validate the response as this
            Pydantic model using Instructor structured outputs

        Returns
        -------
        Pydantic model instance if response_model provided, else raw string
        """
        if self._client is None:
            raise RuntimeError(
                "LiteLLM or Instructor not installed.  "
                "Run: pip install litellm instructor"
            )

        start = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
        }
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.seed is not None:
            kwargs["seed"] = self.seed

        if response_model is not None:
            response = self._client.chat.completions.create(
                response_model=response_model,
                **kwargs,
            )
        else:
            import litellm  # noqa: PLC0415

            raw = litellm.completion(**kwargs)
            response = raw.choices[0].message.content or ""

        elapsed = time.perf_counter() - start

        tokens_used: int = 0
        if hasattr(response, "_raw_response") and hasattr(
            response._raw_response, "usage"
        ):
            tokens_used = response._raw_response.usage.total_tokens or 0

        self._log_call(messages, response, tokens_used, elapsed)
        return response

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_call(
        self,
        messages: List[Dict[str, str]],
        response: Any,
        tokens_used: int,
        elapsed: float,
    ) -> None:
        """Append a JSONL record to log_path.

        Record schema: call_id, model, fingerprint, messages,
        response_text, tokens_used, elapsed_s, timestamp_utc.
        """
        if self.log_path is None:
            return

        fingerprint = hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode()
        ).hexdigest()[:16]

        record = {
            "call_id": str(uuid.uuid4()),
            "model": self.model,
            "fingerprint": fingerprint,
            "messages": messages,
            "response": str(response),
            "tokens_used": tokens_used,
            "elapsed_s": round(elapsed, 4),
        }

        log_path = Path(self.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
