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
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class RateLimitExceeded(RuntimeError):
    """Raised when the per-day request budget is exhausted."""


class RateLimiter:
    """Sliding-window limiter: <= rpm requests per 60 s and <= rpd per rolling 24 h.

    ``acquire()`` blocks (sleeps) until a per-minute slot is free, and raises
    ``RateLimitExceeded`` once the daily budget is gone (waiting could be hours).
    Uses a monotonic clock and is thread-safe. A limit of 0 disables that dimension.
    """

    def __init__(self, requests_per_minute: int = 15, requests_per_day: int = 1500) -> None:
        self.rpm = int(requests_per_minute or 0)
        self.rpd = int(requests_per_day or 0)
        self._minute: deque[float] = deque()
        self._day: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                m_cut = now - 60.0
                while self._minute and self._minute[0] <= m_cut:
                    self._minute.popleft()
                d_cut = now - 86_400.0
                while self._day and self._day[0] <= d_cut:
                    self._day.popleft()
                if self.rpd and len(self._day) >= self.rpd:
                    reset_h = (self._day[0] + 86_400.0 - now) / 3600.0
                    raise RateLimitExceeded(
                        f"Daily request limit ({self.rpd}) reached; "
                        f"resets in ~{reset_h:.1f} h."
                    )
                if (not self.rpm) or len(self._minute) < self.rpm:
                    self._minute.append(now)
                    self._day.append(now)
                    return
                wait = self._minute[0] + 60.0 - now + 0.05
            time.sleep(max(wait, 0.0))  # sleep OUTSIDE the lock


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
        requests_per_minute: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        force_reasoning_effort: bool = False,
        instructor_mode: str = "json",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.log_path = log_path
        self.base_url = base_url
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.force_reasoning_effort = bool(force_reasoning_effort)
        self.instructor_mode = str(instructor_mode).lower()
        self._reasoning_supported = self._detect_reasoning_support()
        self._client = self._build_client()
        self._log_reasoning_decision()
        self._rate_limiter = (
            RateLimiter(requests_per_minute or 0, requests_per_day or 0)
            if (requests_per_minute or requests_per_day)
            else None
        )

    # ------------------------------------------------------------------
    # Client factory
    # ------------------------------------------------------------------

    def _detect_reasoning_support(self) -> bool:
        """True iff LiteLLM's model DB registers this model as reasoning-capable.

        Defensive across litellm versions; any failure -> False (treat as
        non-reasoning, which is the safe omit-or-force path).
        """
        try:
            import litellm  # noqa: PLC0415

            fn = getattr(litellm, "supports_reasoning", None)
            if fn is None:
                fn = getattr(getattr(litellm, "utils", None), "supports_reasoning", None)
            return bool(fn(model=self.model)) if fn is not None else False
        except Exception:
            return False

    def _log_reasoning_decision(self) -> None:
        import logging  # noqa: PLC0415

        log = logging.getLogger(__name__)
        if self.reasoning_effort is None:
            return
        if self._reasoning_supported:
            log.info("reasoning_effort=%r passed natively (model %s is registered "
                     "reasoning-capable).", self.reasoning_effort, self.model)
        elif self.force_reasoning_effort:
            log.info("reasoning_effort=%r FORCED via allowed_openai_params (model %s "
                     "not in LiteLLM reasoning DB).", self.reasoning_effort, self.model)
        else:
            log.info("reasoning_effort=%r OMITTED: model %s not registered as "
                     "reasoning-capable and force_reasoning_effort=False. Thinking "
                     "uses the provider default. Set force_reasoning_effort=True to "
                     "push it through.", self.reasoning_effort, self.model)

    def _build_client(self) -> Any:
        """Initialize LiteLLM + Instructor with an explicit structured-output mode.

        JSON mode (default) is far more robust than the TOOLS default for
        Gemini/Gemma models, which otherwise emit empty tool calls when thinking
        is active. Falls back to None if the libraries are not installed.
        """
        try:
            import instructor  # noqa: PLC0415
            import litellm  # noqa: PLC0415
        except ImportError:
            return None
        mode_map = {
            "json": instructor.Mode.JSON,
            "md_json": instructor.Mode.MD_JSON,
            "tools": instructor.Mode.TOOLS,
        }
        mode = mode_map.get(getattr(self, "instructor_mode", "json"), instructor.Mode.JSON)
        return instructor.from_litellm(litellm.completion, mode=mode)

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

        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

        start = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
        }
        if self.reasoning_effort is not None:
            if self._reasoning_supported:
                kwargs["reasoning_effort"] = self.reasoning_effort
            elif self.force_reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
                kwargs["allowed_openai_params"] = ["reasoning_effort"]
            # else: omit — model not registered reasoning-capable (logged at init)
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.seed is not None:
            kwargs["seed"] = self.seed
        # Universal safety net: any param LiteLLM does not support for this model
        # is dropped instead of crashing the call. Whitelisted params (above) and
        # supported params are unaffected.
        kwargs.setdefault("drop_params", True)

        if response_model is not None:
            try:
                response = self._client.chat.completions.create(
                    response_model=response_model,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                response = self._complete_json_fallback(kwargs, response_model, exc)
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

    def _complete_json_fallback(self, kwargs: Dict[str, Any], response_model, original_exc):
        """Raw completion + manual JSON extraction when Instructor parsing fails.

        Robust to Gemma structured-mode quirks: extracts the outermost JSON
        object from the response text and validates it against ``response_model``.
        Re-raises the original exception (and logs the raw text) if parsing fails.
        """
        import json  # noqa: PLC0415
        import litellm  # noqa: PLC0415

        raw_kwargs = {k: v for k, v in kwargs.items() if k != "max_retries"}
        raw_kwargs["drop_params"] = True  # keep the raw retry clean of param errors
        content = ""
        try:
            raw = litellm.completion(**raw_kwargs)
            content = raw.choices[0].message.content or ""
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(content[start : end + 1])
                return response_model.model_validate(obj)
            raise ValueError("no JSON object found in fallback content")
        except Exception as exc2:  # noqa: BLE001
            self._log_raw_failure(kwargs.get("messages", []), content, original_exc, exc2)
            raise original_exc

    def _log_raw_failure(self, messages, content, exc1, exc2) -> None:
        """Append a failure record (raw text + both exceptions) to the call log."""
        if self.log_path is None:
            return
        import json  # noqa: PLC0415
        rec = {
            "call_id": str(uuid.uuid4()),
            "model": self.model,
            "status": "FAILED",
            "raw_content": content[:4000],
            "error_structured": repr(exc1)[:500],
            "error_fallback": repr(exc2)[:500],
        }
        log_path = Path(self.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

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
