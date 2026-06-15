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
# NOTE: tenacity is imported lazily inside _build_retrying (it ships with instructor).
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
        timeout: Optional[float] = 120.0,
        transient_max_attempts: int = 5,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        capture_raw: bool = False,
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
        self.timeout = timeout
        self.transient_max_attempts = int(transient_max_attempts)
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self.capture_raw = bool(capture_raw)
        self._reasoning_supported = self._detect_reasoning_support()
        self._client = self._build_client()
        self._log_reasoning_decision()
        self._transient_types = self._transient_exc_types()
        self._retrying = self._build_retrying()
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
            import logging as _logging  # noqa: PLC0415

            import instructor  # noqa: PLC0415
            import litellm  # noqa: PLC0415
        except ImportError:
            return None
        try:
            litellm.suppress_debug_info = True
            _logging.getLogger("LiteLLM").setLevel(_logging.WARNING)
        except Exception:
            pass
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
        """Send a chat completion request with transient-error backoff."""
        if self._client is None:
            raise RuntimeError(
                "LiteLLM or Instructor not installed.  Run: pip install litellm instructor"
            )
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

        start = time.perf_counter()
        kwargs = self._build_kwargs(messages)
        if response_model is not None:
            response, _raw = self._structured_call(kwargs, response_model, capture_raw=False)
        else:
            import litellm  # noqa: PLC0415

            raw = litellm.completion(num_retries=self.transient_max_attempts, **kwargs)
            response = raw.choices[0].message.content or ""

        elapsed = time.perf_counter() - start
        self._log_call(messages, response, self._extract_tokens(response), elapsed)
        return response

    def complete_with_transcript(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
    ) -> tuple[Any, str]:
        """Like ``complete`` but also returns a human-readable transcript of the
        full provider response (reasoning, raw content, parsed envelope, usage)."""
        if self._client is None:
            raise RuntimeError(
                "LiteLLM or Instructor not installed.  Run: pip install litellm instructor"
            )
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

        start = time.perf_counter()
        kwargs = self._build_kwargs(messages)
        parsed, raw = self._structured_call(kwargs, response_model, capture_raw=True)
        elapsed = time.perf_counter() - start
        transcript = self._format_transcript(raw, parsed)
        self._log_call(messages, parsed, self._extract_tokens(parsed), elapsed)
        return parsed, transcript

    def _build_kwargs(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Assemble the per-call LiteLLM kwargs (no max_retries: retries are
        controlled by the tenacity Retrying passed to create())."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.reasoning_effort is not None:
            if self._reasoning_supported:
                kwargs["reasoning_effort"] = self.reasoning_effort
            elif self.force_reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
                kwargs["allowed_openai_params"] = ["reasoning_effort"]
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.seed is not None:
            kwargs["seed"] = self.seed
        kwargs.setdefault("drop_params", True)
        return kwargs

    def _structured_call(self, kwargs, response_model, capture_raw):
        """Structured create with transient backoff; JSON fallback only on
        non-transient (parse/validation) errors. Returns (parsed, raw_or_None)."""
        comp = self._client.chat.completions
        try:
            if capture_raw:
                parsed, raw = comp.create_with_completion(
                    response_model=response_model, max_retries=self._retrying, **kwargs
                )
                return parsed, raw
            parsed = comp.create(
                response_model=response_model, max_retries=self._retrying, **kwargs
            )
            return parsed, None
        except Exception as exc:  # noqa: BLE001
            if self._is_transient(exc):
                raise  # already retried with backoff; let the CEO retain previous
            parsed = self._complete_json_fallback(kwargs, response_model, exc)
            return parsed, None

    @staticmethod
    def _extract_tokens(response: Any) -> int:
        if hasattr(response, "_raw_response") and hasattr(response._raw_response, "usage"):
            try:
                return response._raw_response.usage.total_tokens or 0
            except Exception:
                return 0
        return 0

    def _transient_exc_types(self) -> tuple:
        try:
            import litellm  # noqa: PLC0415

            names = ["InternalServerError", "ServiceUnavailableError",
                     "RateLimitError", "Timeout", "APIConnectionError"]
            return tuple(
                t for n in names
                if isinstance(getattr(litellm, n, None), type)
            )
        except Exception:
            return tuple()

    def _is_transient(self, exc: BaseException) -> bool:
        if self._transient_types and isinstance(exc, self._transient_types):
            return True
        s = repr(exc).lower()
        markers = ("internalservererror", "internal error", "serviceunavailable",
                   "unavailable", "overloaded", "resource_exhausted", "rate limit",
                   "timeout", "code': 500", "code': 503", "code': 429")
        return any(m in s for m in markers)

    def _build_retrying(self):
        import logging  # noqa: PLC0415

        import tenacity  # noqa: PLC0415

        log = logging.getLogger(__name__)

        def _should_retry(retry_state) -> bool:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            return bool(exc is not None and self._is_transient(exc))

        def _before_sleep(retry_state) -> None:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            sleep = getattr(retry_state.next_action, "sleep", 0.0)
            log.warning(
                "Transient LLM error (%s); backoff retry %d/%d in ~%.1fs.",
                type(exc).__name__ if exc else "?",
                retry_state.attempt_number, self.transient_max_attempts, sleep,
            )

        return tenacity.Retrying(
            stop=tenacity.stop_after_attempt(max(1, self.transient_max_attempts)),
            wait=tenacity.wait_random_exponential(
                multiplier=self.backoff_base, max=self.backoff_max
            ),
            retry=_should_retry,
            reraise=True,
            before_sleep=_before_sleep,
        )

    def _format_transcript(self, raw: Any, parsed: Any) -> str:
        reasoning = ""
        content = ""
        usage_line = ""
        try:
            msg = raw.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if not reasoning:
                tb = getattr(msg, "thinking_blocks", None)
                if tb:
                    reasoning = json.dumps(tb, indent=2, default=str)
            u = getattr(raw, "usage", None)
            if u is not None:
                usage_line = (
                    f"prompt_tokens={getattr(u, 'prompt_tokens', None)} "
                    f"completion_tokens={getattr(u, 'completion_tokens', None)} "
                    f"total_tokens={getattr(u, 'total_tokens', None)}"
                )
        except Exception:
            pass
        try:
            parsed_json = parsed.model_dump_json(indent=2)
        except Exception:
            parsed_json = str(parsed)
        return (
            "=== PROVIDER REASONING (thinking) ===\n"
            f"{reasoning or '(none returned by provider)'}\n\n"
            "=== RAW MODEL CONTENT ===\n"
            f"{content or '(empty)'}\n\n"
            "=== PARSED / VALIDATED ENVELOPE ===\n"
            f"{parsed_json}\n\n"
            "=== USAGE ===\n"
            f"{usage_line or '(unavailable)'}\n"
        )

    def _complete_json_fallback(self, kwargs: Dict[str, Any], response_model, original_exc):
        """Raw completion + manual JSON extraction when Instructor parsing fails.

        Robust to Gemma structured-mode quirks: extracts the outermost JSON
        object from the response text and validates it against ``response_model``.
        Re-raises the original exception (and logs the raw text) if parsing fails.
        """
        import json  # noqa: PLC0415
        import litellm  # noqa: PLC0415

        raw_kwargs = {k: v for k, v in kwargs.items() if k != "max_retries"}
        raw_kwargs["drop_params"] = True
        raw_kwargs.setdefault("num_retries", self.transient_max_attempts)
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
