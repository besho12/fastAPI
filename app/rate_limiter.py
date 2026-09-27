"""
app/rate_limiter.py
--------------------
Process-wide rate limiter for Gemini API calls.

Both app/extraction.py (job extraction) and app/job_identity.py (LLM
title-equivalence checks) call the Gemini API. On the free tier this
is limited to a small number of requests per minute (see
GEMINI_FREE_TIER_RPM below) -- exceeding it returns HTTP 429
RESOURCE_EXHAUSTED. This module provides ONE shared, thread-safe rate
limiter that both call sites use, so the pipeline as a whole -- not
just one call site in isolation -- stays under the quota.

This is a simple sliding-window limiter: before allowing a call, it
drops timestamps older than the window, and if the number of calls
still in the window is at or above the limit, it sleeps just long
enough for the oldest call to fall out of the window.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

_DEFAULT_RPM = int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "12"))
_WINDOW_SECONDS = 60.0


class _SlidingWindowRateLimiter:
    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self._max_calls = max_calls
        self._window_seconds = window_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()

            while self._calls and (now - self._calls[0]) >= self._window_seconds:
                self._calls.popleft()

            if len(self._calls) >= self._max_calls:
                sleep_for = self._window_seconds - (now - self._calls[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._calls and (now - self._calls[0]) >= self._window_seconds:
                    self._calls.popleft()

            self._calls.append(time.monotonic())


# Single, process-wide instance shared by every Gemini call site
# (extraction + job identity title checks), set slightly below the
# free-tier limit (15 RPM) to leave headroom.
gemini_rate_limiter = _SlidingWindowRateLimiter(
    max_calls=_DEFAULT_RPM,
    window_seconds=_WINDOW_SECONDS,
)


def call_gemini_with_retry(fn, *, max_retries: int = 3):
    """
    Run `fn` (a zero-arg callable that performs one Gemini API call),
    applying the shared rate limiter before each attempt and retrying
    on HTTP 429 RESOURCE_EXHAUSTED using the server-provided retryDelay
    when available (falls back to a fixed backoff otherwise).

    Any other exception is re-raised immediately, unretried -- this
    function only smooths over quota-exhaustion, not genuine errors.
    """
    from google.genai import errors as genai_errors

    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        gemini_rate_limiter.acquire()

        try:
            return fn()
        except genai_errors.ClientError as exc:
            status_code = getattr(exc, "code", None) or getattr(
                exc, "status_code", None
            )
            is_quota_error = status_code == 429 or "RESOURCE_EXHAUSTED" in str(exc)

            if not is_quota_error or attempt == max_retries:
                raise

            delay = _extract_retry_delay(exc) or (5.0 * (attempt + 1))
            time.sleep(delay)
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("call_gemini_with_retry: unreachable")


def _extract_retry_delay(exc: Exception) -> float | None:
    """
    Best-effort extraction of the server-suggested retry delay (in
    seconds) from a google.genai ClientError's response details, e.g.
    {'@type': '.../google.rpc.RetryInfo', 'retryDelay': '37s'}.
    """
    try:
        details = getattr(exc, "details", None) or {}
        error = details.get("error", details) if isinstance(details, dict) else {}
        for item in error.get("details", []) if isinstance(error, dict) else []:
            retry_delay = item.get("retryDelay")
            if retry_delay and isinstance(retry_delay, str) and retry_delay.endswith("s"):
                return float(retry_delay[:-1])
    except Exception:  # noqa: BLE001
        pass
    return None


__all__ = ["gemini_rate_limiter", "call_gemini_with_retry"]