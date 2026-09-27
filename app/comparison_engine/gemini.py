"""
app/comparison/gemini.py

A thin gateway around Gemini.

CONTRACT
- It returns parsed JSON, or it raises LLMUnavailable.
- It never interprets, never counts, never classifies.
- It retries transient failures with backoff and a repair hint.

Every caller of this module is expected to survive LLMUnavailable with a
deterministic fallback. Nothing in this engine is allowed to make a
comparison fail outright because a model call misbehaved.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Deterministic-as-possible settings.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 7
DEFAULT_MAX_ATTEMPTS = 1


def _comparison_timeout_ms() -> int:
    raw_value = os.getenv("GEMINI_COMPARISON_TIMEOUT_SECONDS", "20")
    try:
        seconds = int(raw_value)
    except (TypeError, ValueError):
        seconds = 20
    return max(5, seconds) * 1000


class LLMUnavailable(RuntimeError):
    """Raised when the model could not return usable JSON."""


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


class GeminiGateway:
    """
    Wraps google-genai. Injectable so the engine is testable without a
    network call: pass any object exposing `generate_json`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int = DEFAULT_SEED,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        from google import genai  # imported lazily: keeps tests offline

        if api_key is None or model is None:
            from app.config import settings

            api_key = api_key or getattr(settings, "GEMINI_API_KEY", None)
            model = model or getattr(
                settings, "GEMINI_MODEL", "gemini-2.5-pro"
            )

        if not api_key:
            raise LLMUnavailable("GEMINI_API_KEY is not configured.")

        self._genai = genai
        self._api_key = api_key
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_attempts = max(1, max_attempts)

        self.call_count = 0
        self.failure_count = 0

    # ------------------------------------------------------------------

    def generate_json(
        self,
        system_instruction: str,
        user_content: str,
        response_schema: Any,
        label: str = "gemini_call",
    ) -> Dict[str, Any]:
        """
        Send one request and return the parsed JSON object.

        `response_schema` is a pydantic model class; google-genai converts
        it to a response schema and constrains decoding to it.
        """

        from google.genai import types

        content = user_content
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            self.call_count += 1

            try:
                model_name = str(self.model or "").lower()
                thinking_config = None
                if model_name.startswith("gemini-3"):
                    thinking_config = types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL,
                    )
                elif "gemini-2.5-flash" in model_name:
                    thinking_config = types.ThinkingConfig(
                        thinking_budget=0,
                    )

                with self._genai.Client(api_key=self._api_key) as client:
                    response = client.models.generate_content(
                        model=self.model,
                        contents=content,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=self.temperature,
                            seed=self.seed,
                            response_mime_type="application/json",
                            response_schema=response_schema,
                            thinking_config=thinking_config,
                            automatic_function_calling=(
                                types.AutomaticFunctionCallingConfig(
                                    disable=True
                                )
                            ),
                            http_options=types.HttpOptions(
                                timeout=_comparison_timeout_ms(),
                            ),
                        ),
                    )

                parsed = getattr(response, "parsed", None)

                if parsed is not None:
                    if hasattr(parsed, "model_dump"):
                        return parsed.model_dump(mode="json")
                    if isinstance(parsed, dict):
                        return parsed

                raw = getattr(response, "text", None)

                if not raw:
                    raise ValueError("Gemini returned an empty response.")

                return json.loads(_strip_code_fence(raw))

            except Exception as exc:
                last_error = exc
                self.failure_count += 1

                logger.warning(
                    "%s failed on attempt %s/%s: %s",
                    label,
                    attempt,
                    self.max_attempts,
                    exc,
                )

                if attempt >= self.max_attempts:
                    break

                # Backoff with jitter, then append a repair hint so the
                # retry is not a blind repeat of the same request.
                time.sleep((2 ** (attempt - 1)) + random.uniform(0, 0.4))

                content = (
                    f"{user_content}\n\n"
                    "PREVIOUS ATTEMPT FAILED VALIDATION. "
                    "Return ONLY valid JSON matching the schema exactly. "
                    "Do not add prose, markdown or trailing commentary."
                )

        raise LLMUnavailable(
            f"{label} did not produce usable JSON after "
            f"{self.max_attempts} attempts."
        ) from last_error


__all__ = ["GeminiGateway", "LLMUnavailable"]
