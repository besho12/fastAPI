"""Small OpenAI Responses API gateway with Pydantic Structured Outputs."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from app.comparison_engine.gateway import LLMUnavailable
from app.config import settings


logger = logging.getLogger(__name__)
TModel = TypeVar("TModel", bound=BaseModel)


class OpenAIGateway:
    """Return validated Pydantic models from the OpenAI Responses API.

    The SDK is imported lazily so the deterministic comparison fallback and
    the unit tests can run without making a network request.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.timeout_seconds = max(
            5,
            int(timeout_seconds or settings.OPENAI_REQUEST_TIMEOUT_SECONDS),
        )
        self.max_output_tokens = max(
            256,
            int(max_output_tokens or settings.OPENAI_MAX_OUTPUT_TOKENS),
        )
        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else settings.OPENAI_REASONING_EFFORT
        ).strip().lower()
        self.max_retries = max(
            0,
            int(
                max_retries
                if max_retries is not None
                else settings.OPENAI_MAX_RETRIES
            ),
        )
        self.call_count = 0
        self.failure_count = 0
        self.last_usage: Optional[Dict[str, Any]] = None

        if not self.api_key:
            raise LLMUnavailable("OPENAI_API_KEY is not configured.")
        if not self.model:
            raise LLMUnavailable("OPENAI_MODEL is not configured.")

    def generate_model(
        self,
        *,
        system_instruction: str,
        user_content: str,
        response_schema: Type[TModel],
        label: str = "openai_call",
    ) -> TModel:
        """Generate and validate one structured response."""

        self.call_count += 1

        try:
            from openai import OpenAI

            with OpenAI(
                api_key=self.api_key,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            ) as client:
                request: Dict[str, Any] = {
                    "model": self.model,
                    "input": [
                        {
                            "role": "system",
                            "content": system_instruction,
                        },
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ],
                    "text_format": response_schema,
                    "max_output_tokens": self.max_output_tokens,
                    "store": False,
                }
                if self.reasoning_effort:
                    request["reasoning"] = {
                        "effort": self.reasoning_effort,
                    }

                response = client.responses.parse(
                    **request,
                )

            if getattr(response, "status", None) != "completed":
                error = getattr(response, "error", None)
                incomplete = getattr(response, "incomplete_details", None)
                raise ValueError(
                    f"response status={getattr(response, 'status', None)!r}; "
                    f"error={error!r}; incomplete_details={incomplete!r}"
                )

            usage = getattr(response, "usage", None)
            if usage is not None:
                self.last_usage = (
                    usage.model_dump(mode="json")
                    if hasattr(usage, "model_dump")
                    else dict(usage)
                )
                logger.info(
                    "%s completed with usage=%s",
                    label,
                    self.last_usage,
                )

            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                raise ValueError(
                    "OpenAI returned no structured output (possibly a refusal)."
                )

            if isinstance(parsed, response_schema):
                return parsed

            return response_schema.model_validate(parsed)

        except Exception as exc:
            self.failure_count += 1
            logger.warning("%s failed: %s", label, exc)
            raise LLMUnavailable(f"{label} failed: {exc}") from exc

    def generate_json(
        self,
        system_instruction: str,
        user_content: str,
        response_schema: Type[BaseModel],
        label: str = "openai_call",
    ) -> Dict[str, Any]:
        """Compatibility contract used by the comparison engine."""

        parsed = self.generate_model(
            system_instruction=system_instruction,
            user_content=user_content,
            response_schema=response_schema,
            label=label,
        )
        return parsed.model_dump(mode="json")


__all__ = ["OpenAIGateway"]
