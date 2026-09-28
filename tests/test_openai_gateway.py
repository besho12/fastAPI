from types import SimpleNamespace
import sys

from pydantic import BaseModel

from app.openai_gateway import OpenAIGateway


class _Result(BaseModel):
    value: str


def test_gateway_uses_low_reasoning_and_configured_retry_limit(monkeypatch):
    captured = {}

    class _OpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.responses = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def parse(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                status="completed",
                usage=None,
                output_parsed=_Result(value="ok"),
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))
    gateway = OpenAIGateway(
        api_key="test-key",
        model="gpt-6-sol",
        reasoning_effort="low",
        max_retries=0,
    )

    result = gateway.generate_model(
        system_instruction="Return the schema.",
        user_content="Input",
        response_schema=_Result,
    )

    assert result == _Result(value="ok")
    assert captured["client"]["max_retries"] == 0
    assert captured["request"]["reasoning"] == {"effort": "low"}
