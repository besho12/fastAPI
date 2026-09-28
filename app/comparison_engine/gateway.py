"""Shared exceptions and contracts for comparison-model gateways."""


class LLMUnavailable(RuntimeError):
    """Raised when a model gateway cannot return a usable result."""


__all__ = ["LLMUnavailable"]
