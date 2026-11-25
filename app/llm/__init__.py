"""Provider-independent interface for language model generation."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMRequest:
    """Single-turn input shared by all generation providers."""

    system_prompt: str
    user_prompt: str
    max_output_tokens: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str):
            raise TypeError("system_prompt must be a string")
        if not isinstance(self.user_prompt, str):
            raise TypeError("user_prompt must be a string")
        if not self.user_prompt.strip():
            raise ValueError("user_prompt must not be empty")
        if (
            not isinstance(self.max_output_tokens, int)
            or isinstance(self.max_output_tokens, bool)
            or self.max_output_tokens < 1
        ):
            raise ValueError("max_output_tokens must be a positive integer")


@dataclass(frozen=True)
class LLMResponse:
    """Generated text returned by a provider."""

    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("response content must be a string")
        if not self.content.strip():
            raise ValueError("response content must not be empty")


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal structural contract implemented by each LLM provider."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for one validated request."""
        ...


__all__ = ["LLMProvider", "LLMRequest", "LLMResponse"]
