"""Deterministic, network-free provider for local development and tests."""

from app.llm import LLMRequest, LLMResponse


class MockLLMProvider:
    """Return the same configured response for every valid request."""

    def __init__(self, response_text: str = "Mock response") -> None:
        if not isinstance(response_text, str):
            raise TypeError("response_text must be a string")
        if not response_text.strip():
            raise ValueError("response_text must not be empty")
        self._response_text = response_text

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")
        return LLMResponse(content=self._response_text)
