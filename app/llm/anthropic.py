"""Anthropic Messages API adapter for the provider-independent LLM contract."""

import os

from anthropic import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    Anthropic as AnthropicClient,
    AuthenticationError,
)

from app.llm import LLMProvider, LLMRequest, LLMResponse


class AnthropicProviderError(RuntimeError):
    """Safe provider error that does not expose SDK response details."""


class AnthropicLLMProvider:
    """Synchronous Claude provider configured through environment variables."""

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        model = os.environ.get("ANTHROPIC_MODEL", "").strip()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        if not model:
            raise ValueError("ANTHROPIC_MODEL must be set")

        self._model = model
        # Keep calls bounded and predictable; callers receive a sanitized error.
        self._client = AnthropicClient(api_key=api_key, timeout=30.0, max_retries=0)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")

        parameters = {
            "model": self._model,
            "max_tokens": request.max_output_tokens,
            "messages": [{"role": "user", "content": request.user_prompt}],
        }
        if request.system_prompt:
            parameters["system"] = request.system_prompt

        try:
            message = self._client.messages.create(**parameters)
        except AuthenticationError:
            raise AnthropicProviderError("Anthropic authentication failed.") from None
        except APITimeoutError:
            raise AnthropicProviderError("Anthropic request timed out.") from None
        except APIConnectionError:
            raise AnthropicProviderError("Anthropic connection failed.") from None
        except APIError:
            raise AnthropicProviderError("Anthropic API request failed.") from None

        text_blocks = [
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
            and isinstance(getattr(block, "text", None), str)
        ]
        content = "\n".join(text_blocks)
        if not content.strip():
            raise AnthropicProviderError("Anthropic returned no text content.")
        return LLMResponse(content=content)


__all__ = ["AnthropicLLMProvider", "AnthropicProviderError"]
