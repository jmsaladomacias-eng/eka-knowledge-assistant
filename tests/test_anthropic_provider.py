import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.llm import LLMProvider, LLMRequest, LLMResponse
from app.llm.anthropic import AnthropicLLMProvider, AnthropicProviderError


class AnthropicProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "synthetic-test-key", "ANTHROPIC_MODEL": "synthetic-model"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.client = Mock()
        self.client.messages.create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Synthetic response"),
                SimpleNamespace(type="tool_use", text=None),
            ]
        )
        self.client_factory = patch(
            "app.llm.anthropic.AnthropicClient", return_value=self.client
        )
        self.client_factory_mock = self.client_factory.start()
        self.addCleanup(self.client_factory.stop)

    def test_maps_request_and_extracts_text_response(self):
        provider = AnthropicLLMProvider()
        request = LLMRequest(
            system_prompt="Use only synthetic test evidence.",
            user_prompt="Reply with the test phrase.",
            max_output_tokens=37,
        )

        response = provider.generate(request)

        self.assertEqual(response, LLMResponse("Synthetic response"))
        self.client.messages.create.assert_called_once_with(
            model="synthetic-model",
            max_tokens=37,
            messages=[{"role": "user", "content": "Reply with the test phrase."}],
            system="Use only synthetic test evidence.",
        )

    def test_empty_system_prompt_is_omitted(self):
        provider = AnthropicLLMProvider()
        provider.generate(LLMRequest(system_prompt="", user_prompt="Synthetic question"))

        self.client.messages.create.assert_called_once_with(
            model="synthetic-model",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Synthetic question"}],
        )

    def test_client_configuration_uses_environment_values_and_bounded_timeout(self):
        AnthropicLLMProvider()

        self.client_factory_mock.assert_called_once_with(
            api_key="synthetic-test-key", timeout=30.0, max_retries=0
        )

    def test_missing_api_key_or_model_is_rejected(self):
        for variable, message in (
            ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY must be set"),
            ("ANTHROPIC_MODEL", "ANTHROPIC_MODEL must be set"),
        ):
            with self.subTest(variable=variable):
                with patch.dict(os.environ, {variable: ""}):
                    with self.assertRaisesRegex(ValueError, message):
                        AnthropicLLMProvider()
        self.client_factory_mock.assert_not_called()

    def test_invalid_request_type_is_rejected_without_api_call(self):
        provider = AnthropicLLMProvider()
        with self.assertRaisesRegex(TypeError, "request must be an LLMRequest"):
            provider.generate("Synthetic question")
        self.client.messages.create.assert_not_called()

    def test_authentication_error_is_sanitized(self):
        provider = AnthropicLLMProvider()

        class FakeAuthenticationError(Exception):
            pass

        self.client.messages.create.side_effect = FakeAuthenticationError(
            "secret synthetic key and response headers"
        )
        with patch("app.llm.anthropic.AuthenticationError", FakeAuthenticationError):
            with self.assertRaisesRegex(AnthropicProviderError, "authentication failed") as caught:
                provider.generate(LLMRequest(system_prompt="", user_prompt="Synthetic prompt"))
        self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn("headers", str(caught.exception))

    def test_timeout_error_is_sanitized(self):
        provider = AnthropicLLMProvider()

        class FakeTimeoutError(Exception):
            pass

        self.client.messages.create.side_effect = FakeTimeoutError("sensitive timeout details")
        with patch("app.llm.anthropic.APITimeoutError", FakeTimeoutError):
            with self.assertRaisesRegex(AnthropicProviderError, "timed out"):
                provider.generate(LLMRequest(system_prompt="", user_prompt="Synthetic prompt"))

    def test_connection_error_is_sanitized(self):
        provider = AnthropicLLMProvider()

        class FakeConnectionError(Exception):
            pass

        self.client.messages.create.side_effect = FakeConnectionError("sensitive connection detail")
        with patch("app.llm.anthropic.APIConnectionError", FakeConnectionError):
            with self.assertRaisesRegex(AnthropicProviderError, "connection failed") as caught:
                provider.generate(LLMRequest(system_prompt="", user_prompt="Synthetic prompt"))
        self.assertNotIn("sensitive", str(caught.exception))

    def test_api_error_is_sanitized(self):
        provider = AnthropicLLMProvider()

        class FakeAPIError(Exception):
            pass

        self.client.messages.create.side_effect = FakeAPIError("sensitive response headers")
        with patch("app.llm.anthropic.APIError", FakeAPIError):
            with self.assertRaisesRegex(AnthropicProviderError, "API request failed") as caught:
                provider.generate(LLMRequest(system_prompt="", user_prompt="Synthetic prompt"))
        self.assertNotIn("headers", str(caught.exception))

    def test_provider_implements_shared_protocol(self):
        self.assertIsInstance(AnthropicLLMProvider(), LLMProvider)

    def test_empty_text_response_is_rejected_safely(self):
        self.client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", text=None)]
        )
        with self.assertRaisesRegex(AnthropicProviderError, "no text content"):
            AnthropicLLMProvider().generate(
                LLMRequest(system_prompt="", user_prompt="Synthetic prompt")
            )


if __name__ == "__main__":
    unittest.main()
