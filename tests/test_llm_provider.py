import unittest

from app.llm import LLMProvider, LLMRequest, LLMResponse
from app.llm.mock import MockLLMProvider


class LLMProviderTests(unittest.TestCase):
    def test_valid_request_and_response(self):
        request = LLMRequest(
            system_prompt="Answer only from supplied evidence.",
            user_prompt="What does the evidence say?",
            max_output_tokens=120,
        )
        provider = MockLLMProvider("Evidence-based mock answer")

        response = provider.generate(request)

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Evidence-based mock answer")

    def test_mock_behaviour_is_deterministic(self):
        provider = MockLLMProvider("Stable response")
        request = LLMRequest(system_prompt="", user_prompt="Question")

        self.assertEqual(provider.generate(request), provider.generate(request))

    def test_empty_user_input_is_rejected(self):
        for user_prompt in ("", " \t\n"):
            with self.subTest(user_prompt=repr(user_prompt)):
                with self.assertRaisesRegex(ValueError, "user_prompt must not be empty"):
                    LLMRequest(system_prompt="", user_prompt=user_prompt)

    def test_invalid_request_fields_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "system_prompt must be a string"):
            LLMRequest(system_prompt=None, user_prompt="Question")
        with self.assertRaisesRegex(TypeError, "user_prompt must be a string"):
            LLMRequest(system_prompt="", user_prompt=None)
        for max_output_tokens in (0, -1, True, "100"):
            with self.subTest(max_output_tokens=max_output_tokens):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    LLMRequest(
                        system_prompt="",
                        user_prompt="Question",
                        max_output_tokens=max_output_tokens,
                    )

    def test_response_structure_is_validated(self):
        self.assertEqual(LLMResponse("Response text").content, "Response text")
        with self.assertRaisesRegex(ValueError, "response content must not be empty"):
            LLMResponse("  ")
        with self.assertRaisesRegex(TypeError, "response content must be a string"):
            LLMResponse(None)

    def test_provider_contract_compliance(self):
        provider = MockLLMProvider()
        self.assertIsInstance(provider, LLMProvider)
        request = LLMRequest(system_prompt="", user_prompt="Question")
        self.assertIsInstance(provider.generate(request), LLMResponse)

    def test_mock_rejects_invalid_request_type(self):
        with self.assertRaisesRegex(TypeError, "request must be an LLMRequest"):
            MockLLMProvider().generate("Question")

    def test_mock_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "response_text must not be empty"):
            MockLLMProvider(" ")
        with self.assertRaisesRegex(TypeError, "response_text must be a string"):
            MockLLMProvider(None)


if __name__ == "__main__":
    unittest.main()
