"""
tests/test_llm.py — Unit tests for services/llm.py.

All tests mock the OpenAI client. No real paid LLM API calls are made.
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import os

from services.llm import (
    OpenAICompatibleProvider,
    generate_answer,
    get_llm_provider,
    set_llm_provider,
    _sanitise_error,
)


class MockChoice:
    def __init__(self, content):
        self.message = MagicMock()
        self.message.content = content


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


def _make_provider(api_key="test-key"):
    return OpenAICompatibleProvider(
        api_key=api_key,
        model="test-model",
        base_url="https://api.test.example.com/v1",
    )


class TestOpenAICompatibleProvider(unittest.TestCase):

    def _mock_openai(self, response_content="This is a test answer."):
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = MockResponse(response_content)
        mock_openai_cls = MagicMock(return_value=mock_client_instance)
        return mock_openai_cls, mock_client_instance

    def test_successful_generation(self):
        """Provider returns generated answer on success."""
        mock_cls, mock_client = self._mock_openai("Authentication is handled in auth.py.")
        with patch("services.llm.OpenAICompatibleProvider.generate.__wrapped__", create=True):
            pass
        provider = _make_provider()
        with patch("openai.OpenAI", mock_cls):
            result = provider.generate(
                question="How does auth work?",
                context_chunks=[{
                    "file_path": "auth.py",
                    "start_line": 1,
                    "end_line": 20,
                    "language": "Python",
                    "content": "def login(): pass",
                    "score": 0.9,
                }],
                repository="owner/repo",
            )
        self.assertIn("auth.py", result)
        mock_client.chat.completions.create.assert_called_once()

    def test_empty_context_chunks(self):
        """Provider handles empty context chunks gracefully."""
        mock_cls, _ = self._mock_openai("The provided repository evidence does not contain enough information.")
        provider = _make_provider()
        with patch("openai.OpenAI", mock_cls):
            result = provider.generate(
                question="What does this app do?",
                context_chunks=[],
                repository="owner/repo",
            )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_missing_api_key_raises_runtime_error(self):
        """Provider raises RuntimeError when API key is not set."""
        provider = OpenAICompatibleProvider(
            api_key="",
            model="test-model",
            base_url="https://api.test.example.com/v1",
        )
        with self.assertRaises(RuntimeError) as ctx:
            provider.generate("question", [], "owner/repo")
        self.assertIn("LLM_API_KEY", str(ctx.exception))

    def test_provider_api_error_raises_runtime_error(self):
        """Provider wraps API exceptions in RuntimeError."""
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = Exception("Connection refused")
        mock_cls = MagicMock(return_value=mock_client_instance)
        provider = _make_provider()
        with patch("openai.OpenAI", mock_cls):
            with self.assertRaises(RuntimeError) as ctx:
                provider.generate("question", [], "owner/repo")
        self.assertIn("LLM provider error", str(ctx.exception))

    def test_malformed_response_raises_runtime_error(self):
        """Provider raises RuntimeError on malformed provider response."""
        mock_response = MagicMock()
        mock_response.choices = []  # Empty choices list
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client_instance)
        provider = _make_provider()
        with patch("openai.OpenAI", mock_cls):
            with self.assertRaises(RuntimeError) as ctx:
                provider.generate("question", [], "owner/repo")
        self.assertIn("Malformed LLM response", str(ctx.exception))

    def test_none_message_content_raises_runtime_error(self):
        """Provider raises RuntimeError when message content is None."""
        mock_response = MockResponse(None)
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client_instance)
        provider = _make_provider()
        with patch("openai.OpenAI", mock_cls):
            with self.assertRaises(RuntimeError):
                provider.generate("question", [], "owner/repo")

    def test_no_credential_leakage_in_error(self):
        """Error messages do not contain the API key."""
        fake_key = "sk-" + "X" * 48
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = Exception(
            f"Unauthorized: {fake_key}"
        )
        mock_cls = MagicMock(return_value=mock_client_instance)
        provider = OpenAICompatibleProvider(
            api_key=fake_key,
            model="test-model",
            base_url="https://api.test.example.com/v1",
        )
        with patch("openai.OpenAI", mock_cls):
            with self.assertRaises(RuntimeError) as ctx:
                provider.generate("question", [], "owner/repo")
        self.assertNotIn(fake_key, str(ctx.exception))

    def test_context_truncation_at_max_chars(self):
        """Build context truncates cleanly when exceeding MAX_CONTEXT_CHARS."""
        mock_cls, mock_client = self._mock_openai("Truncated context answer.")
        provider = _make_provider()
        # Create many large chunks
        big_chunks = [
            {
                "file_path": f"file_{i}.py",
                "start_line": 1,
                "end_line": 100,
                "language": "Python",
                "content": "x" * 5000,
                "score": 0.9 - i * 0.01,
            }
            for i in range(20)
        ]
        with patch("openai.OpenAI", mock_cls):
            result = provider.generate(
                question="Explain the codebase",
                context_chunks=big_chunks,
                repository="owner/repo",
            )
        # Should not raise, and the call should contain truncation notice
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        self.assertIn("omitted", user_msg)


class TestSanitiseError(unittest.TestCase):

    def test_redacts_long_token_strings(self):
        token = "ghp_" + "A" * 36
        result = _sanitise_error(f"Unauthorized: {token}")
        self.assertNotIn(token, result)
        self.assertIn("[REDACTED]", result)

    def test_preserves_short_words(self):
        result = _sanitise_error("Connection refused by server")
        self.assertIn("Connection refused by server", result)


class TestModuleSingleton(unittest.TestCase):

    def setUp(self):
        set_llm_provider(None)  # Reset singleton

    def tearDown(self):
        set_llm_provider(None)

    def test_set_and_get_provider(self):
        """set_llm_provider / get_llm_provider singleton injection works."""
        mock_provider = MagicMock()
        set_llm_provider(mock_provider)
        self.assertIs(get_llm_provider(), mock_provider)

    def test_generate_answer_delegates_to_provider(self):
        """generate_answer calls the injected provider."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Mocked answer"
        set_llm_provider(mock_provider)

        result = generate_answer(
            question="How does auth work?",
            context_chunks=[{"file_path": "a.py", "content": "pass"}],
            repository="owner/repo",
        )
        self.assertEqual(result, "Mocked answer")
        mock_provider.generate.assert_called_once()

    def test_generate_answer_raises_on_empty_question(self):
        """generate_answer raises ValueError for empty questions."""
        mock_provider = MagicMock()
        set_llm_provider(mock_provider)
        with self.assertRaises(ValueError):
            generate_answer("", [], "owner/repo")

    def test_generate_answer_raises_on_whitespace_question(self):
        """generate_answer raises ValueError for whitespace-only questions."""
        mock_provider = MagicMock()
        set_llm_provider(mock_provider)
        with self.assertRaises(ValueError):
            generate_answer("   ", [], "owner/repo")


if __name__ == "__main__":
    unittest.main()
