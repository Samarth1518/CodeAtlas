"""
tests/test_chat_endpoint.py — Integration tests for POST /api/chat.

All LLM calls, embedding calls, and vector store searches are mocked.
No real API calls are made.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from app import create_app
from services.llm import set_llm_provider
from services.embeddings import set_embedding_model


def _make_mock_model(dim=384):
    mock_model = MagicMock()
    def _encode(texts, **kwargs):
        n = len(texts)
        vecs = np.ones((n, dim), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / norms).astype(np.float32)
    mock_model.encode.side_effect = _encode
    return mock_model


SAMPLE_SEARCH_RESULTS = [
    {
        "chunk_id": "chunk-001",
        "file_path": "backend/auth.py",
        "language": "Python",
        "start_line": 1,
        "end_line": 30,
        "content": "def login(user, password): pass",
        "score": 0.92,
        "line_count": 30,
        "char_count": 200,
    },
    {
        "chunk_id": "chunk-002",
        "file_path": "backend/routes/api.py",
        "language": "Python",
        "start_line": 10,
        "end_line": 50,
        "content": "@app.route('/api/login')",
        "score": 0.85,
        "line_count": 40,
        "char_count": 300,
    },
]

MOCK_ANSWER = (
    "## Summary\n\nAuthentication is handled in `backend/auth.py` via the `login()` function."
)


class TestChatEndpoint(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app.config["TESTING"] = True

        # Inject mock embedding model
        self._mock_model = _make_mock_model()
        set_embedding_model(self._mock_model)

        # Inject mock LLM provider
        self._mock_llm = MagicMock()
        self._mock_llm.generate.return_value = MOCK_ANSWER
        set_llm_provider(self._mock_llm)

    def tearDown(self):
        set_embedding_model(None)
        set_llm_provider(None)

    def _post(self, body):
        return self.client.post(
            "/api/chat",
            data=json.dumps(body),
            content_type="application/json",
        )

    # ── Input validation ──────────────────────────────────────────────────────

    def test_missing_body_returns_400(self):
        resp = self.client.post("/api/chat", content_type="application/json")
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])

    def test_missing_repo_url_returns_400(self):
        resp = self._post({"question": "How does auth work?"})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("repo_url", data["error"])

    def test_invalid_repo_url_returns_422(self):
        resp = self._post({
            "repo_url": "not-a-github-url",
            "question": "How does auth work?",
        })
        self.assertEqual(resp.status_code, 422)
        data = resp.get_json()
        self.assertFalse(data["success"])

    def test_non_github_url_returns_422(self):
        resp = self._post({
            "repo_url": "https://gitlab.com/owner/repo",
            "question": "How does auth work?",
        })
        self.assertEqual(resp.status_code, 422)
        data = resp.get_json()
        self.assertFalse(data["success"])

    def test_missing_question_returns_400(self):
        resp = self._post({"repo_url": "https://github.com/owner/repo"})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Question", data["error"])

    def test_empty_question_returns_400(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/repo",
            "question": "   ",
        })
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])

    def test_question_too_long_returns_400(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/repo",
            "question": "x" * 501,
        })
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("too long", data["error"])

    def test_invalid_top_k_defaults(self):
        """Invalid top_k falls back to default without error."""
        with patch("services.rag.default_vector_store.search", return_value=SAMPLE_SEARCH_RESULTS):
            resp = self._post({
                "repo_url": "https://github.com/owner/repo",
                "question": "How does auth work?",
                "top_k": "invalid",
            })
        self.assertEqual(resp.status_code, 200)

    def test_top_k_capped_at_max(self):
        """top_k above 10 is silently capped."""
        mock_search = MagicMock(return_value=SAMPLE_SEARCH_RESULTS)
        with patch("services.rag.default_vector_store.search", mock_search):
            resp = self._post({
                "repo_url": "https://github.com/owner/repo",
                "question": "How does auth work?",
                "top_k": 999,
            })
        self.assertEqual(resp.status_code, 200)
        call_kwargs = mock_search.call_args[1]
        self.assertLessEqual(call_kwargs["top_k"], 10)

    # ── Successful response ───────────────────────────────────────────────────

    def test_successful_response_structure(self):
        """Successful response has success, answer, and sources fields."""
        with patch("services.rag.default_vector_store.search", return_value=SAMPLE_SEARCH_RESULTS):
            resp = self._post({
                "repo_url": "https://github.com/owner/repo",
                "question": "How does authentication work?",
                "top_k": 5,
            })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("answer", data)
        self.assertIn("sources", data)
        self.assertIsInstance(data["answer"], str)
        self.assertIsInstance(data["sources"], list)

    def test_answer_content(self):
        """Returned answer matches mock LLM output."""
        with patch("services.rag.default_vector_store.search", return_value=SAMPLE_SEARCH_RESULTS):
            resp = self._post({
                "repo_url": "https://github.com/owner/repo",
                "question": "How does authentication work?",
            })
        data = resp.get_json()
        self.assertEqual(data["answer"], MOCK_ANSWER)

    def test_source_citation_fields(self):
        """Every source citation has the required fields."""
        with patch("services.rag.default_vector_store.search", return_value=SAMPLE_SEARCH_RESULTS):
            resp = self._post({
                "repo_url": "https://github.com/owner/repo",
                "question": "How does authentication work?",
            })
        data = resp.get_json()
        for src in data["sources"]:
            self.assertIn("file_path", src)
            self.assertIn("start_line", src)
            self.assertIn("end_line", src)
            self.assertIn("language", src)
            self.assertIn("score", src)
            self.assertIn("chunk_id", src)

    def test_empty_search_results_still_returns_200(self):
        """Empty vector store results still return 200 with answer and empty sources."""
        with patch("services.rag.default_vector_store.search", return_value=[]):
            resp = self._post({
                "repo_url": "https://github.com/owner/repo",
                "question": "How does authentication work?",
            })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["sources"], [])

    # ── Error handling ────────────────────────────────────────────────────────

    def test_llm_api_key_missing_returns_503(self):
        """Missing LLM API key returns 503 Service Unavailable."""
        self._mock_llm.generate.side_effect = RuntimeError(
            "LLM_API_KEY is not configured. Set it in your .env file to enable AI answers."
        )
        with patch("services.rag.default_vector_store.search", return_value=SAMPLE_SEARCH_RESULTS):
            resp = self._post({
                "repo_url": "https://github.com/owner/repo",
                "question": "How does auth work?",
            })
        self.assertEqual(resp.status_code, 503)
        data = resp.get_json()
        self.assertFalse(data["success"])

    def test_llm_provider_error_returns_500(self):
        """LLM provider error returns 500."""
        self._mock_llm.generate.side_effect = RuntimeError("LLM provider error: timeout")
        with patch("services.rag.default_vector_store.search", return_value=SAMPLE_SEARCH_RESULTS):
            resp = self._post({
                "repo_url": "https://github.com/owner/repo",
                "question": "How does auth work?",
            })
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertFalse(data["success"])

    def test_github_token_not_in_response(self):
        """GITHUB_TOKEN is never present in any error or success response."""
        import os
        fake_token = "ghp_test_token_never_leak"
        with patch.dict(os.environ, {"GITHUB_TOKEN": fake_token}):
            with patch("services.rag.default_vector_store.search", return_value=SAMPLE_SEARCH_RESULTS):
                resp = self._post({
                    "repo_url": "https://github.com/owner/repo",
                    "question": "How does auth work?",
                })
        response_text = resp.get_data(as_text=True)
        self.assertNotIn(fake_token, response_text)

    def test_llm_api_key_not_in_response(self):
        """LLM_API_KEY is never present in any response."""
        import os
        fake_llm_key = "sk-test-llm-api-key-never-leak"
        with patch.dict(os.environ, {"LLM_API_KEY": fake_llm_key}):
            with patch("services.rag.default_vector_store.search", return_value=SAMPLE_SEARCH_RESULTS):
                resp = self._post({
                    "repo_url": "https://github.com/owner/repo",
                    "question": "How does auth work?",
                })
        response_text = resp.get_data(as_text=True)
        self.assertNotIn(fake_llm_key, response_text)

    def test_url_trailing_slash_normalised(self):
        """Trailing slash in repo URL is handled gracefully."""
        with patch("services.rag.default_vector_store.search", return_value=[]):
            resp = self._post({
                "repo_url": "https://github.com/owner/repo/",
                "question": "question",
            })
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
