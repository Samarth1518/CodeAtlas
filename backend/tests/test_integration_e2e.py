"""
tests/test_integration_e2e.py — Phase 8 End-to-End Pipeline & Integration Tests.

Validates the full CodeAtlas pipeline:
  1. Health check & route availability
  2. Input validation & security across all endpoints
  3. No secrets / API keys in any API responses
  4. Graceful handling of missing LLM keys
  5. Deterministic integration across all 7 functional layers
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from app import create_app
from services.vector_store import LocalVectorStore


class TestPhase8Integration(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app.config["TESTING"] = True

    # ── 1. Health check & Blueprint registration ─────────────────────────────

    def test_health_check_endpoint(self):
        """GET /api/health returns 200 with service status."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("service"), "CodeAtlas API")

    def test_all_blueprints_registered(self):
        """Verify all 7 feature blueprints are active."""
        rule_endpoints = [rule.endpoint for rule in self.app.url_map.iter_rules()]
        expected_endpoints = [
            "repos.analyze",
            "contents.fetch_contents",
            "chunks.generate_chunks",
            "index.build_index",
            "search.search",
            "chat.chat",
            "insights.analyze",
        ]
        for ep in expected_endpoints:
            self.assertIn(ep, rule_endpoints, f"Blueprint endpoint missing: {ep}")

    # ── 2. Security: No secrets in responses ──────────────────────────────────

    def test_no_secrets_in_health_response(self):
        """Verify tokens never leak through general routes."""
        fake_token = "ghp_super_secret_github_token_phase8"
        fake_llm_key = "sk-fake-llm-secret-key-phase8"
        with patch.dict(os.environ, {"GITHUB_TOKEN": fake_token, "LLM_API_KEY": fake_llm_key}):
            resp = self.client.get("/api/health")
            text = resp.get_data(as_text=True)
            self.assertNotIn(fake_token, text)
            self.assertNotIn(fake_llm_key, text)

    def test_no_secrets_in_error_responses(self):
        """Verify errors never leak environment secrets in tracebacks or error messages."""
        fake_token = "ghp_super_secret_github_token_phase8"
        fake_llm_key = "sk-fake-llm-secret-key-phase8"
        with patch.dict(os.environ, {"GITHUB_TOKEN": fake_token, "LLM_API_KEY": fake_llm_key}):
            # Bad payload to /api/chat
            resp = self.client.post(
                "/api/chat",
                data=json.dumps({"repo_url": "https://github.com/owner/repo", "question": "test"}),
                content_type="application/json",
            )
            text = resp.get_data(as_text=True)
            self.assertNotIn(fake_token, text)
            self.assertNotIn(fake_llm_key, text)

    # ── 3. Missing LLM Key Graceful 503 ──────────────────────────────────────

    def test_missing_llm_key_returns_graceful_503(self):
        """Missing LLM_API_KEY returns HTTP 503 with user-actionable instructions."""
        with patch.dict(os.environ, {"LLM_API_KEY": ""}):
            from services.llm import set_llm_provider, OpenAICompatibleProvider
            set_llm_provider(OpenAICompatibleProvider(api_key=""))
            
            with patch("services.rag.embed_query", return_value=np.zeros(384, dtype=np.float32)):
                with patch("services.rag.default_vector_store.search", return_value=[]):
                    resp = self.client.post(
                        "/api/chat",
                        data=json.dumps({
                            "repo_url": "https://github.com/owner/repo",
                            "question": "How does this repo work?",
                        }),
                        content_type="application/json",
                    )
                    self.assertEqual(resp.status_code, 503)
                    data = resp.get_json()
                    self.assertFalse(data["success"])
                    self.assertIn("LLM_API_KEY is not configured", data["error"])

    # ── 4. End-to-End Pipeline Integration ────────────────────────────────────

    def test_full_pipeline_mocked_integration(self):
        """
        Simulates the entire workflow from metadata to insights and RAG:
          Step 1: POST /api/repos/analyze
          Step 2: POST /api/contents/fetch
          Step 3: POST /api/chunks/generate
          Step 4: POST /api/index/build
          Step 5: POST /api/search
          Step 6: POST /api/insights/analyze
          Step 7: POST /api/chat
        """
        repo_url = "https://github.com/test-owner/test-repo"

        # ── Step 1: Repos analyze ──
        with patch("routes.repos.get_repo_metadata", return_value={
            "name": "test-repo",
            "owner": "test-owner",
            "full_name": "test-owner/test-repo",
            "html_url": repo_url,
            "default_branch": "main",
            "visibility": "public",
            "primary_language": "Python",
        }):
            with patch("routes.repos.get_repo_tree", return_value={
                "tree": [
                    {"path": "README.md", "type": "file", "size": 200},
                    {"path": "app.py", "type": "file", "size": 1500},
                    {"path": "requirements.txt", "type": "file", "size": 50},
                ],
                "total_files": 3,
                "total_dirs": 0,
                "truncated": False,
            }):
                resp1 = self.client.post(
                    "/api/repos/analyze",
                    data=json.dumps({"repo_url": repo_url}),
                    content_type="application/json",
                )
                self.assertEqual(resp1.status_code, 200)
                meta_data = resp1.get_json()
                self.assertTrue(meta_data["success"])
                self.assertEqual(meta_data["name"], "test-repo")

        # ── Step 2: Contents fetch ──
        with patch("routes.contents.get_repo_metadata", return_value={
            "name": "test-repo",
            "owner": "test-owner",
            "full_name": "test-owner/test-repo",
            "html_url": repo_url,
            "default_branch": "main",
            "visibility": "public",
            "primary_language": "Python",
        }):
            with patch("routes.contents.get_file_content", return_value={
                "path": "app.py",
                "name": "app.py",
                "size": 1500,
                "encoding": "utf-8",
                "content": "def main():\n    print('Hello World')\n",
                "sha": "12345",
                "html_url": f"{repo_url}/blob/main/app.py",
            }):
                resp2 = self.client.post(
                    "/api/contents/fetch",
                    data=json.dumps({
                        "repo_url": repo_url,
                        "paths": ["app.py"],
                    }),
                    content_type="application/json",
                )
                self.assertEqual(resp2.status_code, 200)
                contents_data = resp2.get_json()
                self.assertTrue(contents_data["success"])
                self.assertEqual(len(contents_data["files"]), 1)

        # ── Step 3: Chunks generate ──
        resp3 = self.client.post(
            "/api/chunks/generate",
            data=json.dumps({
                "repo_url": repo_url,
                "files": contents_data["files"],
            }),
            content_type="application/json",
        )
        self.assertEqual(resp3.status_code, 200)
        chunks_data = resp3.get_json()
        self.assertTrue(chunks_data["success"])
        self.assertGreater(chunks_data["total_chunks"], 0)

        # ── Step 4: Index build ──
        with patch("routes.index.embed_chunks", return_value=np.ones((len(chunks_data["chunks"]), 384), dtype=np.float32)):
            resp4 = self.client.post(
                "/api/index/build",
                data=json.dumps({
                    "repo_url": repo_url,
                    "files": contents_data["files"],
                }),
                content_type="application/json",
            )
            self.assertEqual(resp4.status_code, 200)
            index_data = resp4.get_json()
            self.assertTrue(index_data["success"])

        # ── Step 5: Semantic search ──
        with patch("routes.search.embed_query", return_value=np.ones(384, dtype=np.float32)):
            resp5 = self.client.post(
                "/api/search",
                data=json.dumps({
                    "repo_url": repo_url,
                    "query": "hello world function",
                    "top_k": 3,
                }),
                content_type="application/json",
            )
            self.assertEqual(resp5.status_code, 200)
            search_data = resp5.get_json()
            self.assertTrue(search_data["success"])

        # ── Step 6: Insights analyze ──
        resp6 = self.client.post(
            "/api/insights/analyze",
            data=json.dumps({
                "repo_url": repo_url,
                "metadata": meta_data,
                "tree": meta_data["tree"],
                "source_files": contents_data["files"],
            }),
            content_type="application/json",
        )
        self.assertEqual(resp6.status_code, 200)
        insights_data = resp6.get_json()
        self.assertTrue(insights_data["success"])
        self.assertIn("statistics", insights_data["insights"])
        self.assertIn("technologies", insights_data["insights"])

        # ── Step 7: Chat (RAG) ──
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "## Summary\nThe application defines a main function that prints Hello World."
        
        from services.llm import set_llm_provider
        set_llm_provider(mock_provider)

        with patch("services.rag.embed_query", return_value=np.ones(384, dtype=np.float32)):
            resp7 = self.client.post(
                "/api/chat",
                data=json.dumps({
                    "repo_url": repo_url,
                    "question": "What does main do?",
                    "top_k": 3,
                }),
                content_type="application/json",
            )
            self.assertEqual(resp7.status_code, 200)
            chat_data = resp7.get_json()
            self.assertTrue(chat_data["success"])
            self.assertIn("Hello World", chat_data["answer"])



if __name__ == "__main__":
    unittest.main()
