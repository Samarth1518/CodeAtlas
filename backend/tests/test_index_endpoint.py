"""
tests/test_index_endpoint.py — Unit tests for POST /api/index/build endpoint.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import create_app
from services.embeddings import set_embedding_model
from services.github import GitHubAPIError


class IndexEndpointTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Mock embedding model
        self.mock_model = MagicMock()
        def mock_encode(texts, **kwargs):
            return np.ones((len(texts), 384), dtype=np.float32)
        self.mock_model.encode.side_effect = mock_encode
        set_embedding_model(self.mock_model)

    def tearDown(self):
        set_embedding_model(None)

    def test_missing_body_returns_400(self):
        resp = self.client.post("/api/index/build", data="not json", content_type="text/plain")
        self.assertEqual(resp.status_code, 400)

    def test_missing_repo_url_returns_400(self):
        resp = self.client.post("/api/index/build", json={"files": []})
        self.assertEqual(resp.status_code, 400)

    def test_invalid_repo_url_returns_422(self):
        resp = self.client.post("/api/index/build", json={"repo_url": "invalid-url", "files": []})
        self.assertEqual(resp.status_code, 422)

    def test_neither_files_nor_paths_returns_400(self):
        resp = self.client.post("/api/index/build", json={"repo_url": "https://github.com/owner/repo"})
        self.assertEqual(resp.status_code, 400)

    def test_successful_build_with_direct_files(self):
        payload = {
            "repo_url": "https://github.com/owner/repo",
            "files": [
                {
                    "path": "src/main.py",
                    "content": "def main():\n    print('CodeAtlas')\n",
                    "language": "Python",
                },
                {
                    "path": "README.md",
                    "content": "# CodeAtlas\n\nAI Code Intelligence.\n",
                    "language": "Markdown",
                },
            ],
        }
        resp = self.client.post("/api/index/build", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["repo_url"], "https://github.com/owner/repo")
        self.assertIn("summary", data)
        self.assertEqual(data["summary"]["files_processed"], 2)
        self.assertGreater(data["summary"]["chunks_indexed"], 0)

    def test_sensitive_files_rejected_from_indexing(self):
        payload = {
            "repo_url": "https://github.com/owner/repo",
            "files": [
                {
                    "path": ".env",
                    "content": "API_SECRET=supersecret123",
                },
                {
                    "path": "src/app.py",
                    "content": "print('hello')",
                    "language": "Python",
                },
            ],
        }
        resp = self.client.post("/api/index/build", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["summary"]["files_processed"], 1)

    @patch("routes.index.get_file_content")
    @patch("routes.index.get_repo_metadata")
    def test_successful_build_with_paths_mode(self, mock_meta, mock_content):
        mock_meta.return_value = {
            "name": "repo",
            "owner": "owner",
            "full_name": "owner/repo",
            "default_branch": "main",
            "visibility": "public",
        }
        mock_content.return_value = {
            "path": "src/app.py",
            "name": "app.py",
            "size": 50,
            "encoding": "utf-8",
            "content": "def run(): pass\n",
            "sha": "abc",
            "html_url": "https://github.com/owner/repo/blob/main/src/app.py",
        }

        payload = {
            "repo_url": "https://github.com/owner/repo",
            "paths": ["src/app.py"],
        }
        resp = self.client.post("/api/index/build", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["summary"]["files_processed"], 1)

    @patch("routes.index.get_repo_metadata")
    def test_private_repo_rejected(self, mock_meta):
        mock_meta.side_effect = GitHubAPIError(403, "Repository is private.")
        payload = {
            "repo_url": "https://github.com/owner/private",
            "paths": ["src/app.py"],
        }
        resp = self.client.post("/api/index/build", json=payload)
        self.assertEqual(resp.status_code, 403)

    def test_index_status_missing_repo_url(self):
        resp = self.client.get("/api/index/status")
        self.assertEqual(resp.status_code, 400)

    def test_index_status_not_indexed(self):
        resp = self.client.get("/api/index/status?repo_url=https://github.com/nonexistent/repo")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "not_indexed")

    def test_index_status_after_build(self):
        payload = {
            "repo_url": "https://github.com/owner/status-test-repo",
            "files": [{"path": "app.py", "content": "print('hello')"}],
        }
        build_resp = self.client.post("/api/index/build", json=payload)
        self.assertEqual(build_resp.status_code, 200)

        status_resp = self.client.get("/api/index/status?repo_url=https://github.com/owner/status-test-repo")
        self.assertEqual(status_resp.status_code, 200)
        data = status_resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "ready")
        self.assertIn("summary", data)
        self.assertEqual(data["summary"]["files_processed"], 1)

    def test_async_background_build_and_status_polling(self):
        import time
        payload = {
            "repo_url": "https://github.com/owner/async-test-repo",
            "files": [{"path": "main.py", "content": "def main(): pass"}],
            "sync": False,
        }
        self.app.config["TESTING"] = False
        try:
            build_resp = self.client.post("/api/index/build", json=payload)
            self.assertIn(build_resp.status_code, (200, 202))
            data = build_resp.get_json()
            self.assertTrue(data["success"])
            self.assertEqual(data["status"], "indexing")

            # Wait for background thread to complete
            for _ in range(20):
                time.sleep(0.05)
                status_resp = self.client.get("/api/index/status?repo_url=https://github.com/owner/async-test-repo")
                s_data = status_resp.get_json()
                if s_data.get("status") == "ready":
                    break

            self.assertEqual(s_data["status"], "ready")
            self.assertIsNotNone(s_data["summary"])
        finally:
            self.app.config["TESTING"] = True

    def test_async_background_build_failure_reporting(self):
        import time
        self.mock_model.encode.side_effect = RuntimeError("Simulated FastEmbed memory/runtime error")
        self.mock_model.embed.side_effect = RuntimeError("Simulated FastEmbed memory/runtime error")
        payload = {
            "repo_url": "https://github.com/owner/failure-test-repo",
            "files": [{"path": "fail.py", "content": "def fail(): pass"}],
            "sync": False,
        }
        self.app.config["TESTING"] = False
        try:
            build_resp = self.client.post("/api/index/build", json=payload)
            self.assertIn(build_resp.status_code, (200, 202))

            # Wait for background thread to encounter failure
            for _ in range(20):
                time.sleep(0.05)
                status_resp = self.client.get("/api/index/status?repo_url=https://github.com/owner/failure-test-repo")
                s_data = status_resp.get_json()
                if s_data.get("status") == "failed":
                    break

            self.assertEqual(s_data["status"], "failed")
            self.assertIn("Simulated FastEmbed", s_data.get("error", ""))
        finally:
            self.app.config["TESTING"] = True


if __name__ == "__main__":
    unittest.main()
