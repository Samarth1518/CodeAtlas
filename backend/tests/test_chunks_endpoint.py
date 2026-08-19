"""
tests/test_chunks_endpoint.py — Unit tests for POST /api/chunks/generate endpoint.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import create_app
from services.github import GitHubAPIError


class ChunksEndpointTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    # ── Request Validation ────────────────────────────────────────────────────

    def test_missing_body_returns_400(self):
        resp = self.client.post("/api/chunks/generate", data="invalid", content_type="text/plain")
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["success"])

    def test_missing_repo_url_returns_400(self):
        resp = self.client.post("/api/chunks/generate", json={"files": []})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("repo_url", resp.get_json()["error"])

    def test_invalid_repo_url_returns_422(self):
        resp = self.client.post(
            "/api/chunks/generate",
            json={"repo_url": "https://gitlab.com/owner/repo", "files": []},
        )
        self.assertEqual(resp.status_code, 422)

    def test_neither_files_nor_paths_returns_400(self):
        resp = self.client.post(
            "/api/chunks/generate",
            json={"repo_url": "https://github.com/owner/repo"},
        )
        self.assertEqual(resp.status_code, 400)

    # ── Direct Files Mode ─────────────────────────────────────────────────────

    def test_direct_files_chunking_success(self):
        payload = {
            "repo_url": "https://github.com/facebook/react",
            "files": [
                {
                    "path": "src/index.js",
                    "content": "export function render() { return true; }\n",
                    "language": "JavaScript",
                },
                {
                    "path": "README.md",
                    "content": "# React\n\nA declarative, efficient UI library.\n",
                    "language": "Markdown",
                },
            ],
        }
        resp = self.client.post("/api/chunks/generate", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["repo_url"], "https://github.com/facebook/react")
        self.assertGreaterEqual(data["total_chunks"], 2)
        self.assertEqual(data["total_files_processed"], 2)
        self.assertIn("chunks", data)
        self.assertIn("summary", data)

    def test_sensitive_files_in_direct_mode_filtered_out(self):
        payload = {
            "repo_url": "https://github.com/owner/repo",
            "files": [
                {
                    "path": ".env",
                    "content": "SECRET_KEY=123456\n",
                },
                {
                    "path": "src/app.py",
                    "content": "print('hello')\n",
                    "language": "Python",
                },
            ],
        }
        resp = self.client.post("/api/chunks/generate", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        # .env should be filtered out
        file_paths = {c["file_path"] for c in data["chunks"]}
        self.assertNotIn(".env", file_paths)
        self.assertIn("src/app.py", file_paths)

    # ── On-Demand Path Fetch Mode ─────────────────────────────────────────────

    @patch("routes.chunks.get_file_content")
    @patch("routes.chunks.get_repo_metadata")
    def test_on_demand_path_fetch_mode(self, mock_meta, mock_content):
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
            "size": 35,
            "encoding": "utf-8",
            "content": "def main():\n    print('CodeAtlas')\n",
            "sha": "123",
            "html_url": "https://github.com/owner/repo/blob/main/src/app.py",
        }

        payload = {
            "repo_url": "https://github.com/owner/repo",
            "paths": ["src/app.py"],
        }
        resp = self.client.post("/api/chunks/generate", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["total_files_processed"], 1)
        self.assertGreaterEqual(data["total_chunks"], 1)

    @patch("routes.chunks.get_repo_metadata")
    def test_private_repo_rejected_in_path_mode(self, mock_meta):
        mock_meta.side_effect = GitHubAPIError(403, "Repository is private.")
        payload = {
            "repo_url": "https://github.com/owner/private",
            "paths": ["src/app.py"],
        }
        resp = self.client.post("/api/chunks/generate", json=payload)
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
