"""
tests/test_contents_endpoint.py
Unit tests for POST /api/contents/fetch endpoint.
"""

import base64
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import create_app
from services.github import GitHubAPIError

# ── Helper fixture data ────────────────────────────────────────────────────────

_MOCK_METADATA = {
    "name": "my-repo",
    "owner": "owner",
    "full_name": "owner/my-repo",
    "html_url": "https://github.com/owner/my-repo",
    "default_branch": "main",
    "visibility": "public",
    "primary_language": "Python",
}

_MOCK_FILE = {
    "path": "src/app.py",
    "name": "app.py",
    "size": 42,
    "encoding": "utf-8",
    "content": "print('hello')\n",
    "sha": "abc123",
    "html_url": "https://github.com/owner/my-repo/blob/main/src/app.py",
}


class ContentsEndpointTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    # ── Input validation ──────────────────────────────────────────────────────

    def test_missing_body_returns_400(self):
        resp = self.client.post("/api/contents/fetch", data="not-json",
                                content_type="text/plain")
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["success"])

    def test_missing_repo_url_returns_400(self):
        resp = self.client.post("/api/contents/fetch",
                                json={"paths": ["app.py"]})
        self.assertEqual(resp.status_code, 400)

    def test_invalid_github_url_returns_422(self):
        resp = self.client.post("/api/contents/fetch",
                                json={"repo_url": "https://gitlab.com/o/r",
                                      "paths": ["app.py"]})
        self.assertEqual(resp.status_code, 422)

    @patch("routes.contents.get_repo_metadata")
    def test_missing_paths_returns_400(self, mock_meta):
        mock_meta.return_value = _MOCK_METADATA
        resp = self.client.post("/api/contents/fetch",
                                json={"repo_url": "https://github.com/owner/repo"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("paths", resp.get_json()["error"])

    def test_empty_paths_returns_400(self):
        resp = self.client.post("/api/contents/fetch",
                                json={"repo_url": "https://github.com/owner/repo",
                                      "paths": []})
        self.assertEqual(resp.status_code, 400)

    # ── Private repo rejection ────────────────────────────────────────────────

    @patch("routes.contents.get_repo_metadata")
    def test_private_repo_rejected(self, mock_meta):
        mock_meta.side_effect = GitHubAPIError(
            403,
            "Repository 'owner/private' is private. CodeAtlas currently only supports public repositories."
        )
        resp = self.client.post("/api/contents/fetch",
                                json={"repo_url": "https://github.com/owner/private",
                                      "paths": ["README.md"]})
        self.assertEqual(resp.status_code, 403)
        self.assertIn("private", resp.get_json()["error"])

    # ── Successful fetch ──────────────────────────────────────────────────────

    @patch("routes.contents.get_file_content")
    @patch("routes.contents.get_repo_metadata")
    def test_successful_fetch_returns_files(self, mock_meta, mock_content):
        mock_meta.return_value = _MOCK_METADATA
        mock_content.return_value = _MOCK_FILE

        resp = self.client.post("/api/contents/fetch",
                                json={"repo_url": "https://github.com/owner/my-repo",
                                      "paths": ["src/app.py"]})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["files"]), 1)
        self.assertEqual(data["files"][0]["path"], "src/app.py")
        self.assertEqual(data["files"][0]["language"], "Python")
        self.assertEqual(data["files"][0]["content"], "print('hello')\n")
        self.assertEqual(data["summary"]["fetched"], 1)
        self.assertEqual(data["summary"]["skipped"], 0)

    # ── Filtering / skipping ──────────────────────────────────────────────────

    @patch("routes.contents.get_file_content")
    @patch("routes.contents.get_repo_metadata")
    def test_binary_file_skipped(self, mock_meta, mock_content):
        mock_meta.return_value = _MOCK_METADATA

        resp = self.client.post("/api/contents/fetch",
                                json={"repo_url": "https://github.com/owner/my-repo",
                                      "paths": ["assets/logo.png"]})
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["files"]), 0)
        self.assertEqual(len(data["skipped"]), 1)
        self.assertIn("binary/media", data["skipped"][0]["reason"])
        mock_content.assert_not_called()

    @patch("routes.contents.get_file_content")
    @patch("routes.contents.get_repo_metadata")
    def test_sensitive_env_file_skipped(self, mock_meta, mock_content):
        mock_meta.return_value = _MOCK_METADATA

        resp = self.client.post("/api/contents/fetch",
                                json={"repo_url": "https://github.com/owner/my-repo",
                                      "paths": [".env"]})
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["files"]), 0)
        self.assertEqual(len(data["skipped"]), 1)
        self.assertIn("sensitive", data["skipped"][0]["reason"])
        mock_content.assert_not_called()

    @patch("routes.contents.get_file_content")
    @patch("routes.contents.get_repo_metadata")
    def test_node_modules_skipped(self, mock_meta, mock_content):
        mock_meta.return_value = _MOCK_METADATA

        resp = self.client.post("/api/contents/fetch",
                                json={"repo_url": "https://github.com/owner/my-repo",
                                      "paths": ["node_modules/lodash/index.js"]})
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["skipped"]), 1)
        self.assertIn("node_modules", data["skipped"][0]["reason"])
        mock_content.assert_not_called()

    @patch("routes.contents.get_file_content")
    @patch("routes.contents.get_repo_metadata")
    def test_mixed_paths_fetched_and_skipped(self, mock_meta, mock_content):
        mock_meta.return_value = _MOCK_METADATA
        mock_content.return_value = _MOCK_FILE

        resp = self.client.post(
            "/api/contents/fetch",
            json={
                "repo_url": "https://github.com/owner/my-repo",
                "paths": ["src/app.py", "logo.png", ".env"],
            },
        )
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["summary"]["fetched"], 1)
        self.assertEqual(data["summary"]["skipped"], 2)

    # ── GitHub API errors during fetch ────────────────────────────────────────

    @patch("routes.contents.get_file_content")
    @patch("routes.contents.get_repo_metadata")
    def test_github_error_during_fetch_counted_as_error(self, mock_meta, mock_content):
        mock_meta.return_value = _MOCK_METADATA
        mock_content.side_effect = GitHubAPIError(404, "File not found.")

        resp = self.client.post("/api/contents/fetch",
                                json={"repo_url": "https://github.com/owner/my-repo",
                                      "paths": ["src/app.py"]})
        data = resp.get_json()
        self.assertTrue(data["success"])   # endpoint itself succeeded
        self.assertEqual(data["summary"]["fetched"], 0)
        self.assertEqual(data["summary"]["errors"], 1)

    # ── Too many paths ────────────────────────────────────────────────────────

    @patch("routes.contents.get_repo_metadata")
    def test_too_many_paths_returns_400(self, mock_meta):
        mock_meta.return_value = _MOCK_METADATA
        paths = [f"file{i}.py" for i in range(51)]
        resp = self.client.post("/api/contents/fetch",
                                json={"repo_url": "https://github.com/owner/my-repo",
                                      "paths": paths})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Too many paths", resp.get_json()["error"])

    # ── ref overrides default branch ──────────────────────────────────────────

    @patch("routes.contents.get_file_content")
    @patch("routes.contents.get_repo_metadata")
    def test_custom_ref_passed_to_get_file_content(self, mock_meta, mock_content):
        mock_meta.return_value = _MOCK_METADATA
        mock_content.return_value = _MOCK_FILE

        self.client.post("/api/contents/fetch",
                         json={"repo_url": "https://github.com/owner/my-repo",
                               "paths": ["src/app.py"],
                               "ref": "develop"})
        # Verify `ref="develop"` was passed through
        mock_content.assert_called_once_with("owner", "my-repo", "src/app.py", "develop")


if __name__ == "__main__":
    unittest.main()
