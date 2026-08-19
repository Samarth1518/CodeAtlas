"""
tests/test_repos.py — Unit tests for the repository analysis endpoint and GitHub service.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

import sys
from pathlib import Path

# Add backend directory to sys.path so imports work properly
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import create_app
from services.github import GitHubAPIError


class ReposEndpointTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_analyze_missing_body(self):
        response = self.client.post("/api/repos/analyze", json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Missing 'repo_url'", data["error"])

    def test_analyze_invalid_url(self):
        response = self.client.post(
            "/api/repos/analyze", json={"repo_url": "https://gitlab.com/owner/repo"}
        )
        self.assertEqual(response.status_code, 422)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Invalid GitHub repository URL", data["error"])

    @patch("routes.repos.get_repo_tree")
    @patch("routes.repos.get_repo_metadata")
    def test_analyze_success_with_tree(self, mock_get_metadata, mock_get_tree):
        mock_get_metadata.return_value = {
            "name": "react",
            "owner": "facebook",
            "full_name": "facebook/react",
            "html_url": "https://github.com/facebook/react",
            "default_branch": "main",
            "visibility": "public",
            "primary_language": "JavaScript",
        }
        mock_get_tree.return_value = {
            "tree": [
                {"path": "package.json", "type": "file", "size": 1200},
                {"path": "src", "type": "directory", "size": None},
                {"path": "src/index.js", "type": "file", "size": 340},
            ],
            "total_files": 2,
            "total_dirs": 1,
            "truncated": False,
        }

        response = self.client.post(
            "/api/repos/analyze", json={"repo_url": "https://github.com/facebook/react"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["name"], "react")
        self.assertEqual(data["owner"], "facebook")
        self.assertEqual(data["full_name"], "facebook/react")
        self.assertEqual(data["html_url"], "https://github.com/facebook/react")
        self.assertEqual(data["default_branch"], "main")
        self.assertEqual(data["visibility"], "public")
        self.assertEqual(data["primary_language"], "JavaScript")
        self.assertEqual(len(data["tree"]), 3)
        self.assertEqual(data["tree_summary"]["total_files"], 2)
        self.assertEqual(data["tree_summary"]["total_dirs"], 1)
        self.assertFalse(data["tree_summary"]["truncated"])

    @patch("routes.repos.get_repo_metadata")
    def test_analyze_not_found(self, mock_get_metadata):
        mock_get_metadata.side_effect = GitHubAPIError(
            404, "Repository 'owner/nonexistent' not found on GitHub or is private/inaccessible."
        )

        response = self.client.post(
            "/api/repos/analyze",
            json={"repo_url": "https://github.com/owner/nonexistent"},
        )
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("not found", data["error"])

    @patch("routes.repos.get_repo_metadata")
    def test_analyze_forbidden_or_private(self, mock_get_metadata):
        mock_get_metadata.side_effect = GitHubAPIError(
            403, "Access denied for repository 'owner/private-repo'."
        )

        response = self.client.post(
            "/api/repos/analyze",
            json={"repo_url": "https://github.com/owner/private-repo"},
        )
        self.assertEqual(response.status_code, 403)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Access denied", data["error"])

    @patch("routes.repos.get_repo_metadata")
    def test_analyze_rate_limit(self, mock_get_metadata):
        mock_get_metadata.side_effect = GitHubAPIError(
            429, "GitHub API rate limit exceeded. Please try again later."
        )

        response = self.client.post(
            "/api/repos/analyze",
            json={"repo_url": "https://github.com/owner/repo"},
        )
        self.assertEqual(response.status_code, 429)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("rate limit exceeded", data["error"])

    @patch("routes.repos.get_repo_tree")
    @patch("routes.repos.get_repo_metadata")
    def test_analyze_tree_api_error(self, mock_get_metadata, mock_get_tree):
        mock_get_metadata.return_value = {
            "name": "react",
            "owner": "facebook",
            "full_name": "facebook/react",
            "html_url": "https://github.com/facebook/react",
            "default_branch": "main",
            "visibility": "public",
            "primary_language": "JavaScript",
        }
        mock_get_tree.side_effect = GitHubAPIError(502, "GitHub API is currently unavailable.")

        response = self.client.post(
            "/api/repos/analyze", json={"repo_url": "https://github.com/facebook/react"}
        )
        self.assertEqual(response.status_code, 502)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("unavailable", data["error"])


if __name__ == "__main__":
    unittest.main()
