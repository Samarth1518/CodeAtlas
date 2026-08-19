"""
tests/test_github_service.py — Unit tests for GitHub API service functions and error mapping.
"""

import io
import json
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from services.github import GitHubAPIError, get_repo_metadata, get_repo_tree


class GitHubServiceTestCase(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_get_repo_metadata_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        payload = {
            "name": "CodeAtlas",
            "owner": {"login": "testowner"},
            "full_name": "testowner/CodeAtlas",
            "html_url": "https://github.com/testowner/CodeAtlas",
            "default_branch": "main",
            "visibility": "public",
            "language": "Python",
        }
        mock_response.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        meta = get_repo_metadata("testowner", "CodeAtlas")
        self.assertEqual(meta["name"], "CodeAtlas")
        self.assertEqual(meta["owner"], "testowner")
        self.assertEqual(meta["full_name"], "testowner/CodeAtlas")
        self.assertEqual(meta["html_url"], "https://github.com/testowner/CodeAtlas")
        self.assertEqual(meta["default_branch"], "main")
        self.assertEqual(meta["visibility"], "public")
        self.assertEqual(meta["primary_language"], "Python")

    @patch("urllib.request.urlopen")
    def test_get_repo_metadata_private_repo_rejected(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        payload = {
            "name": "secret-repo",
            "owner": {"login": "testowner"},
            "full_name": "testowner/secret-repo",
            "html_url": "https://github.com/testowner/secret-repo",
            "default_branch": "main",
            "visibility": "private",
            "private": True,
            "language": "Python",
        }
        mock_response.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with self.assertRaises(GitHubAPIError) as ctx:
            get_repo_metadata("testowner", "secret-repo")
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("only supports public", ctx.exception.message)

    @patch("urllib.request.urlopen")
    def test_get_repo_metadata_404(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/repos/test/nonexistent",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b"{}"),
        )
        with self.assertRaises(GitHubAPIError) as ctx:
            get_repo_metadata("test", "nonexistent")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("not found", ctx.exception.message)

    @patch("urllib.request.urlopen")
    def test_get_repo_metadata_403_rate_limit(self, mock_urlopen):
        headers = {"x-ratelimit-remaining": "0"}
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/repos/test/repo",
            code=403,
            msg="Forbidden",
            hdrs=headers,
            fp=io.BytesIO(b"{}"),
        )
        with self.assertRaises(GitHubAPIError) as ctx:
            get_repo_metadata("test", "repo")
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("rate limit exceeded", ctx.exception.message)

    @patch("urllib.request.urlopen")
    def test_get_repo_metadata_500(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/repos/test/repo",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b"{}"),
        )
        with self.assertRaises(GitHubAPIError) as ctx:
            get_repo_metadata("test", "repo")
        self.assertEqual(ctx.exception.status_code, 502)

    @patch("urllib.request.urlopen")
    def test_get_repo_tree_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        payload = {
            "sha": "abc123sha",
            "tree": [
                {"path": "README.md", "type": "blob", "size": 512},
                {"path": "src", "type": "tree"},
                {"path": "src/app.py", "type": "blob", "size": 1024},
            ],
            "truncated": False,
        }
        mock_response.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        tree_res = get_repo_tree("testowner", "CodeAtlas", "main")
        self.assertEqual(len(tree_res["tree"]), 3)
        self.assertEqual(tree_res["total_files"], 2)
        self.assertEqual(tree_res["total_dirs"], 1)
        self.assertFalse(tree_res["truncated"])

        # Check item structure
        readme = tree_res["tree"][0]
        self.assertEqual(readme["path"], "README.md")
        self.assertEqual(readme["type"], "file")
        self.assertEqual(readme["size"], 512)

        src_dir = tree_res["tree"][1]
        self.assertEqual(src_dir["path"], "src")
        self.assertEqual(src_dir["type"], "directory")
        self.assertIsNone(src_dir["size"])

    @patch("urllib.request.urlopen")
    def test_get_repo_tree_empty_repository(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/repos/test/empty-repo/git/trees/main",
            code=409,
            msg="Conflict",
            hdrs={},
            fp=io.BytesIO(b'{"message": "Git Repository is empty."}'),
        )

        tree_res = get_repo_tree("test", "empty-repo", "main")
        self.assertEqual(tree_res["tree"], [])
        self.assertEqual(tree_res["total_files"], 0)
        self.assertEqual(tree_res["total_dirs"], 0)
        self.assertFalse(tree_res["truncated"])


if __name__ == "__main__":
    unittest.main()
