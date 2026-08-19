"""
tests/test_github_content_service.py
Unit tests for services/github.py :: get_file_content()
"""

import base64
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import urllib.error

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from services.github import GitHubAPIError, get_file_content


def _make_response(payload: dict) -> MagicMock:
    """Helper: build a mock urlopen context-manager response."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    return mock_resp


def _b64(text: str) -> str:
    """Helper: encode text to a Base64 string as GitHub would return it."""
    raw = base64.b64encode(text.encode("utf-8")).decode("ascii")
    # GitHub inserts newlines every 60 chars
    return "\n".join(raw[i : i + 60] for i in range(0, len(raw), 60)) + "\n"


class GetFileContentTestCase(unittest.TestCase):
    # ── Success case ──────────────────────────────────────────────────────────

    @patch("urllib.request.urlopen")
    def test_success_decodes_base64_content(self, mock_urlopen):
        source = "import os\nprint('hello')\n"
        payload = {
            "type": "file",
            "name": "app.py",
            "path": "src/app.py",
            "size": len(source),
            "encoding": "base64",
            "content": _b64(source),
            "sha": "deadbeef",
            "html_url": "https://github.com/owner/repo/blob/main/src/app.py",
        }
        mock_urlopen.return_value.__enter__.return_value = _make_response(payload)

        result = get_file_content("owner", "repo", "src/app.py", "main")

        self.assertEqual(result["path"], "src/app.py")
        self.assertEqual(result["name"], "app.py")
        self.assertEqual(result["content"], source)
        self.assertEqual(result["encoding"], "utf-8")
        self.assertEqual(result["sha"], "deadbeef")

    @patch("urllib.request.urlopen")
    def test_content_with_embedded_newlines_in_b64_decoded_correctly(self, mock_urlopen):
        """Ensure GitHub's newline-padded Base64 is stripped correctly."""
        source = "x" * 1000  # long enough to trigger multi-line B64
        payload = {
            "type": "file",
            "name": "data.txt",
            "path": "data.txt",
            "size": len(source),
            "encoding": "base64",
            "content": _b64(source),
            "sha": "aaa",
            "html_url": "",
        }
        mock_urlopen.return_value.__enter__.return_value = _make_response(payload)

        result = get_file_content("owner", "repo", "data.txt", "main")
        self.assertEqual(result["content"], source)

    # ── Directory path ────────────────────────────────────────────────────────

    @patch("urllib.request.urlopen")
    def test_directory_path_raises_error(self, mock_urlopen):
        """GitHub returns a list when the path is a directory."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        # Directory listing is a JSON array
        mock_resp.read.return_value = json.dumps(
            [{"type": "file", "name": "app.py", "path": "src/app.py"}]
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        with self.assertRaises(GitHubAPIError) as ctx:
            get_file_content("owner", "repo", "src", "main")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("directory", ctx.exception.message)

    # ── Non-file type (submodule, symlink) ────────────────────────────────────

    @patch("urllib.request.urlopen")
    def test_submodule_raises_error(self, mock_urlopen):
        payload = {"type": "submodule", "name": "dep", "path": "dep", "sha": "abc"}
        mock_urlopen.return_value.__enter__.return_value = _make_response(payload)

        with self.assertRaises(GitHubAPIError) as ctx:
            get_file_content("owner", "repo", "dep", "main")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("not a regular file", ctx.exception.message)

    # ── Unexpected encoding ───────────────────────────────────────────────────

    @patch("urllib.request.urlopen")
    def test_non_base64_encoding_raises_value_error(self, mock_urlopen):
        payload = {
            "type": "file",
            "name": "app.py",
            "path": "src/app.py",
            "size": 10,
            "encoding": "none",   # edge-case GitHub returns for empty files
            "content": "",
            "sha": "abc",
            "html_url": "",
        }
        mock_urlopen.return_value.__enter__.return_value = _make_response(payload)

        with self.assertRaises(ValueError):
            get_file_content("owner", "repo", "src/app.py", "main")

    # ── HTTP errors ───────────────────────────────────────────────────────────

    @patch("urllib.request.urlopen")
    def test_404_raises_github_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/repos/owner/repo/contents/missing.py",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b"{}"),
        )
        with self.assertRaises(GitHubAPIError) as ctx:
            get_file_content("owner", "repo", "missing.py", "main")
        self.assertEqual(ctx.exception.status_code, 404)

    @patch("urllib.request.urlopen")
    def test_rate_limit_raises_429(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/repos/owner/repo/contents/app.py",
            code=429,
            msg="Too Many Requests",
            hdrs={"x-ratelimit-remaining": "0"},
            fp=io.BytesIO(b"{}"),
        )
        with self.assertRaises(GitHubAPIError) as ctx:
            get_file_content("owner", "repo", "app.py", "main")
        self.assertEqual(ctx.exception.status_code, 429)

    @patch("urllib.request.urlopen")
    def test_github_500_mapped_to_502(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/repos/owner/repo/contents/app.py",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b"{}"),
        )
        with self.assertRaises(GitHubAPIError) as ctx:
            get_file_content("owner", "repo", "app.py", "main")
        self.assertEqual(ctx.exception.status_code, 502)


if __name__ == "__main__":
    unittest.main()
