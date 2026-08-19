"""
tests/test_live_integration.py — Integration test hitting the real GitHub API with the token from .env
"""

import os
import unittest
from pathlib import Path

import sys
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import create_app


class LiveIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_live_github_api_metadata_and_tree(self):
        response = self.client.post(
            "/api/repos/analyze",
            json={"repo_url": "https://github.com/octocat/Hello-World"},
        )
        if response.status_code == 503:
            self.skipTest("GitHub API unreachable or timed out from local environment.")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["name"], "Hello-World")
        self.assertEqual(data["owner"].lower(), "octocat")
        self.assertEqual(data["full_name"], "octocat/Hello-World")
        self.assertEqual(data["html_url"], "https://github.com/octocat/Hello-World")
        self.assertIn("default_branch", data)
        self.assertIn("visibility", data)
        self.assertIn("primary_language", data)
        self.assertIn("tree", data)
        self.assertIsInstance(data["tree"], list)
        self.assertGreater(len(data["tree"]), 0)
        self.assertIn("tree_summary", data)


if __name__ == "__main__":
    unittest.main()
