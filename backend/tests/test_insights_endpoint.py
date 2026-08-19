"""
tests/test_insights_endpoint.py — Integration tests for POST /api/insights/analyze.

No real GitHub API calls are made.
"""

import json
import unittest

from app import create_app

SAMPLE_METADATA = {
    "name": "my-app",
    "owner": "owner",
    "full_name": "owner/my-app",
    "primary_language": "Python",
    "default_branch": "main",
    "visibility": "public",
}

SAMPLE_TREE = [
    {"path": "README.md", "type": "file", "size": 1200},
    {"path": "requirements.txt", "type": "file", "size": 300},
    {"path": "app.py", "type": "file", "size": 2400},
    {"path": "Dockerfile", "type": "file", "size": 800},
    {"path": ".github/workflows/ci.yml", "type": "file", "size": 600},
    {"path": "backend", "type": "directory", "size": None},
    {"path": "backend/routes/api.py", "type": "file", "size": 1500},
    {"path": "tests", "type": "directory", "size": None},
    {"path": "tests/test_app.py", "type": "file", "size": 900},
]

SAMPLE_SOURCE_FILES = [
    {
        "path": "requirements.txt",
        "content": "flask==3.0.3\nnumpy>=1.26.0\nrequests>=2.28\n",
        "language": "text",
    },
]


class TestInsightsEndpoint(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app.config["TESTING"] = True

    def _post(self, body):
        return self.client.post(
            "/api/insights/analyze",
            data=json.dumps(body),
            content_type="application/json",
        )

    # ── Validation tests ──────────────────────────────────────────────────────

    def test_missing_body_returns_400(self):
        resp = self.client.post("/api/insights/analyze", content_type="application/json")
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["success"])

    def test_missing_repo_url_returns_400(self):
        resp = self._post({"metadata": SAMPLE_METADATA, "tree": SAMPLE_TREE})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("repo_url", data["error"])

    def test_invalid_repo_url_returns_422(self):
        resp = self._post({
            "repo_url": "not-a-github-url",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
        })
        self.assertEqual(resp.status_code, 422)
        self.assertFalse(resp.get_json()["success"])

    def test_non_github_url_returns_422(self):
        resp = self._post({
            "repo_url": "https://gitlab.com/owner/repo",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
        })
        self.assertEqual(resp.status_code, 422)

    def test_missing_metadata_returns_400(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/repo",
            "tree": SAMPLE_TREE,
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("metadata", resp.get_json()["error"])

    def test_invalid_metadata_type_returns_400(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/repo",
            "metadata": "not-a-dict",
            "tree": SAMPLE_TREE,
        })
        self.assertEqual(resp.status_code, 400)

    def test_missing_tree_returns_400(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/repo",
            "metadata": SAMPLE_METADATA,
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("tree", resp.get_json()["error"])

    def test_invalid_tree_type_returns_400(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/repo",
            "metadata": SAMPLE_METADATA,
            "tree": "not-a-list",
        })
        self.assertEqual(resp.status_code, 400)

    def test_invalid_source_files_type_returns_400(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/repo",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
            "source_files": "not-a-list",
        })
        self.assertEqual(resp.status_code, 400)

    # ── Success tests ─────────────────────────────────────────────────────────

    def test_successful_response_structure(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/my-app",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
            "source_files": SAMPLE_SOURCE_FILES,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("insights", data)

    def test_insights_contains_all_sections(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/my-app",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
        })
        insights = resp.get_json()["insights"]
        for key in ["statistics", "technologies", "dependencies",
                    "important_files", "directory_structure", "ci_cd",
                    "documentation", "architecture_notes"]:
            self.assertIn(key, insights, f"Missing key: {key}")

    def test_statistics_have_counts(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/my-app",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
        })
        stats = resp.get_json()["insights"]["statistics"]
        self.assertGreater(stats["total_files"], 0)

    def test_empty_tree_returns_200(self):
        """Empty repository should not cause an error."""
        resp = self._post({
            "repo_url": "https://github.com/owner/empty-repo",
            "metadata": {**SAMPLE_METADATA, "name": "empty-repo"},
            "tree": [],
        })
        self.assertEqual(resp.status_code, 200)
        insights = resp.get_json()["insights"]
        self.assertEqual(insights["statistics"]["total_files"], 0)

    def test_source_files_without_content_does_not_crash(self):
        """Manifests in tree but not in source_files handled gracefully."""
        resp = self._post({
            "repo_url": "https://github.com/owner/my-app",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
            "source_files": [],  # No content
        })
        self.assertEqual(resp.status_code, 200)

    def test_ci_cd_detects_github_actions(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/my-app",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
        })
        ci_cd = resp.get_json()["insights"]["ci_cd"]
        self.assertIn("GitHub Actions", ci_cd)

    def test_documentation_detects_readme(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/my-app",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
        })
        docs = resp.get_json()["insights"]["documentation"]
        self.assertTrue(docs["has_readme"])

    def test_flask_detected_from_requirements(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/my-app",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
            "source_files": SAMPLE_SOURCE_FILES,
        })
        frameworks = resp.get_json()["insights"]["technologies"]["detected_frameworks"]
        self.assertIn("Flask", frameworks)

    def test_important_files_includes_entry_point(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/my-app",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
        })
        important = resp.get_json()["insights"]["important_files"]
        roles = [f["role"] for f in important]
        self.assertIn("Entry point", roles)

    def test_url_with_trailing_slash_normalised(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/my-app/",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
        })
        self.assertEqual(resp.status_code, 200)

    def test_architecture_notes_list(self):
        resp = self._post({
            "repo_url": "https://github.com/owner/my-app",
            "metadata": SAMPLE_METADATA,
            "tree": SAMPLE_TREE,
        })
        notes = resp.get_json()["insights"]["architecture_notes"]
        self.assertIsInstance(notes, list)

    def test_no_secret_in_response(self):
        """GitHub token or LLM API keys must not appear in the response."""
        import os
        fake_token = "ghp_test_never_leak_in_insights"
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
            os.environ, {"GITHUB_TOKEN": fake_token}
        ):
            resp = self._post({
                "repo_url": "https://github.com/owner/my-app",
                "metadata": SAMPLE_METADATA,
                "tree": SAMPLE_TREE,
            })
        self.assertNotIn(fake_token, resp.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
