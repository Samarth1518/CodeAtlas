"""
tests/test_search_endpoint.py — Unit tests for POST /api/search endpoint.
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
from services.vector_store import default_vector_store


class SearchEndpointTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

        # Mock embedding model to return normalized vectors
        self.mock_model = MagicMock()
        def mock_encode(texts, **kwargs):
            arr = np.ones((len(texts), 384), dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / norms
        self.mock_model.encode.side_effect = mock_encode
        set_embedding_model(self.mock_model)

        # Seed sample repository index
        self.repo_url = "https://github.com/owner/demo-repo"
        chunks = [
            {
                "chunk_id": "chunk_auth",
                "file_path": "src/auth.py",
                "language": "Python",
                "start_line": 10,
                "end_line": 35,
                "content": "def authenticate_user(token):\n    return token.is_valid()",
            },
            {
                "chunk_id": "chunk_routes",
                "file_path": "src/routes.py",
                "language": "Python",
                "start_line": 1,
                "end_line": 20,
                "content": "@app.route('/home')\ndef home(): return 'ok'",
            },
        ]
        # Vectors of size 384
        emb_matrix = np.zeros((2, 384), dtype=np.float32)
        emb_matrix[0, 0] = 1.0  # auth vector points along dim 0
        emb_matrix[1, 1] = 1.0  # routes vector points along dim 1
        default_vector_store.build_index(self.repo_url, chunks, emb_matrix)

    def tearDown(self):
        set_embedding_model(None)

    def test_missing_body_returns_400(self):
        resp = self.client.post("/api/search", data="invalid", content_type="text/plain")
        self.assertEqual(resp.status_code, 400)

    def test_missing_repo_url_returns_400(self):
        resp = self.client.post("/api/search", json={"query": "find auth"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("repo_url", resp.get_json()["error"])

    def test_invalid_repo_url_returns_422(self):
        resp = self.client.post("/api/search", json={"repo_url": "invalid://url", "query": "find auth"})
        self.assertEqual(resp.status_code, 422)

    def test_empty_query_returns_400(self):
        resp = self.client.post("/api/search", json={"repo_url": self.repo_url, "query": "   "})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Query string cannot be empty", resp.get_json()["error"])

    def test_query_too_long_returns_400(self):
        long_query = "a" * 501
        resp = self.client.post("/api/search", json={"repo_url": self.repo_url, "query": long_query})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Query too long", resp.get_json()["error"])

    def test_successful_semantic_search(self):
        # Override mock encode for this test to query dim 0 (auth)
        def auth_query_encode(texts, **kwargs):
            arr = np.zeros((len(texts), 384), dtype=np.float32)
            arr[:, 0] = 1.0
            return arr
        self.mock_model.encode.side_effect = auth_query_encode

        resp = self.client.post(
            "/api/search",
            json={
                "repo_url": self.repo_url,
                "query": "Where is user authentication handled?",
                "top_k": 2,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["query"], "Where is user authentication handled?")
        self.assertEqual(len(data["results"]), 2)
        # First result should be auth chunk
        first = data["results"][0]
        self.assertEqual(first["chunk_id"], "chunk_auth")
        self.assertEqual(first["file_path"], "src/auth.py")
        self.assertAlmostEqual(first["score"], 1.0, places=3)

    def test_search_unindexed_repository_returns_empty_results(self):
        resp = self.client.post(
            "/api/search",
            json={
                "repo_url": "https://github.com/owner/unknown-repo",
                "query": "something",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["results"], [])

    def test_top_k_clamping(self):
        resp = self.client.post(
            "/api/search",
            json={
                "repo_url": self.repo_url,
                "query": "test query",
                "top_k": 999,  # over max of 20
            },
        )
        self.assertEqual(resp.status_code, 200)
        # Result count bounded by available chunks (2)
        self.assertLessEqual(len(resp.get_json()["results"]), 20)


if __name__ == "__main__":
    unittest.main()
