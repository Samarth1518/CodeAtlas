"""
tests/test_vector_store.py — Unit tests for services/vector_store.py
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from services.vector_store import LocalVectorStore


class VectorStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = LocalVectorStore(storage_dir=Path(self.temp_dir))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_and_search_index(self):
        repo_url = "https://github.com/testowner/testrepo"
        chunks = [
            {
                "chunk_id": "chunk_1",
                "file_path": "src/auth.py",
                "language": "Python",
                "start_line": 1,
                "end_line": 20,
                "content": "def login(): pass",
                "line_count": 20,
                "char_count": 17,
            },
            {
                "chunk_id": "chunk_2",
                "file_path": "src/db.py",
                "language": "Python",
                "start_line": 1,
                "end_line": 15,
                "content": "def connect_db(): pass",
                "line_count": 15,
                "char_count": 22,
            },
        ]
        # Vectors: orthogonal 2D-like in 4D space
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],  # auth chunk
            [0.0, 1.0, 0.0, 0.0],  # db chunk
        ], dtype=np.float32)

        summary = self.store.build_index(repo_url, chunks, embeddings)
        self.assertEqual(summary["files_processed"], 2)
        self.assertEqual(summary["chunks_indexed"], 2)

        # Query vector close to auth
        query_vec = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = self.store.search(repo_url, query_vec, top_k=2)

        self.assertEqual(len(results), 2)
        # Auth chunk should be top result with highest score
        self.assertEqual(results[0]["chunk_id"], "chunk_1")
        self.assertEqual(results[0]["file_path"], "src/auth.py")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_persistence_reloads_from_disk(self):
        repo_url = "https://github.com/testowner/testrepo"
        chunks = [
            {
                "chunk_id": "c1",
                "file_path": "main.py",
                "language": "Python",
                "start_line": 1,
                "end_line": 10,
                "content": "print('hello')",
            }
        ]
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        self.store.build_index(repo_url, chunks, embeddings)

        # Create a fresh store pointing to same directory (simulates server restart)
        new_store = LocalVectorStore(storage_dir=Path(self.temp_dir))
        results = new_store.search(repo_url, np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=3)

    def test_repository_isolation(self):
        repo_a = "https://github.com/owner/repo-a"
        repo_b = "https://github.com/owner/repo-b"

        chunks_a = [{"chunk_id": "a1", "file_path": "a.py", "content": "a"}]
        chunks_b = [{"chunk_id": "b1", "file_path": "b.py", "content": "b"}]

        self.store.build_index(repo_a, chunks_a, np.array([[1.0, 0.0]], dtype=np.float32))
        self.store.build_index(repo_b, chunks_b, np.array([[0.0, 1.0]], dtype=np.float32))

        # Search repo_a
        res_a = self.store.search(repo_a, np.array([1.0, 0.0], dtype=np.float32), top_k=5)
        self.assertEqual(len(res_a), 1)
        self.assertEqual(res_a[0]["chunk_id"], "a1")

        # Search repo_c (unindexed)
        res_c = self.store.search("https://github.com/owner/unindexed", np.array([1.0, 0.0], dtype=np.float32))
        self.assertEqual(res_c, [])

    def test_empty_index_search_returns_empty_list(self):
        repo_url = "https://github.com/owner/empty-repo"
        self.store.build_index(repo_url, [], np.empty((0, 4), dtype=np.float32))
        results = self.store.search(repo_url, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
