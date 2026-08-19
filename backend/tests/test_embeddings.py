"""
tests/test_embeddings.py — Unit tests for services/embeddings.py (FastEmbed ONNX backend).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from services.embeddings import (
    EMBEDDING_DIMENSION,
    embed_chunks,
    embed_query,
    embed_texts,
    set_embedding_model,
)


class EmbeddingsServiceTestCase(unittest.TestCase):
    def setUp(self):
        # Create a mock FastEmbed model returning generator of numpy vectors
        self.mock_model = MagicMock()
        def mock_embed(texts, batch_size=8, **kwargs):
            arr = np.ones((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            normalized = arr / norms
            for vec in normalized:
                yield vec
        def mock_encode(texts, batch_size=8, **kwargs):
            arr = np.ones((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / norms

        self.mock_model.embed.side_effect = mock_embed
        self.mock_model.encode.side_effect = mock_encode
        set_embedding_model(self.mock_model)

    def tearDown(self):
        set_embedding_model(None)

    def test_empty_texts_returns_empty_array(self):
        result = embed_texts([])
        self.assertEqual(result.shape, (0, EMBEDDING_DIMENSION))
        self.mock_model.embed.assert_not_called()

    def test_embed_texts_produces_correct_shape_and_dtype(self):
        texts = ["def main(): pass", "console.log('hello')"]
        embeddings = embed_texts(texts)
        self.assertEqual(embeddings.shape, (2, EMBEDDING_DIMENSION))
        self.assertEqual(embeddings.dtype, np.float32)
        # Verify batching and normalized length
        self.mock_model.embed.assert_called_once()
        _, kwargs = self.mock_model.embed.call_args
        self.assertEqual(kwargs.get("batch_size"), 8)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)

    def test_embed_query_valid(self):
        q_vec = embed_query("how to authenticate")
        self.assertEqual(q_vec.shape, (EMBEDDING_DIMENSION,))
        self.assertEqual(q_vec.dtype, np.float32)
        self.mock_model.embed.assert_called_once()
        _, kwargs = self.mock_model.embed.call_args
        self.assertEqual(kwargs.get("batch_size"), 8)
        self.assertAlmostEqual(float(np.linalg.norm(q_vec)), 1.0, places=5)

    def test_embed_query_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            embed_query("")
        with self.assertRaises(ValueError):
            embed_query("   ")

    def test_embed_chunks(self):
        chunks = [
            {
                "file_path": "src/app.py",
                "language": "Python",
                "content": "import flask",
            },
            {
                "file_path": "README.md",
                "language": "Markdown",
                "content": "# Docs",
            },
        ]
        embeddings = embed_chunks(chunks)
        self.assertEqual(embeddings.shape, (2, EMBEDDING_DIMENSION))

    def test_embed_chunks_empty(self):
        embeddings = embed_chunks([])
        self.assertEqual(embeddings.shape, (0, EMBEDDING_DIMENSION))

    def test_model_error_raises_runtime_error(self):
        self.mock_model.embed.side_effect = RuntimeError("Inference failed.")
        with self.assertRaises(RuntimeError) as ctx:
            embed_texts(["test"])
        self.assertIn("Error during batch text embedding", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
