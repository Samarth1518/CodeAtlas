"""
tests/test_rag.py — Unit tests for services/rag.py.

All LLM calls and embedding calls are mocked. No real API calls made.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from services.rag import (
    answer_question,
    _validate_repo_url,
    _deduplicate_chunks,
    _build_sources,
    _extract_repo_name,
)
from services.llm import set_llm_provider
from services.embeddings import set_embedding_model


def _make_mock_model(dim=384):
    """Returns a mock SentenceTransformer that returns deterministic unit vectors."""
    mock_model = MagicMock()
    def _encode(texts, **kwargs):
        n = len(texts)
        vecs = np.ones((n, dim), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / norms).astype(np.float32)
    mock_model.encode.side_effect = _encode
    return mock_model


def _make_mock_llm(answer="Mocked LLM answer"):
    mock_provider = MagicMock()
    mock_provider.generate.return_value = answer
    return mock_provider


SAMPLE_CHUNKS = [
    {
        "chunk_id": "chunk-001",
        "file_path": "backend/auth.py",
        "language": "Python",
        "start_line": 1,
        "end_line": 30,
        "content": "def login(user, password): pass",
        "score": 0.92,
    },
    {
        "chunk_id": "chunk-002",
        "file_path": "backend/routes/api.py",
        "language": "Python",
        "start_line": 10,
        "end_line": 50,
        "content": "@app.route('/api/login', methods=['POST'])",
        "score": 0.85,
    },
]


class TestValidateRepoUrl(unittest.TestCase):

    def test_valid_url_normalised(self):
        result = _validate_repo_url("https://github.com/owner/repo")
        self.assertEqual(result, "https://github.com/owner/repo")

    def test_valid_url_with_trailing_slash(self):
        result = _validate_repo_url("https://github.com/owner/repo/")
        self.assertEqual(result, "https://github.com/owner/repo")

    def test_valid_url_with_git_suffix(self):
        result = _validate_repo_url("https://github.com/owner/repo.git")
        self.assertEqual(result, "https://github.com/owner/repo")

    def test_invalid_url_raises(self):
        with self.assertRaises(ValueError):
            _validate_repo_url("not-a-url")

    def test_non_github_url_raises(self):
        with self.assertRaises(ValueError):
            _validate_repo_url("https://gitlab.com/owner/repo")


class TestDeduplicateChunks(unittest.TestCase):

    def test_deduplication_removes_duplicates(self):
        chunks = [
            {"chunk_id": "a", "score": 0.9},
            {"chunk_id": "b", "score": 0.8},
            {"chunk_id": "a", "score": 0.7},  # duplicate
        ]
        result = _deduplicate_chunks(chunks)
        self.assertEqual(len(result), 2)
        ids = [c["chunk_id"] for c in result]
        self.assertEqual(ids, ["a", "b"])

    def test_no_duplicates_unchanged(self):
        chunks = [
            {"chunk_id": "a", "score": 0.9},
            {"chunk_id": "b", "score": 0.8},
        ]
        result = _deduplicate_chunks(chunks)
        self.assertEqual(len(result), 2)

    def test_empty_chunk_id_all_kept(self):
        """Chunks with empty chunk_id are all kept (can't dedup without ID)."""
        chunks = [
            {"chunk_id": "", "score": 0.9},
            {"chunk_id": "", "score": 0.8},
        ]
        result = _deduplicate_chunks(chunks)
        self.assertEqual(len(result), 2)


class TestBuildSources(unittest.TestCase):

    def test_sources_contain_expected_fields(self):
        sources = _build_sources(SAMPLE_CHUNKS)
        self.assertEqual(len(sources), 2)
        for src in sources:
            self.assertIn("chunk_id", src)
            self.assertIn("file_path", src)
            self.assertIn("language", src)
            self.assertIn("start_line", src)
            self.assertIn("end_line", src)
            self.assertIn("score", src)

    def test_sources_exclude_content(self):
        """Sources should NOT include raw content to keep response lean."""
        sources = _build_sources(SAMPLE_CHUNKS)
        for src in sources:
            self.assertNotIn("content", src)


class TestExtractRepoName(unittest.TestCase):

    def test_extracts_owner_repo(self):
        self.assertEqual(_extract_repo_name("https://github.com/owner/repo"), "owner/repo")

    def test_fallback_on_bad_url(self):
        self.assertEqual(_extract_repo_name("not-a-url"), "not-a-url")


class TestAnswerQuestion(unittest.TestCase):

    def setUp(self):
        self._mock_model = _make_mock_model()
        set_embedding_model(self._mock_model)
        self._mock_llm = _make_mock_llm("## Summary\n\nAuthentication is in auth.py.")
        set_llm_provider(self._mock_llm)

    def tearDown(self):
        set_embedding_model(None)
        set_llm_provider(None)

    def _mock_vector_search(self, results):
        patcher = patch("services.rag.default_vector_store.search", return_value=results)
        return patcher

    def test_successful_answer(self):
        """Full pipeline returns answer and sources."""
        with self._mock_vector_search(SAMPLE_CHUNKS):
            result = answer_question(
                repo_url="https://github.com/owner/repo",
                question="How does auth work?",
                top_k=5,
            )
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 0)

    def test_sources_match_chunks(self):
        """Sources returned match the chunks retrieved from the vector store."""
        with self._mock_vector_search(SAMPLE_CHUNKS):
            result = answer_question(
                repo_url="https://github.com/owner/repo",
                question="How does auth work?",
                top_k=5,
            )
        self.assertEqual(len(result["sources"]), len(SAMPLE_CHUNKS))
        self.assertEqual(result["sources"][0]["file_path"], "backend/auth.py")

    def test_empty_search_results(self):
        """Pipeline still calls LLM with empty context and returns answer."""
        with self._mock_vector_search([]):
            result = answer_question(
                repo_url="https://github.com/owner/repo",
                question="How does auth work?",
            )
        self.assertIn("answer", result)
        self.assertEqual(result["sources"], [])
        self._mock_llm.generate.assert_called_once()

    def test_repository_isolation(self):
        """Vector store is searched with the correct normalised repo URL."""
        mock_search = MagicMock(return_value=SAMPLE_CHUNKS)
        with patch("services.rag.default_vector_store.search", mock_search):
            answer_question(
                repo_url="https://github.com/owner/repo",
                question="question",
            )
        call_kwargs = mock_search.call_args[1]
        self.assertEqual(call_kwargs["repo_url"], "https://github.com/owner/repo")

    def test_invalid_repo_url_raises_value_error(self):
        """Invalid URL raises ValueError before reaching embedding or LLM."""
        with self.assertRaises(ValueError):
            answer_question(
                repo_url="not-a-github-url",
                question="question",
            )

    def test_empty_question_raises_value_error(self):
        """Empty question raises ValueError."""
        with self.assertRaises(ValueError):
            answer_question(
                repo_url="https://github.com/owner/repo",
                question="",
            )

    def test_top_k_capped_at_max(self):
        """top_k is capped at _MAX_CHUNKS regardless of input."""
        mock_search = MagicMock(return_value=[])
        with patch("services.rag.default_vector_store.search", mock_search):
            answer_question(
                repo_url="https://github.com/owner/repo",
                question="question",
                top_k=999,  # Way over the cap
            )
        call_kwargs = mock_search.call_args[1]
        from services.rag import _MAX_CHUNKS
        self.assertLessEqual(call_kwargs["top_k"], _MAX_CHUNKS)

    def test_duplicate_chunks_deduplicated(self):
        """Duplicate chunk IDs are deduplicated before LLM call."""
        dup_chunks = SAMPLE_CHUNKS + [SAMPLE_CHUNKS[0]]  # Add duplicate
        with self._mock_vector_search(dup_chunks):
            result = answer_question(
                repo_url="https://github.com/owner/repo",
                question="question",
            )
        # Sources should be deduplicated
        source_ids = [s["chunk_id"] for s in result["sources"]]
        self.assertEqual(len(source_ids), len(set(source_ids)))

    def test_embedding_error_raises_runtime_error(self):
        """Embedding failure raises RuntimeError."""
        self._mock_model.encode.side_effect = Exception("Embedding failed")
        with self._mock_vector_search([]):
            with self.assertRaises(RuntimeError) as ctx:
                answer_question(
                    repo_url="https://github.com/owner/repo",
                    question="question",
                )
        self.assertIn("embedding", str(ctx.exception).lower())

    def test_vector_search_error_raises_runtime_error(self):
        """Vector store failure raises RuntimeError."""
        with patch("services.rag.default_vector_store.search", side_effect=Exception("store error")):
            with self.assertRaises(RuntimeError) as ctx:
                answer_question(
                    repo_url="https://github.com/owner/repo",
                    question="question",
                )
        self.assertIn("search", str(ctx.exception).lower())

    def test_llm_error_propagates(self):
        """LLM failure propagates as RuntimeError."""
        self._mock_llm.generate.side_effect = RuntimeError("LLM provider error: timeout")
        with self._mock_vector_search(SAMPLE_CHUNKS):
            with self.assertRaises(RuntimeError) as ctx:
                answer_question(
                    repo_url="https://github.com/owner/repo",
                    question="question",
                )
        self.assertIn("LLM", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
