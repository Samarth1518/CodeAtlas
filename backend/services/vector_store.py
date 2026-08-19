"""
services/vector_store.py — Local, persistent vector store using NumPy for cosine similarity search.

Stores normalized embedding matrices (embeddings.npy) separately from chunk metadata
(metadata.json) per repository in a local storage directory.
Provides in-memory caching and persistence across Flask application restarts.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Storage path relative to backend root
DEFAULT_STORAGE_DIR: Path = Path(__file__).resolve().parent.parent / "storage" / "vector_store"


def _normalize_repo_key(repo_url: str) -> str:
    """Normalizes a GitHub repo URL or name into a safe filesystem-friendly folder key."""
    clean = re.sub(r"^https?://github\.com/", "", repo_url.strip(), flags=re.IGNORECASE)
    clean = clean.strip("/").removesuffix(".git")
    # Replace non-alphanumeric characters with underscore
    return re.sub(r"[^A-Za-z0-9._-]", "_", clean).lower()


class LocalVectorStore:
    """
    Local persistent vector store managing embedding matrices and metadata per repository.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir: Path = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # In-memory cache: repo_key -> {"embeddings": np.ndarray, "metadata": List[Dict]}
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_repo_dir(self, repo_url: str) -> Path:
        key = _normalize_repo_key(repo_url)
        repo_dir = self.storage_dir / key
        repo_dir.mkdir(parents=True, exist_ok=True)
        return repo_dir

    def build_index(
        self,
        repo_url: str,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Creates or replaces the vector index for a repository.

        Args:
            repo_url: Full URL or name of the repository.
            chunks: List of code chunk dictionaries.
            embeddings: 2D numpy array of shape (len(chunks), D) with normalized vectors.

        Returns:
            Dictionary with index summary stats.
        """
        key = _normalize_repo_key(repo_url)
        repo_dir = self._get_repo_dir(repo_url)

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings row count ({len(embeddings)})."
            )

        # Standardize metadata records
        clean_metadata: List[Dict[str, Any]] = []
        unique_files = set()
        lang_counts: Dict[str, int] = {}

        for c in chunks:
            file_path = c.get("file_path", "")
            unique_files.add(file_path)
            lang = c.get("language") or "Other"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            clean_metadata.append(
                {
                    "chunk_id": c.get("chunk_id", ""),
                    "repo_url": repo_url,
                    "file_path": file_path,
                    "language": c.get("language"),
                    "start_line": int(c.get("start_line", 1)),
                    "end_line": int(c.get("end_line", 1)),
                    "content": c.get("content", ""),
                    "line_count": int(c.get("line_count", 0)),
                    "char_count": int(c.get("char_count", len(c.get("content", "")))),
                }
            )

        embeddings_arr = np.asarray(embeddings, dtype=np.float32)

        # Ensure embeddings are 2D
        if embeddings_arr.ndim == 1:
            embeddings_arr = embeddings_arr.reshape(1, -1)

        # 1. Persist embeddings.npy
        emb_file = repo_dir / "embeddings.npy"
        np.save(emb_file, embeddings_arr)

        # 2. Persist metadata.json
        meta_file = repo_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(clean_metadata, f, indent=2, ensure_ascii=False)

        # 3. Update memory cache
        self._cache[key] = {
            "embeddings": embeddings_arr,
            "metadata": clean_metadata,
        }

        return {
            "files_processed": len(unique_files),
            "chunks_indexed": len(clean_metadata),
            "languages": lang_counts,
        }

    def _load_repo_index(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """Loads index from in-memory cache or local disk if available."""
        key = _normalize_repo_key(repo_url)
        if key in self._cache:
            return self._cache[key]

        repo_dir = self.storage_dir / key
        emb_file = repo_dir / "embeddings.npy"
        meta_file = repo_dir / "metadata.json"

        if not emb_file.exists() or not meta_file.exists():
            return None

        try:
            embeddings = np.load(emb_file)
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if len(embeddings) != len(metadata):
                return None

            entry = {"embeddings": embeddings, "metadata": metadata}
            self._cache[key] = entry
            return entry
        except Exception:
            return None

    def search(
        self,
        repo_url: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search against the repository index.

        Args:
            repo_url: Repository identifier to search within.
            query_embedding: 1D numpy array representing the normalized query vector.
            top_k: Maximum number of ranked results to return.

        Returns:
            List of matching records ordered by descending similarity score.
        """
        index_data = self._load_repo_index(repo_url)
        if not index_data:
            return []

        embeddings = index_data["embeddings"]  # Shape (N, D)
        metadata = index_data["metadata"]      # Length N

        if len(embeddings) == 0:
            return []

        q_vec = np.asarray(query_embedding, dtype=np.float32).flatten()
        if q_vec.ndim != 1 or len(q_vec) == 0:
            return []

        # Cosine similarity for L2-normalized vectors: dot product
        # If norms might differ, explicitly compute cosine similarity:
        emb_norms = np.linalg.norm(embeddings, axis=1)
        q_norm = np.linalg.norm(q_vec)

        if q_norm == 0:
            return []

        # Avoid divide-by-zero
        safe_norms = np.where(emb_norms == 0, 1e-10, emb_norms)
        similarities = np.dot(embeddings, q_vec) / (safe_norms * q_norm)
        # Clip numerical precision artifacts to [-1.0, 1.0]
        similarities = np.clip(similarities, -1.0, 1.0)

        # Get top-k indices sorted descending
        k = min(top_k, len(similarities))
        if k <= 0:
            return []

        top_indices = np.argsort(-similarities)[:k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(similarities[idx])
            # Round score to 4 decimals for clean API response
            clean_score = round(score, 4)
            record = metadata[idx]

            results.append(
                {
                    "score": clean_score,
                    "chunk_id": record.get("chunk_id", ""),
                    "file_path": record.get("file_path", ""),
                    "language": record.get("language"),
                    "start_line": record.get("start_line", 1),
                    "end_line": record.get("end_line", 1),
                    "content": record.get("content", ""),
                    "line_count": record.get("line_count", 0),
                    "char_count": record.get("char_count", 0),
                }
            )

        return results

    def clear(self, repo_url: Optional[str] = None) -> None:
        """Clears in-memory cache and/or storage directory."""
        if repo_url:
            key = _normalize_repo_key(repo_url)
            self._cache.pop(key, None)
        else:
            self._cache.clear()


# Module-level default vector store instance
default_vector_store: LocalVectorStore = LocalVectorStore()
