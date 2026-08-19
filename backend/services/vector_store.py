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


_STOP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "the", "and", "or", "not", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "should", "would",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "too", "very", "will", "just",
    "handles", "handle", "request", "requests", "return", "returns", "code",
    "show", "find", "tell", "explain", "where", "which",
}

_PATH_RE = re.compile(r"(/[a-zA-Z0-9_\-./]+)")
_FILENAME_RE = re.compile(r"\b([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]{1,8})\b")
_CODE_IDENTIFIER_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b")
_QUOTED_RE = re.compile(r"[`'\"]([^`'\"]+)[`'\"]")


def _extract_query_identifiers(query: str) -> List[str]:
    """
    Extracts distinct code identifiers (API paths, filenames, snake_case/camelCase
    tokens, and quoted terms) from a natural language query.
    """
    if not query:
        return []

    identifiers: List[str] = []
    seen = set()

    def _add(token: str):
        t = token.strip()
        if t and len(t) >= 2 and t.lower() not in seen and t.lower() not in _STOP_WORDS:
            seen.add(t.lower())
            identifiers.append(t)

    # 1. Quoted/backticked strings
    for m in _QUOTED_RE.finditer(query):
        _add(m.group(1))

    # 2. Paths (e.g. /api/chat, /v1/index)
    for m in _PATH_RE.finditer(query):
        val = m.group(1).rstrip("?.,;!):")
        if len(val) >= 2:
            _add(val)

    # 3. Filenames (e.g. app.py, vector_store.py)
    for m in _FILENAME_RE.finditer(query):
        _add(m.group(1))

    # 4. Code identifiers (e.g. function_name, ClassName)
    for m in _CODE_IDENTIFIER_RE.finditer(query):
        token = m.group(1)
        # Check if identifier has distinctive code casing/characters (underscore or mixed case)
        if "_" in token or any(c.isupper() for c in token[1:]) or token.isupper():
            _add(token)
        elif len(token) >= 4 and token.lower() not in _STOP_WORDS:
            _add(token)

    return identifiers


def _compute_identifier_boost(
    query_text: Optional[str],
    metadata: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Calculates exact-match relevance boosts for each chunk based on extracted query identifiers.
    """
    n = len(metadata)
    boosts = np.zeros(n, dtype=np.float32)
    if not query_text or n == 0:
        return boosts

    identifiers = _extract_query_identifiers(query_text)
    if not identifiers:
        return boosts

    for i, chunk in enumerate(metadata):
        content = chunk.get("content", "")
        file_path = chunk.get("file_path", "")
        chunk_boost = 0.0

        for ident in identifiers:
            ident_lower = ident.lower()
            content_lower = content.lower()
            file_path_lower = file_path.lower()

            # Exact path or route match (e.g. /api/chat)
            if ident.startswith("/"):
                if ident in content or ident_lower in content_lower:
                    chunk_boost += 0.25
                elif ident in file_path or ident_lower in file_path_lower:
                    chunk_boost += 0.15

            # Filename match (e.g. app.py, vector_store.py)
            elif "." in ident and (ident_lower == Path(file_path).name.lower() or ident_lower in file_path_lower):
                chunk_boost += 0.20

            # Function / route / class definition or symbol match
            else:
                # Direct presence in file path
                if ident_lower in file_path_lower:
                    chunk_boost += 0.10

                # Def or class or route declaration in content
                def_patterns = [
                    f"def {ident}",
                    f"class {ident}",
                    f"@{ident}",
                    f'"{ident}"',
                    f"'{ident}'",
                    f"`{ident}`",
                    f"/{ident}",
                ]
                if any(p.lower() in content_lower for p in def_patterns):
                    chunk_boost += 0.20
                elif re.search(rf"\b{re.escape(ident)}\b", content, re.IGNORECASE):
                    chunk_boost += 0.08

        # Cap boost per chunk so semantic similarity remains the primary foundation
        boosts[i] = min(0.35, chunk_boost)

    return boosts


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
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search against the repository index, enhanced
        with exact identifier boosting when queries reference specific paths, filenames, or symbols.

        Args:
            repo_url: Repository identifier to search within.
            query_embedding: 1D numpy array representing the normalized query vector.
            top_k: Maximum number of ranked results to return.
            query_text: Optional original text query for identifier boosting.

        Returns:
            List of matching records ordered by descending relevance score.
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

        # Cosine similarity for L2-normalized vectors
        emb_norms = np.linalg.norm(embeddings, axis=1)
        q_norm = np.linalg.norm(q_vec)

        if q_norm == 0:
            return []

        safe_norms = np.where(emb_norms == 0, 1e-10, emb_norms)
        similarities = np.dot(embeddings, q_vec) / (safe_norms * q_norm)
        similarities = np.clip(similarities, -1.0, 1.0)

        # Apply exact identifier boost if query text is provided
        if query_text:
            boosts = _compute_identifier_boost(query_text, metadata)
            final_scores = np.clip(similarities + boosts, -1.0, 1.0)
        else:
            final_scores = similarities

        # Filter completely negative correlation (noise)
        valid_mask = final_scores >= 0.0
        if not np.any(valid_mask):
            # If all are negative, take top result if any exists
            valid_indices = np.arange(len(final_scores))
        else:
            valid_indices = np.where(valid_mask)[0]

        # Rank valid results descending by score
        ranked_order = np.argsort(-final_scores[valid_indices])
        sorted_indices = valid_indices[ranked_order]

        k = min(top_k, len(sorted_indices))
        if k <= 0:
            return []

        top_indices = sorted_indices[:k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(final_scores[idx])
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
