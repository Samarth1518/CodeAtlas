"""
services/embeddings.py — Local semantic embedding service using sentence-transformers.

Uses the lightweight, high-performance 'all-MiniLM-L6-v2' model (384-dimensional vectors).
Loads the model lazily on first inference so Flask startup is fast.
All inference is performed locally without transmitting repository code or tokens externally.

Memory Optimization for Constrained Environments (e.g., Render free tier):
  - Token sequence length limited to 256 tokens.
  - Batched encoding with batch_size=8 to prevent spike memory allocation.
  - Evaluation mode (model.eval()) and torch.inference_mode() during encoding.
"""

from typing import Any, Dict, List, Optional
import numpy as np

# ── Configuration & Limits ───────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384
MAX_SEQ_LENGTH: int = 256
DEFAULT_BATCH_SIZE: int = 8

# Module-level singleton instance for lazy loading
_model_instance: Optional[Any] = None


def get_embedding_model() -> Any:
    """
    Lazily loads and returns the SentenceTransformer model instance.
    Cached as a singleton to avoid redundant weight reloading.
    """
    global _model_instance
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            # Limit sequence length to 256 tokens to control memory footprint
            if hasattr(model, "max_seq_length"):
                model.max_seq_length = MAX_SEQ_LENGTH
            if hasattr(model, "eval"):
                model.eval()
            _model_instance = model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sentence-transformers model '{EMBEDDING_MODEL_NAME}': {exc}"
            ) from exc
    return _model_instance


def set_embedding_model(model: Optional[Any]) -> None:
    """Explicitly inject a model instance (useful for unit testing with mocks)."""
    global _model_instance
    _model_instance = model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generates normalized dense embeddings for a list of strings.
    Processes texts in small batches (batch_size=8) and runs in torch inference
    mode to prevent out-of-memory errors on resource-constrained hosts.

    Args:
        texts: List of text strings to embed.

    Returns:
        2D numpy array of shape (N, 384) with dtype float32, L2-normalized.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    # Filter/clean strings
    clean_texts = [str(t) if t is not None else "" for t in texts]

    model = get_embedding_model()
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = MAX_SEQ_LENGTH

    # Use torch.inference_mode() when available for zero-overhead inference
    try:
        import torch
        inference_context = torch.inference_mode()
    except (ImportError, AttributeError):
        import contextlib
        inference_context = contextlib.nullcontext()

    try:
        with inference_context:
            # normalize_embeddings=True ensures cosine similarity is equivalent to dot product
            # batch_size=8 bounds working memory during tensor operations
            embeddings = model.encode(
                clean_texts,
                batch_size=DEFAULT_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception as exc:
        raise RuntimeError(f"Error during batch text embedding: {exc}") from exc


def embed_query(query: str) -> np.ndarray:
    """
    Generates a normalized dense embedding vector for a search query.
    Reuses the safe, memory-optimized embed_texts path.

    Args:
        query: Query string.

    Returns:
        1D numpy array of shape (384,) with dtype float32, L2-normalized.
    """
    clean_query = str(query).strip()
    if not clean_query:
        raise ValueError("Query string cannot be empty.")

    batch = embed_texts([clean_query])
    return batch[0]


def embed_chunks(chunks: List[Dict[str, Any]]) -> np.ndarray:
    """
    Generates embeddings for a list of CodeChunk dictionaries.
    Constructs an informative text payload per chunk (file path + language + content).

    Args:
        chunks: List of code chunk dictionaries (each containing 'content', 'file_path', 'language').

    Returns:
        2D numpy array of shape (len(chunks), 384) with dtype float32.
    """
    if not chunks:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    text_payloads: List[str] = []
    for c in chunks:
        file_path = c.get("file_path", "")
        language = c.get("language") or ""
        content = c.get("content", "")
        # Prefix with file path context for enhanced retrieval accuracy
        header = f"File: {file_path}"
        if language:
            header += f" ({language})"
        text_payloads.append(f"{header}\n\n{content}")

    return embed_texts(text_payloads)
